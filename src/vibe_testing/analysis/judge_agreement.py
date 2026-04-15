"""
LLM-judge agreement analysis utilities for Stage 6.

This module quantifies how consistent multiple LLM judges are when evaluating the
same pairwise-comparison samples (Stage 5b artifacts). It computes agreement at:

1) Sample-level overall outcomes (A/B/tie, represented as row/col/tie after
   direction normalization).
2) Per-vibe-dimension outcomes (same categorical winner labels).

It also estimates stability of model rankings across judges based on joint
preference win-rate tables.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS

LOGGER = logging.getLogger(__name__)

# Normalized categorical outcomes from the row-model's perspective.
OUTCOME_ROW = "row"
OUTCOME_COL = "col"
OUTCOME_TIE = "tie"
OUTCOME_CATEGORIES = (OUTCOME_ROW, OUTCOME_COL, OUTCOME_TIE)


class JudgeAgreementError(RuntimeError):
    """Raised when judge agreement analysis encounters invalid or ambiguous data."""


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap confidence interval container."""

    lower: float
    upper: float


def compute_judge_agreement_for_joint_preference(
    pairwise_df: pd.DataFrame,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 42,
    judge_column: str = "judge_model_name",
) -> Dict[str, pd.DataFrame]:
    """
    Compute judge agreement and stability artifacts for Stage-6 joint preference outputs.

    This is the main entry point used by `scripts/stage_6_analyze_results.py`.

    Args:
        pairwise_df (pd.DataFrame): Output of `AnalysisInputLoader.load_pairwise_results()`.
            Must include judge identity, per-sample overall winner label, and per-dimension
            winner labels.
        n_bootstrap (int): Number of bootstrap resamples for confidence intervals.
        seed (int): RNG seed used for bootstrap resampling.
        judge_column (str): Column name storing judge model identifier.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of artifacts:
            - agreement_condition_summary: per (persona,prompt_type,row_model,col_model)
              agreement metrics (overall + per-dimension) including bootstrap CIs.
            - agreement_judge_pairs_overall: per-condition judge-pair agreement stats (overall).
            - agreement_judge_pairs_by_dimension: per-condition judge-pair agreement stats (per dim).
            - ranking_stability: per (persona,prompt_type) judge-induced ranking stability stats.
            - agreement_summary: compact summary across all conditions (overall + per dim).

    Raises:
        JudgeAgreementError: If required columns are missing, or duplicate/conflicting
            ratings are detected for the same (item, judge, condition).
    """
    if pairwise_df is None or pairwise_df.empty:
        raise JudgeAgreementError(
            "pairwise_df is empty; cannot compute judge agreement."
        )
    if int(n_bootstrap) <= 0:
        raise JudgeAgreementError(f"n_bootstrap must be > 0. Got: {n_bootstrap}")

    required = [
        "user_id",
        "variant_label",
        "task_id",
        "variant_id",
        "model_a_name",
        "model_b_name",
        "overall_winner_label",
        judge_column,
    ]
    missing = [c for c in required if c not in pairwise_df.columns]
    if missing:
        raise JudgeAgreementError(
            "pairwise_df is missing required columns for judge agreement: "
            + ", ".join(missing)
        )

    # Ensure dimension-level winner labels are present when available; skip missing dims gracefully.
    dim_cols = [f"dim_{dim}_winner_label" for dim in PAIRWISE_DIMENSIONS]
    dims_present = [
        dim for dim, col in zip(PAIRWISE_DIMENSIONS, dim_cols) if col in pairwise_df
    ]
    if not dims_present:
        raise JudgeAgreementError(
            "pairwise_df is missing per-dimension winner label columns "
            "(expected columns like 'dim_clarity_winner_label')."
        )

    directional = _build_directional_outcome_frame(
        pairwise_df,
        judge_column=judge_column,
        dimensions=dims_present,
    )
    if directional.empty:
        raise JudgeAgreementError(
            "Directional outcome frame is empty after normalization."
        )

    # Compute per-condition agreement summaries.
    condition_summary_rows: List[Dict[str, object]] = []
    judge_pair_rows_overall: List[Dict[str, object]] = []
    judge_pair_rows_dim: List[Dict[str, object]] = []

    rng = np.random.default_rng(int(seed))

    group_keys = ["persona", "prompt_type", "row_model", "col_model"]
    for keys, grp in directional.groupby(group_keys, dropna=False):
        context = dict(zip(group_keys, keys if isinstance(keys, tuple) else [keys]))
        judges = sorted(grp[judge_column].dropna().astype(str).unique().tolist())
        n_judges = int(len(judges))

        record: Dict[str, object] = {
            **context,
            "judge_agreement_n_judges": n_judges,
            "judge_agreement_n_items_total": int(grp["item_id"].nunique()),
        }

        if n_judges < 2:
            # Not enough judges to estimate agreement; keep schema stable with NA.
            record.update(_empty_agreement_metrics(prefix="judge_agreement_overall"))
            for dim in dims_present:
                record.update(
                    _empty_agreement_metrics(prefix=f"judge_agreement_dim_{dim}")
                )
            condition_summary_rows.append(record)
            continue

        # Overall agreement
        overall_table = _build_item_judge_table(
            grp,
            outcome_col="outcome_overall",
            judge_column=judge_column,
        )
        overall_metrics, overall_pairs = _compute_agreement_metrics_for_table(
            overall_table,
            rng=rng,
            n_bootstrap=int(n_bootstrap),
        )
        record.update(_prefix_metrics(overall_metrics, "judge_agreement_overall"))
        for row in overall_pairs:
            judge_pair_rows_overall.append({**context, **row})

        # Per-dimension agreement
        for dim in dims_present:
            outcome_col = f"outcome_dim_{dim}"
            dim_table = _build_item_judge_table(
                grp,
                outcome_col=outcome_col,
                judge_column=judge_column,
            )
            dim_metrics, dim_pairs = _compute_agreement_metrics_for_table(
                dim_table,
                rng=rng,
                n_bootstrap=int(n_bootstrap),
            )
            record.update(_prefix_metrics(dim_metrics, f"judge_agreement_dim_{dim}"))
            for row in dim_pairs:
                judge_pair_rows_dim.append({**context, "dimension": dim, **row})

        condition_summary_rows.append(record)

    agreement_condition_summary = pd.DataFrame(condition_summary_rows)
    agreement_judge_pairs_overall = pd.DataFrame(judge_pair_rows_overall)
    agreement_judge_pairs_by_dimension = pd.DataFrame(judge_pair_rows_dim)

    ranking_stability = _compute_ranking_stability_from_pairwise(
        directional,
        judge_column=judge_column,
    )

    agreement_summary = _summarize_agreement_frames(
        agreement_condition_summary,
        dims_present=dims_present,
    )

    return {
        "agreement_condition_summary": agreement_condition_summary,
        "agreement_judge_pairs_overall": agreement_judge_pairs_overall,
        "agreement_judge_pairs_by_dimension": agreement_judge_pairs_by_dimension,
        "ranking_stability": ranking_stability,
        "agreement_summary": agreement_summary,
    }


def compute_pooled_judge_agreement_metrics(
    item_judge_table: pd.DataFrame,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 42,
    condition_ids: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """
    Compute pooled judge-agreement metrics for a pre-built item x judge table.

    This helper is intended for paper/table exports that pool multiple evaluation
    slices into one subset (for example, all items for a prompt type or persona).
    It reuses the same agreement logic as the main Stage-6 judge-agreement
    pipeline and additionally reports Fleiss' kappa for each judge pair's
    two-judge table so callers can summarize Fleiss-compatible kappas as
    ``mean +/- std`` across judge pairs.

    Args:
        item_judge_table (pd.DataFrame): Rows are items and columns are judges.
            Cell values must be categorical outcomes in ``OUTCOME_CATEGORIES`` or
            missing values.
        n_bootstrap (int): Number of bootstrap resamples for CI estimation.
        seed (int): RNG seed used for bootstrap resampling.
        condition_ids (Optional[pandas.Series]): Optional condition identifier per
            row of ``item_judge_table``. When provided, the reported mean/std are
            computed across conditions and weighted by the number of items in each
            condition.

    Returns:
        Dict[str, object]: Dictionary with keys:
            - ``summary``: aggregate metrics dictionary
            - ``judge_pair_rows``: per-judge-pair DataFrame

    Raises:
        JudgeAgreementError: If inputs are invalid.
    """
    if item_judge_table is None or item_judge_table.empty:
        return {
            "summary": {
                **_empty_agreement_metrics(prefix=""),
                "percent_agreement_std_pairs": np.nan,
                "fleiss_kappa_mean_pairs": np.nan,
                "fleiss_kappa_std_pairs": np.nan,
            },
            "judge_pair_rows": pd.DataFrame(),
        }
    if int(n_bootstrap) <= 0:
        raise JudgeAgreementError(f"n_bootstrap must be > 0. Got: {n_bootstrap}")

    if condition_ids is not None:
        aligned_condition_ids = pd.Series(condition_ids).reindex(item_judge_table.index)
        if aligned_condition_ids.isna().any():
            missing = int(aligned_condition_ids.isna().sum())
            raise JudgeAgreementError(
                "condition_ids is missing values for pooled judge-agreement items. "
                f"missing_items={missing}"
            )

        condition_summaries: List[Dict[str, object]] = []
        enriched_pair_frames: List[pd.DataFrame] = []
        for condition_value, item_ids in aligned_condition_ids.groupby(
            aligned_condition_ids
        ):
            condition_table = item_judge_table.loc[item_ids.index].copy()
            if condition_table.empty:
                continue
            condition_metrics = compute_pooled_judge_agreement_metrics(
                condition_table,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            summary = dict(condition_metrics["summary"])
            summary["condition_id"] = str(condition_value)
            condition_summaries.append(summary)

            pair_rows = condition_metrics.get("judge_pair_rows")
            if isinstance(pair_rows, pd.DataFrame) and not pair_rows.empty:
                pair_rows = pair_rows.copy()
                pair_rows["condition_id"] = str(condition_value)
                enriched_pair_frames.append(pair_rows)

        if not condition_summaries:
            return {
                "summary": {
                    **_empty_agreement_metrics(prefix=""),
                    "percent_agreement_std_pairs": np.nan,
                    "fleiss_kappa_mean_pairs": np.nan,
                    "fleiss_kappa_std_pairs": np.nan,
                },
                "judge_pair_rows": pd.DataFrame(),
            }

        condition_summary_df = pd.DataFrame(condition_summaries)
        weights = condition_summary_df["n_items_any"].astype(float).to_numpy()
        pct_values = (
            condition_summary_df["percent_agreement_mean_pairs"]
            .astype(float)
            .to_numpy()
        )
        cohen_values = (
            condition_summary_df["cohen_kappa_mean_pairs"].astype(float).to_numpy()
        )
        fleiss_values = (
            condition_summary_df["fleiss_kappa_mean_pairs"].astype(float).to_numpy()
        )

        summary = {
            "n_items_any": int(float(np.sum(weights))),
            "n_judges": int(item_judge_table.shape[1]),
            "n_judge_pairs": int(
                max(
                    (int(v) for v in condition_summary_df["n_judge_pairs"].dropna()),
                    default=0,
                )
            ),
            "pair_overlap_items_mean": _weighted_mean(
                condition_summary_df["pair_overlap_items_mean"], weights
            ),
            "pair_overlap_items_min": float(
                condition_summary_df["pair_overlap_items_min"].min()
            ),
            "percent_agreement_mean_pairs": _weighted_mean(pct_values, weights),
            "percent_agreement_std_pairs": _weighted_std(pct_values, weights),
            "cohen_kappa_mean_pairs": _weighted_mean(cohen_values, weights),
            "fleiss_kappa_mean_pairs": _weighted_mean(fleiss_values, weights),
            "fleiss_kappa_std_pairs": _weighted_std(fleiss_values, weights),
            "n_conditions_used": int(len(condition_summary_df)),
        }
        judge_pair_rows = (
            pd.concat(enriched_pair_frames, ignore_index=True)
            if enriched_pair_frames
            else pd.DataFrame()
        )
        return {
            "summary": summary,
            "judge_pair_rows": judge_pair_rows,
        }

    rng = np.random.default_rng(int(seed))
    metrics, judge_pair_rows = _compute_agreement_metrics_for_table(
        item_judge_table,
        rng=rng,
        n_bootstrap=int(n_bootstrap),
    )

    enriched_pair_rows: List[Dict[str, object]] = []
    pct_values: List[float] = []
    fleiss_values: List[float] = []
    for row in judge_pair_rows:
        judge_a = str(row["judge_a"])
        judge_b = str(row["judge_b"])
        pair_table = item_judge_table[[judge_a, judge_b]].copy()
        pair_table = pair_table.dropna(how="all")
        fleiss = _fleiss_kappa_complete(
            pair_table,
            categories=list(OUTCOME_CATEGORIES),
        )["kappa"]
        enriched = dict(row)
        enriched["fleiss_kappa"] = fleiss
        enriched_pair_rows.append(enriched)

        pct = row.get("percent_agreement")
        if pct is not None and not pd.isna(pct):
            pct_values.append(float(pct))
        if fleiss is not None and not pd.isna(fleiss):
            fleiss_values.append(float(fleiss))

    summary = dict(metrics)
    summary["percent_agreement_std_pairs"] = (
        float(np.std(pct_values, ddof=1)) if len(pct_values) > 1 else np.nan
    )
    summary["fleiss_kappa_mean_pairs"] = (
        float(np.mean(fleiss_values)) if fleiss_values else np.nan
    )
    summary["fleiss_kappa_std_pairs"] = (
        float(np.std(fleiss_values, ddof=1)) if len(fleiss_values) > 1 else np.nan
    )

    return {
        "summary": summary,
        "judge_pair_rows": pd.DataFrame(enriched_pair_rows),
    }


def _weighted_mean(values: Iterable[object], weights: Iterable[object]) -> float:
    """Compute a NaN-aware weighted mean."""
    pairs = []
    for value, weight in zip(values, weights):
        if pd.isna(value) or pd.isna(weight):
            continue
        weight_float = float(weight)
        if weight_float <= 0:
            continue
        pairs.append((float(value), weight_float))
    if not pairs:
        return float("nan")
    total_weight = float(sum(weight for _, weight in pairs))
    return float(sum(value * weight for value, weight in pairs) / total_weight)


def _weighted_std(values: Iterable[object], weights: Iterable[object]) -> float:
    """Compute a NaN-aware weighted population standard deviation."""
    pairs = []
    for value, weight in zip(values, weights):
        if pd.isna(value) or pd.isna(weight):
            continue
        weight_float = float(weight)
        if weight_float <= 0:
            continue
        pairs.append((float(value), weight_float))
    if not pairs:
        return float("nan")
    if len(pairs) == 1:
        return 0.0
    mean = _weighted_mean(
        [value for value, _ in pairs],
        [weight for _, weight in pairs],
    )
    total_weight = float(sum(weight for _, weight in pairs))
    variance = (
        sum(weight * ((value - mean) ** 2) for value, weight in pairs) / total_weight
    )
    return float(math.sqrt(variance))


def _build_directional_outcome_frame(
    pairwise_df: pd.DataFrame,
    *,
    judge_column: str,
    dimensions: Sequence[str],
) -> pd.DataFrame:
    """
    Normalize Stage-5b outcomes to a row-model-centric directional representation.

    Each input comparison contributes two directional rows:
    - (row_model=model_a, col_model=model_b)
    - (row_model=model_b, col_model=model_a)

    Outcomes are converted from {model_a, model_b, tie} into {row, col, tie}.

    Raises:
        JudgeAgreementError: If winner label columns contain unexpected values.
    """
    df = pairwise_df.copy()
    for col in [
        "user_id",
        "variant_label",
        "task_id",
        "variant_id",
        "model_a_name",
        "model_b_name",
        judge_column,
    ]:
        if df[col].isna().any():
            # Fail loudly: null grouping keys produce hard-to-debug merges downstream.
            counts = int(df[col].isna().sum())
            raise JudgeAgreementError(
                f"pairwise_df has {counts} null value(s) in required column '{col}'."
            )

    valid = {"model_a", "model_b", "tie"}
    for col in ["overall_winner_label"] + [f"dim_{d}_winner_label" for d in dimensions]:
        if col not in df.columns:
            continue
        vals = set(df[col].astype(str).unique().tolist())
        bad = sorted(vals - valid)
        if bad:
            raise JudgeAgreementError(
                f"Column '{col}' contains unexpected values: {bad}. Expected: {sorted(valid)}"
            )

    # Stable item id for agreement: base item + variant instance.
    df["item_id"] = (
        df["task_id"].astype(str).str.strip()
        + "::"
        + df["variant_id"].astype(str).str.strip()
    )

    base_cols = [
        "user_id",
        "variant_label",
        "item_id",
        "task_id",
        "variant_id",
        judge_column,
    ]

    def _map_forward(label: str) -> str:
        if label == "model_a":
            return OUTCOME_ROW
        if label == "model_b":
            return OUTCOME_COL
        return OUTCOME_TIE

    def _map_reverse(label: str) -> str:
        # Reverse direction flips row/col roles.
        if label == "model_a":
            return OUTCOME_COL
        if label == "model_b":
            return OUTCOME_ROW
        return OUTCOME_TIE

    forward = df[
        base_cols + ["model_a_name", "model_b_name", "overall_winner_label"]
    ].copy()
    forward = forward.rename(
        columns={
            "user_id": "persona",
            "variant_label": "prompt_type",
            "model_a_name": "row_model",
            "model_b_name": "col_model",
        }
    )
    forward["outcome_overall"] = (
        forward["overall_winner_label"].astype(str).map(_map_forward)
    )
    forward = forward.drop(columns=["overall_winner_label"])

    reverse = df[
        base_cols + ["model_a_name", "model_b_name", "overall_winner_label"]
    ].copy()
    reverse = reverse.rename(
        columns={
            "user_id": "persona",
            "variant_label": "prompt_type",
            "model_b_name": "row_model",
            "model_a_name": "col_model",
        }
    )
    reverse["outcome_overall"] = (
        reverse["overall_winner_label"].astype(str).map(_map_reverse)
    )
    reverse = reverse.drop(columns=["overall_winner_label"])

    # Add dimension outcomes.
    for dim in dimensions:
        col = f"dim_{dim}_winner_label"
        if col not in df.columns:
            continue
        f = df[base_cols + ["model_a_name", "model_b_name", col]].copy()
        f = f.rename(
            columns={
                "user_id": "persona",
                "variant_label": "prompt_type",
                "model_a_name": "row_model",
                "model_b_name": "col_model",
                col: f"outcome_dim_{dim}",
            }
        )
        f[f"outcome_dim_{dim}"] = f[f"outcome_dim_{dim}"].astype(str).map(_map_forward)
        r = df[base_cols + ["model_a_name", "model_b_name", col]].copy()
        r = r.rename(
            columns={
                "user_id": "persona",
                "variant_label": "prompt_type",
                "model_b_name": "row_model",
                "model_a_name": "col_model",
                col: f"outcome_dim_{dim}",
            }
        )
        r[f"outcome_dim_{dim}"] = r[f"outcome_dim_{dim}"].astype(str).map(_map_reverse)
        # Merge these outcomes into forward/reverse frames by stable identifiers.
        join_cols = [
            "persona",
            "prompt_type",
            "item_id",
            "task_id",
            "variant_id",
            judge_column,
            "row_model",
            "col_model",
        ]
        forward = forward.merge(
            f[join_cols + [f"outcome_dim_{dim}"]], on=join_cols, how="left"
        )
        reverse = reverse.merge(
            r[join_cols + [f"outcome_dim_{dim}"]], on=join_cols, how="left"
        )

    out = pd.concat([forward, reverse], ignore_index=True)
    for col in [
        "persona",
        "prompt_type",
        "item_id",
        "row_model",
        "col_model",
        judge_column,
    ]:
        out[col] = out[col].astype(str)
    return out


def _build_item_judge_table(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    judge_column: str,
) -> pd.DataFrame:
    """
    Build an item×judge table of categorical outcomes.

    Raises:
        JudgeAgreementError: If duplicates exist with conflicting labels.
    """
    if outcome_col not in frame.columns:
        raise JudgeAgreementError(f"Missing outcome column '{outcome_col}' in frame.")

    cols = ["item_id", judge_column, outcome_col]
    df = frame[cols].copy()
    df[judge_column] = df[judge_column].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    # Detect duplicates with conflicting labels.
    dup = df.duplicated(subset=["item_id", judge_column], keep=False)
    if dup.any():
        conflicts: List[str] = []
        for (item_id, judge), grp in df[dup].groupby(
            ["item_id", judge_column], dropna=False
        ):
            labels = sorted(set(grp[outcome_col].dropna().astype(str).tolist()))
            if len(labels) > 1:
                conflicts.append(
                    f"item_id={item_id!r}, judge={judge!r}, labels={labels!r}"
                )
        if conflicts:
            raise JudgeAgreementError(
                "Conflicting duplicate ratings detected for the same (item, judge). "
                "Examples: "
                + "; ".join(conflicts[:5])
                + (" (truncated)" if len(conflicts) > 5 else "")
            )
        # Non-conflicting duplicates: keep the first deterministically.
        df = df.drop_duplicates(subset=["item_id", judge_column], keep="first")

    table = df.pivot(index="item_id", columns=judge_column, values=outcome_col)
    # Normalize to known categories + NaN.
    for j in table.columns:
        table[j] = table[j].where(table[j].isin(OUTCOME_CATEGORIES), other=np.nan)
    return table


def _compute_agreement_metrics_for_table(
    item_judge_table: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Compute agreement metrics for a single categorical rating table.

    Returns:
        (metrics_dict, judge_pair_rows)
    """
    if item_judge_table is None or item_judge_table.empty:
        return _empty_agreement_metrics(prefix=""), []

    judges = list(item_judge_table.columns.astype(str))
    if len(judges) < 2:
        return _empty_agreement_metrics(prefix=""), []

    pair_rows: List[Dict[str, object]] = []
    pct_values: List[float] = []
    kappa_values: List[float] = []
    overlap_counts: List[int] = []

    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            ja = judges[i]
            jb = judges[j]
            a = item_judge_table[ja]
            b = item_judge_table[jb]
            mask = a.notna() & b.notna()
            n = int(mask.sum())
            if n <= 0:
                continue
            aa = a[mask].astype(str).to_numpy()
            bb = b[mask].astype(str).to_numpy()
            pct = float(np.mean(aa == bb))
            kappa = _cohens_kappa(aa, bb, categories=list(OUTCOME_CATEGORIES))
            pct_values.append(pct)
            kappa_values.append(kappa)
            overlap_counts.append(n)
            pair_rows.append(
                {
                    "judge_a": str(ja),
                    "judge_b": str(jb),
                    "n_overlap_items": n,
                    "percent_agreement": pct,
                    "cohen_kappa": kappa,
                }
            )

    metrics: Dict[str, object] = {
        "n_items_any": int(item_judge_table.shape[0]),
        "n_judges": int(len(judges)),
        "n_judge_pairs": int(len(pair_rows)),
        "pair_overlap_items_mean": (
            float(np.mean(overlap_counts)) if overlap_counts else np.nan
        ),
        "pair_overlap_items_min": (
            float(np.min(overlap_counts)) if overlap_counts else np.nan
        ),
    }

    # Mean across judge pairs
    metrics["percent_agreement_mean_pairs"] = (
        float(np.mean(pct_values)) if pct_values else np.nan
    )
    metrics["cohen_kappa_mean_pairs"] = (
        float(np.mean(kappa_values)) if kappa_values else np.nan
    )

    # Judge-to-judge stability for the induced win rate (wins / total incl. ties).
    judge_win_rates: List[float] = []
    judge_item_counts: List[int] = []
    for j in judges:
        col = item_judge_table[j]
        rated = col.dropna().astype(str)
        n_rated = int(len(rated))
        if n_rated <= 0:
            continue
        win_rate = float(np.mean(rated.to_numpy() == OUTCOME_ROW))
        judge_win_rates.append(win_rate)
        judge_item_counts.append(n_rated)
    metrics["judge_win_rate_mean"] = (
        float(np.mean(judge_win_rates)) if judge_win_rates else np.nan
    )
    metrics["judge_win_rate_std"] = (
        float(np.std(judge_win_rates, ddof=1)) if len(judge_win_rates) > 1 else np.nan
    )
    metrics["judge_win_rate_min"] = (
        float(np.min(judge_win_rates)) if judge_win_rates else np.nan
    )
    metrics["judge_win_rate_max"] = (
        float(np.max(judge_win_rates)) if judge_win_rates else np.nan
    )
    metrics["judge_win_rate_judge_count_used"] = int(len(judge_win_rates))
    metrics["judge_win_rate_n_items_mean"] = (
        float(np.mean(judge_item_counts)) if judge_item_counts else np.nan
    )

    # Fleiss' kappa on complete-coverage items (all judges present).
    fleiss = _fleiss_kappa_complete(
        item_judge_table, categories=list(OUTCOME_CATEGORIES)
    )
    metrics["fleiss_kappa_complete"] = fleiss["kappa"]
    metrics["fleiss_kappa_complete_n_items"] = fleiss["n_items"]

    # Bootstrap CIs for mean pairwise agreement + mean pairwise kappa + Fleiss (complete items).
    ci_pct = _bootstrap_ci_for_mean_pairwise_metric(
        item_judge_table,
        metric="percent_agreement",
        rng=rng,
        n_bootstrap=int(n_bootstrap),
    )
    ci_kappa = _bootstrap_ci_for_mean_pairwise_metric(
        item_judge_table,
        metric="cohen_kappa",
        rng=rng,
        n_bootstrap=int(n_bootstrap),
    )
    ci_fleiss = _bootstrap_ci_for_fleiss_kappa(
        item_judge_table,
        rng=rng,
        n_bootstrap=int(n_bootstrap),
        categories=list(OUTCOME_CATEGORIES),
    )

    metrics["percent_agreement_mean_pairs_ci_lower"] = ci_pct.lower
    metrics["percent_agreement_mean_pairs_ci_upper"] = ci_pct.upper
    metrics["cohen_kappa_mean_pairs_ci_lower"] = ci_kappa.lower
    metrics["cohen_kappa_mean_pairs_ci_upper"] = ci_kappa.upper
    metrics["fleiss_kappa_complete_ci_lower"] = ci_fleiss.lower
    metrics["fleiss_kappa_complete_ci_upper"] = ci_fleiss.upper

    return metrics, pair_rows


def _cohens_kappa(
    a: np.ndarray,
    b: np.ndarray,
    *,
    categories: Sequence[str],
) -> float:
    """Compute Cohen's kappa for two categorical label arrays."""
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")

    cats = list(categories)
    idx = {c: i for i, c in enumerate(cats)}
    cm = np.zeros((len(cats), len(cats)), dtype=float)
    for x, y in zip(a, b):
        if x not in idx or y not in idx:
            continue
        cm[idx[x], idx[y]] += 1.0
    n = float(cm.sum())
    if n <= 0:
        return float("nan")
    po = float(np.trace(cm) / n)
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if math.isclose(1.0 - pe, 0.0):
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def _fleiss_kappa_complete(
    item_judge_table: pd.DataFrame,
    *,
    categories: Sequence[str],
) -> Dict[str, object]:
    """
    Compute Fleiss' kappa on the subset of items rated by all judges.

    Returns:
        Dict with keys: kappa, n_items, n_judges
    """
    if item_judge_table is None or item_judge_table.empty:
        return {"kappa": np.nan, "n_items": 0, "n_judges": 0}
    judges = list(item_judge_table.columns)
    j = int(len(judges))
    if j < 2:
        return {"kappa": np.nan, "n_items": 0, "n_judges": j}

    complete = item_judge_table.dropna(axis=0, how="any")
    n_items = int(complete.shape[0])
    if n_items <= 0:
        return {"kappa": np.nan, "n_items": 0, "n_judges": j}

    cats = list(categories)
    counts = np.zeros((n_items, len(cats)), dtype=float)
    for i, (_, row) in enumerate(complete.iterrows()):
        for label in row.astype(str).tolist():
            if label in cats:
                counts[i, cats.index(label)] += 1.0

    n = float(j)
    if n < 2:
        return {"kappa": np.nan, "n_items": n_items, "n_judges": j}

    p_i = (np.sum(counts * (counts - 1.0), axis=1)) / (n * (n - 1.0))
    p_bar = float(np.mean(p_i)) if len(p_i) else float("nan")
    p_j = np.sum(counts, axis=0) / (n_items * n)
    p_e = float(np.sum(p_j**2))
    if math.isclose(1.0 - p_e, 0.0):
        return {"kappa": np.nan, "n_items": n_items, "n_judges": j}
    kappa = float((p_bar - p_e) / (1.0 - p_e))
    return {"kappa": kappa, "n_items": n_items, "n_judges": j}


def _bootstrap_ci_for_mean_pairwise_metric(
    item_judge_table: pd.DataFrame,
    *,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence_level: float = 0.95,
) -> BootstrapCI:
    """
    Bootstrap CI for mean pairwise metric across judge pairs.

    We resample items with replacement and recompute the mean across judge pairs
    on overlapping items within the resampled set.
    """
    if item_judge_table is None or item_judge_table.empty:
        return BootstrapCI(lower=float("nan"), upper=float("nan"))

    items = item_judge_table.index.to_numpy()
    if len(items) <= 1:
        return BootstrapCI(lower=float("nan"), upper=float("nan"))

    judges = list(item_judge_table.columns.astype(str))
    if len(judges) < 2:
        return BootstrapCI(lower=float("nan"), upper=float("nan"))

    values = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        sampled = rng.choice(items, size=len(items), replace=True)
        sub = item_judge_table.loc[sampled]
        per_pair: List[float] = []
        for a_idx in range(len(judges)):
            for b_idx in range(a_idx + 1, len(judges)):
                ja = judges[a_idx]
                jb = judges[b_idx]
                a = sub[ja]
                b = sub[jb]
                mask = a.notna() & b.notna()
                n = int(mask.sum())
                if n <= 0:
                    continue
                aa = a[mask].astype(str).to_numpy()
                bb = b[mask].astype(str).to_numpy()
                if metric == "percent_agreement":
                    per_pair.append(float(np.mean(aa == bb)))
                elif metric == "cohen_kappa":
                    per_pair.append(
                        _cohens_kappa(aa, bb, categories=list(OUTCOME_CATEGORIES))
                    )
                else:
                    raise JudgeAgreementError(f"Unknown metric for bootstrap: {metric}")
        values[i] = float(np.mean(per_pair)) if per_pair else float("nan")

    values = values[~np.isnan(values)]
    if len(values) == 0:
        return BootstrapCI(lower=float("nan"), upper=float("nan"))

    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapCI(lower=lower, upper=upper)


def _bootstrap_ci_for_fleiss_kappa(
    item_judge_table: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    categories: Sequence[str],
    confidence_level: float = 0.95,
) -> BootstrapCI:
    """Bootstrap CI for Fleiss' kappa on complete-coverage items."""
    if item_judge_table is None or item_judge_table.empty:
        return BootstrapCI(lower=float("nan"), upper=float("nan"))
    complete = item_judge_table.dropna(axis=0, how="any")
    if complete.shape[0] <= 1:
        return BootstrapCI(lower=float("nan"), upper=float("nan"))

    items = complete.index.to_numpy()
    values = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        sampled = rng.choice(items, size=len(items), replace=True)
        sub = complete.loc[sampled]
        values[i] = float(_fleiss_kappa_complete(sub, categories=categories)["kappa"])

    values = values[~np.isnan(values)]
    if len(values) == 0:
        return BootstrapCI(lower=float("nan"), upper=float("nan"))

    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapCI(lower=lower, upper=upper)


def _compute_ranking_stability_from_pairwise(
    directional: pd.DataFrame,
    *,
    judge_column: str,
) -> pd.DataFrame:
    """
    Compute judge-induced ranking stability per (persona, prompt_type).

    Ranking proxy: for each judge, estimate each model's average win probability
    across opponents based on row-centric outcomes. This avoids requiring the
    joint preference long-by-judge table and keeps the analysis local.
    """
    if directional is None or directional.empty:
        return pd.DataFrame()

    required = [
        "persona",
        "prompt_type",
        "row_model",
        "col_model",
        "item_id",
        judge_column,
    ]
    missing = [c for c in required if c not in directional.columns]
    if missing:
        raise JudgeAgreementError(
            "Directional frame missing required columns for ranking stability: "
            + ", ".join(missing)
        )

    # Compute per judge, per model: average win probability excluding ties.
    # Outcome values are {row,col,tie}; from row_model's perspective, a "row" is a win.
    df = directional.copy()
    df["is_win"] = (df["outcome_overall"] == OUTCOME_ROW).astype(float)
    df["is_loss"] = (df["outcome_overall"] == OUTCOME_COL).astype(float)
    df["is_tie"] = (df["outcome_overall"] == OUTCOME_TIE).astype(float)

    # Use item-level aggregation to keep each item count at most once per judge+pair direction.
    item_group = ["persona", "prompt_type", judge_column, "row_model", "item_id"]
    item_means = (
        df.groupby(item_group, dropna=False)
        .agg(w=("is_win", "mean"), l=("is_loss", "mean"))
        .reset_index()
    )
    # Convert to win probability excluding ties.
    item_means["p_win_excl_ties"] = item_means.apply(
        lambda r: (
            float(r["w"]) / float(r["w"] + r["l"])
            if float(r["w"] + r["l"]) > 0
            else 0.5
        ),
        axis=1,
    )

    model_group = ["persona", "prompt_type", judge_column, "row_model"]
    model_scores = (
        item_means.groupby(model_group, dropna=False)
        .agg(
            avg_win_prob=("p_win_excl_ties", "mean"),
            n_items=("p_win_excl_ties", "size"),
        )
        .reset_index()
        .rename(columns={"row_model": "model_name"})
    )
    if model_scores.empty:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for (persona, prompt_type), slice_df in model_scores.groupby(
        ["persona", "prompt_type"], dropna=False
    ):
        judges = sorted(slice_df[judge_column].dropna().astype(str).unique().tolist())
        if len(judges) < 2:
            continue

        # Build judge->rank series (lower rank value means better).
        rank_map: Dict[str, pd.Series] = {}
        top1: Dict[str, str] = {}
        for j in judges:
            j_df = slice_df[slice_df[judge_column].astype(str) == str(j)].copy()
            if j_df.empty:
                continue
            j_df = j_df.sort_values(
                ["avg_win_prob", "model_name"], ascending=[False, True]
            )
            # Dense rank for stability under ties.
            j_df["rank"] = j_df["avg_win_prob"].rank(method="dense", ascending=False)
            ranks = pd.Series(
                j_df["rank"].to_numpy(), index=j_df["model_name"].astype(str)
            )
            rank_map[str(j)] = ranks
            top1[str(j)] = str(j_df.iloc[0]["model_name"])

        judge_pairs = []
        spearman_values: List[float] = []
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                ja = str(judges[i])
                jb = str(judges[j])
                ra = rank_map.get(ja)
                rb = rank_map.get(jb)
                if ra is None or rb is None:
                    continue
                common = sorted(set(ra.index) & set(rb.index))
                if len(common) < 2:
                    continue
                corr = stats.spearmanr(
                    ra.loc[common].to_numpy(), rb.loc[common].to_numpy()
                ).correlation
                if corr is None or (isinstance(corr, float) and np.isnan(corr)):
                    continue
                spearman_values.append(float(corr))
                judge_pairs.append((ja, jb))

        top1_models = list(top1.values())
        top1_mode = None
        top1_stability = np.nan
        if top1_models:
            # Fraction of judges agreeing with the most common top-1.
            counts: Dict[str, int] = {}
            for m in top1_models:
                counts[m] = counts.get(m, 0) + 1
            top1_mode = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            top1_stability = float(counts[top1_mode] / len(top1_models))

        records.append(
            {
                "persona": str(persona),
                "prompt_type": str(prompt_type),
                "n_judges": int(len(judges)),
                "n_judge_pairs": int(len(judge_pairs)),
                "spearman_rank_corr_mean_pairs": (
                    float(np.mean(spearman_values)) if spearman_values else np.nan
                ),
                "spearman_rank_corr_median_pairs": (
                    float(np.median(spearman_values)) if spearman_values else np.nan
                ),
                "spearman_rank_corr_min_pairs": (
                    float(np.min(spearman_values)) if spearman_values else np.nan
                ),
                "spearman_rank_corr_max_pairs": (
                    float(np.max(spearman_values)) if spearman_values else np.nan
                ),
                "top1_mode_model": top1_mode,
                "top1_mode_fraction": top1_stability,
            }
        )

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["persona", "prompt_type"]).reset_index(drop=True)
    return out


def _summarize_agreement_frames(
    agreement_condition_summary: pd.DataFrame,
    *,
    dims_present: Sequence[str],
) -> pd.DataFrame:
    """Build a compact summary across all conditions (overall + per dimension)."""
    if agreement_condition_summary is None or agreement_condition_summary.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    def _mean(col: str) -> float:
        s = pd.to_numeric(agreement_condition_summary.get(col), errors="coerce")
        return float(s.mean()) if s is not None else float("nan")

    base = {
        "scope": "all_conditions",
        "n_conditions": int(len(agreement_condition_summary)),
    }
    rows.append(
        {
            **base,
            "level": "overall",
            "percent_agreement_mean_pairs_mean": _mean(
                "judge_agreement_overall_percent_agreement_mean_pairs"
            ),
            "cohen_kappa_mean_pairs_mean": _mean(
                "judge_agreement_overall_cohen_kappa_mean_pairs"
            ),
            "fleiss_kappa_complete_mean": _mean(
                "judge_agreement_overall_fleiss_kappa_complete"
            ),
        }
    )
    for dim in dims_present:
        rows.append(
            {
                **base,
                "level": f"dimension:{dim}",
                "percent_agreement_mean_pairs_mean": _mean(
                    f"judge_agreement_dim_{dim}_percent_agreement_mean_pairs"
                ),
                "cohen_kappa_mean_pairs_mean": _mean(
                    f"judge_agreement_dim_{dim}_cohen_kappa_mean_pairs"
                ),
                "fleiss_kappa_complete_mean": _mean(
                    f"judge_agreement_dim_{dim}_fleiss_kappa_complete"
                ),
            }
        )
    return pd.DataFrame(rows)


def _empty_agreement_metrics(*, prefix: str) -> Dict[str, object]:
    """
    Produce a stable set of agreement metric fields.

    When prefix is '', returns raw field names. Otherwise returns prefixed names.
    """
    fields = {
        "n_items_any": pd.NA,
        "n_judges": pd.NA,
        "n_judge_pairs": pd.NA,
        "pair_overlap_items_mean": pd.NA,
        "pair_overlap_items_min": pd.NA,
        "percent_agreement_mean_pairs": pd.NA,
        "percent_agreement_mean_pairs_ci_lower": pd.NA,
        "percent_agreement_mean_pairs_ci_upper": pd.NA,
        "cohen_kappa_mean_pairs": pd.NA,
        "cohen_kappa_mean_pairs_ci_lower": pd.NA,
        "cohen_kappa_mean_pairs_ci_upper": pd.NA,
        "judge_win_rate_mean": pd.NA,
        "judge_win_rate_std": pd.NA,
        "judge_win_rate_min": pd.NA,
        "judge_win_rate_max": pd.NA,
        "judge_win_rate_judge_count_used": pd.NA,
        "judge_win_rate_n_items_mean": pd.NA,
        "fleiss_kappa_complete": pd.NA,
        "fleiss_kappa_complete_n_items": pd.NA,
        "fleiss_kappa_complete_ci_lower": pd.NA,
        "fleiss_kappa_complete_ci_upper": pd.NA,
    }
    if not prefix:
        return dict(fields)
    return {f"{prefix}_{k}": v for k, v in fields.items()}


def _prefix_metrics(metrics: Dict[str, object], prefix: str) -> Dict[str, object]:
    """Prefix metric dict keys for safe merging into joint preference long tables."""
    out: Dict[str, object] = {}
    for k, v in metrics.items():
        out[f"{prefix}_{k}"] = v
    return out
