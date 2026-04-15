"""Aggregation helpers for pairwise comparison analysis.

This module provides functions to aggregate and analyze Stage 5B pairwise
comparison results, computing win rates, preference matrices, and
statistical significance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS
from src.vibe_testing.pairwise_judgment_types import PAIRWISE_JUDGMENT_TYPE_PERSONA


class PairwiseAnalysisError(RuntimeError):
    """Raised when pairwise analysis encounters invalid or missing data."""


@dataclass
class PairwiseAggregationBundle:
    """
    Container for all pairwise analysis outputs.

    Attributes:
        sample_level: Raw comparison records (one row per comparison).
        pair_summary: Aggregated metrics per model pair.
        dimension_summary: Aggregated metrics per dimension.
        user_pair_summary: Aggregated metrics per user x pair.
        preference_matrix: N x N win rate matrix (if multiple model pairs).
        statistical_tests: P-values and confidence intervals.
    """

    sample_level: pd.DataFrame
    pair_summary: pd.DataFrame
    dimension_summary: pd.DataFrame
    user_pair_summary: pd.DataFrame
    preference_matrix: pd.DataFrame
    statistical_tests: pd.DataFrame


def compute_pairwise_rubric_detail_summary(
    pairwise_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rubric-level summary statistics aligned to joint preference conditions.

    This helper aggregates rubric "details" (sub-feature values) for each ordered
    model pair in the pairwise comparisons. For each comparison row, and for each
    vibe dimension, we look up the corresponding rubric details from Stage 5c
    indicator outputs for model A (row_model) and model B (col_model), then compute
    mean/std *separately* for the row and col model values.

    Args:
        pairwise_df (pd.DataFrame): Pairwise comparison rows (Stage 5B or synthetic
            indicator-based pairwise), with columns including:
            ``user_id``, ``task_id``, ``variant_label``, ``variant_id``,
            ``model_a_name``, ``model_b_name``.
        indicator_df (pd.DataFrame): Stage 5c indicator outputs loaded via
            :meth:`src.vibe_testing.analysis.io.AnalysisInputLoader.load_indicator_scores`,
            including ``rubric_dimension_details``.

    Returns:
        pd.DataFrame: Long-form table with one row per
        (persona, prompt_type, judge_model_name, row_model, col_model, vibe_dimension, rubric_name)
        containing:
        - row_mean: Mean(detail_value for row_model)
        - row_std: Standard deviation(detail_value for row_model)
        - col_mean: Mean(detail_value for col_model)
        - col_std: Standard deviation(detail_value for col_model)
        - n: Number of comparisons contributing

    Raises:
        PairwiseAnalysisError: If required columns are missing, indicator details are absent,
            or indicator records cannot be uniquely matched to comparisons.
    """
    if pairwise_df is None or pairwise_df.empty:
        raise PairwiseAnalysisError(
            "pairwise_df is empty; cannot compute rubric summary."
        )
    if indicator_df is None or indicator_df.empty:
        raise PairwiseAnalysisError(
            "indicator_df is empty; cannot compute rubric summary. "
            "Run Stage 5c indicator scoring and provide it to Stage 6."
        )

    required_pairwise = {
        "user_id",
        "task_id",
        "variant_label",
        "variant_id",
        "model_a_name",
        "model_b_name",
    }
    missing_pairwise = sorted(
        [c for c in required_pairwise if c not in pairwise_df.columns]
    )
    if missing_pairwise:
        raise PairwiseAnalysisError(
            "pairwise_df is missing required columns for rubric summary: "
            f"{missing_pairwise}"
        )

    required_indicator = {
        "user_id",
        "task_id",
        "variant_label",
        "variant_id",
        "model_name",
        "rubric_dimension_details",
    }
    missing_indicator = sorted(
        [c for c in required_indicator if c not in indicator_df.columns]
    )
    if missing_indicator:
        raise PairwiseAnalysisError(
            "indicator_df is missing required columns for rubric summary: "
            f"{missing_indicator}. "
            "This typically means Stage 5c was run before rubric details were recorded. "
            "Re-run Stage 5c with `--include-rubric` using the updated code."
        )

    # Validate presence of rubric details.
    details_series = indicator_df["rubric_dimension_details"]
    if details_series.isna().all():
        raise PairwiseAnalysisError(
            "indicator_df has no rubric_dimension_details values. "
            "Re-run Stage 5c with `--include-rubric` using the updated code."
        )

    # Build a strict lookup: (user_id, task_id, variant_label, variant_id, model_name) -> details
    key_cols = ["user_id", "task_id", "variant_label", "variant_id", "model_name"]
    ind = indicator_df.copy()
    ind = ind.dropna(
        subset=["user_id", "task_id", "variant_label", "variant_id", "model_name"]
    )
    ind["_key"] = list(
        zip(
            ind["user_id"].astype(str),
            ind["task_id"].astype(str),
            ind["variant_label"].astype(str),
            ind["variant_id"].astype(str),
            ind["model_name"].astype(str),
        )
    )

    # Detect conflicting duplicates (same key, different details payload).
    dup_mask = ind.duplicated(subset=["_key"], keep=False)
    if dup_mask.any():
        conflicts: List[str] = []
        for key, grp in ind[dup_mask].groupby("_key", dropna=False):
            payloads = []
            for v in grp["rubric_dimension_details"].tolist():
                payloads.append(v)
            # Consider dict payload equality; ignore None entries.
            unique = []
            for p in payloads:
                if p is None or (isinstance(p, float) and pd.isna(p)):
                    continue
                if all(p != q for q in unique):
                    unique.append(p)
            if len(unique) > 1:
                conflicts.append(str(key))
        if conflicts:
            raise PairwiseAnalysisError(
                "indicator_df contains conflicting duplicate rubric detail records for keys: "
                f"{conflicts[:5]}"
                + (" (truncated)" if len(conflicts) > 5 else "")
                + ". Ensure Stage 5c outputs are not duplicated across runs or deduplicate inputs."
            )
        # Non-conflicting duplicates: keep the first deterministically.
        ind = ind.drop_duplicates(subset=["_key"], keep="first")

    lookup: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for _, row in ind.iterrows():
        key = row["_key"]
        payload = row.get("rubric_dimension_details")
        if isinstance(payload, dict):
            lookup[key] = payload

    def _numeric(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        try:
            return float(value)
        except Exception:
            return None

    records: List[Dict[str, Any]] = []
    judge_col = (
        "judge_model_name" if "judge_model_name" in pairwise_df.columns else None
    )

    for _, row in pairwise_df.iterrows():
        persona = str(row["user_id"])
        task_id = str(row["task_id"])
        prompt_type = str(row["variant_label"])
        variant_id = str(row["variant_id"])
        model_a = str(row["model_a_name"])
        model_b = str(row["model_b_name"])
        judge_name = str(row[judge_col]) if judge_col else None

        key_a = (persona, task_id, prompt_type, variant_id, model_a)
        key_b = (persona, task_id, prompt_type, variant_id, model_b)
        details_a = lookup.get(key_a)
        details_b = lookup.get(key_b)
        if details_a is None or details_b is None:
            raise PairwiseAnalysisError(
                "Missing rubric detail records needed for rubric summary. "
                f"Missing A={details_a is None} B={details_b is None} for "
                f"(user_id={persona!r}, task_id={task_id!r}, variant_label={prompt_type!r}, "
                f"variant_id={variant_id!r}, model_a={model_a!r}, model_b={model_b!r}). "
                "Ensure Stage 5c indicator scores were produced for *both* models and prompt types."
            )

        for dim in PAIRWISE_DIMENSIONS:
            a_dim = details_a.get(dim, {}) if isinstance(details_a, dict) else {}
            b_dim = details_b.get(dim, {}) if isinstance(details_b, dict) else {}
            if not isinstance(a_dim, dict) or not isinstance(b_dim, dict):
                continue

            # Only include keys where both sides are numeric.
            all_keys = sorted(set(a_dim.keys()) | set(b_dim.keys()))
            for rubric_name in all_keys:
                va = _numeric(a_dim.get(rubric_name))
                vb = _numeric(b_dim.get(rubric_name))
                if va is None or vb is None:
                    continue
                records.append(
                    {
                        "persona": persona,
                        "prompt_type": prompt_type,
                        "judge_model_name": judge_name,
                        "pairwise_judgment_type": str(
                            row.get(
                                "pairwise_judgment_type",
                                PAIRWISE_JUDGMENT_TYPE_PERSONA,
                            )
                        ),
                        "row_model": model_a,
                        "col_model": model_b,
                        "vibe_dimension": dim,
                        "rubric_name": str(rubric_name),
                        "row_value": float(va),
                        "col_value": float(vb),
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        raise PairwiseAnalysisError(
            "No rubric feature values could be computed. "
            "This typically means rubric_dimension_details lacks numeric fields, "
            "or indicator scores did not overlap the pairwise comparisons."
        )

    group_cols = [
        "persona",
        "prompt_type",
        "judge_model_name",
        "pairwise_judgment_type",
        "row_model",
        "col_model",
        "vibe_dimension",
        "rubric_name",
    ]
    grouped = out.groupby(group_cols, dropna=False)
    agg = grouped.agg(
        row_mean=("row_value", "mean"),
        row_std=("row_value", "std"),
        col_mean=("col_value", "mean"),
        col_std=("col_value", "std"),
        n=("row_value", "count"),
    ).reset_index()
    # Fill NaN std when n==1.
    for col in ("row_std", "col_std"):
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)
    return agg


def build_pairwise_df_from_indicator_scores(
    indicator_df: pd.DataFrame,
    score_field: str = "rubric_dimension_scores",
    judge_model_name: str = "indicator_scores",
    correctness_mode: str = "ignore",
    objective_lookup: Optional[
        Dict[Tuple[str, str, str], Dict[str, Optional[float]]]
    ] = None,
    include_plus_correctness: bool = False,
) -> pd.DataFrame:
    """
    Build a synthetic pairwise comparison DataFrame from Stage 5c indicator scores.

    This allows Stage 6 to reuse the *existing* pairwise aggregation and figure
    pipeline without requiring Stage 5B judge comparisons.

    The core assumption is that ``indicator_df`` contains a per-sample per-model
    dict of per-dimension scalar scores (default: ``rubric_dimension_scores``),
    which can be compared across models to produce a deterministic winner for each
    vibe dimension.

    Args:
        indicator_df (pd.DataFrame): Output of
            :meth:`src.vibe_testing.analysis.io.AnalysisInputLoader.load_indicator_scores`.
        score_field (str): Column containing a dict mapping dimension -> float score.
        judge_model_name (str): Synthetic judge identifier stored in the pairwise rows.
        correctness_mode: How to incorporate pass@1 correctness (``"ignore"``,
            ``"dimension"``, or ``"gate"``).
        objective_lookup: Pre-built objective lookup (required when
            ``correctness_mode != "ignore"``).
        include_plus_correctness: When True and ``correctness_mode == "dimension"``,
            add plus-test pass@1 as an additional vote.

    Returns:
        pd.DataFrame: Pairwise-like DataFrame compatible with :func:`run_pairwise_aggregation`.

    Raises:
        PairwiseAnalysisError: If required fields are missing or rubric scores are absent.
    """
    if indicator_df is None or indicator_df.empty:
        raise PairwiseAnalysisError("Indicator scores DataFrame is empty.")

    required = {
        "user_id",
        "task_id",
        "variant_label",
        "variant_id",
        "model_name",
        score_field,
    }
    missing = sorted([c for c in required if c not in indicator_df.columns])
    if missing:
        raise PairwiseAnalysisError(
            "Indicator scores are missing required columns for pairwise synthesis: "
            f"{missing}. Ensure Stage 5c outputs include '{score_field}'."
        )

    # Prefer stable grouping by (user, task, variant) and keep at most one record per model.
    # If upstream contains duplicate rows (e.g., per-attempt), keep the first deterministically.
    base_cols = ["user_id", "task_id", "variant_label", "variant_id", "model_name"]
    df = indicator_df.copy()
    df = df.dropna(subset=["user_id", "task_id", "model_name"])

    # Deterministic attempt preference: prefer the smallest ::attempt:: index when present.
    attempt_token = "::attempt::"
    if "raw_task_id" in df.columns:

        def _attempt_index(raw: object) -> int:
            if raw is None:
                return 0
            s = str(raw)
            if attempt_token not in s:
                return 0
            try:
                return int(s.split(attempt_token, 1)[1].strip())
            except Exception:
                return 0

        df["_attempt_index"] = df["raw_task_id"].map(_attempt_index)
        df = df.sort_values(base_cols + ["_attempt_index"]).drop(
            columns=["_attempt_index"]
        )

    df = df.drop_duplicates(subset=base_cols, keep="first")

    records: List[Dict[str, Any]] = []

    group_cols = ["user_id", "task_id", "variant_label", "variant_id"]
    for keys, group in df.groupby(group_cols, dropna=False):
        context = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        extras: Dict[str, Any] = {}
        for col in ("generator_model", "filter_model", "evaluated_model"):
            if col in group.columns:
                values = group[col].dropna().unique().tolist()
                extras[col] = values[0] if values else None

        model_scores: Dict[str, Dict[str, float]] = {}
        for _, row in group.iterrows():
            model_name = str(row["model_name"])
            payload = row.get(score_field)
            if not isinstance(payload, dict):
                raise PairwiseAnalysisError(
                    f"Missing or invalid '{score_field}' for model='{model_name}' "
                    f"sample task_id='{context.get('task_id')}'. "
                    "To generate rubric scores, run Stage 5c with --include-rubric."
                )

            # Ensure all required dimensions are present and numeric.
            scores_for_model: Dict[str, float] = {}
            for dim in PAIRWISE_DIMENSIONS:
                if dim not in payload:
                    raise PairwiseAnalysisError(
                        f"Missing dimension '{dim}' in '{score_field}' for model='{model_name}' "
                        f"task_id='{context.get('task_id')}'."
                    )
                try:
                    scores_for_model[dim] = float(payload[dim])
                except Exception as exc:
                    raise PairwiseAnalysisError(
                        f"Non-numeric score for dim='{dim}' in '{score_field}' for model='{model_name}' "
                        f"task_id='{context.get('task_id')}': {payload.get(dim)!r}"
                    ) from exc

            model_scores[model_name] = scores_for_model

        models = sorted(model_scores.keys())
        if len(models) < 2:
            continue

        # Compare every model pair for this sample.
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]
                a_scores = model_scores[model_a]
                b_scores = model_scores[model_b]

                win_a = 0
                win_b = 0
                win_tie = 0
                row_out: Dict[str, Any] = {
                    **context,
                    **extras,
                    "raw_task_id": context.get("task_id"),
                    "model_a_name": model_a,
                    "model_b_name": model_b,
                    "model_pair": f"{model_a}_vs_{model_b}",
                    "judge_model_name": judge_model_name,
                    "pairwise_judgment_type": "indicator_scores",
                    "position_swap_enabled": False,
                    "total_bias_detected": 0,
                    "bias_rate": 0.0,
                    "input_text": None,
                    "pairwise_source": "indicator_scores",
                }

                for dim in PAIRWISE_DIMENSIONS:
                    a = a_scores[dim]
                    b = b_scores[dim]
                    if a > b:
                        winner = "A"
                        winner_model = model_a
                        winner_label = "model_a"
                        win_a += 1
                    elif b > a:
                        winner = "B"
                        winner_model = model_b
                        winner_label = "model_b"
                        win_b += 1
                    else:
                        winner = "tie"
                        winner_model = None
                        winner_label = "tie"
                        win_tie += 1

                    row_out[f"dim_{dim}_winner"] = winner
                    row_out[f"dim_{dim}_winner_model"] = winner_model
                    row_out[f"dim_{dim}_winner_label"] = winner_label
                    row_out[f"dim_{dim}_confidence"] = "low"
                    row_out[f"dim_{dim}_bias_detected"] = False
                    row_out[f"dim_{dim}_rationale"] = (
                        f"indicator_scores: {score_field} compare (A={a:.6g}, B={b:.6g})"
                    )

                # Incorporate correctness into per-sample winner
                has_correctness = False
                base_a_val: Optional[float] = None
                base_b_val: Optional[float] = None
                plus_a_val: Optional[float] = None
                plus_b_val: Optional[float] = None

                if correctness_mode != "ignore" and objective_lookup:
                    sample_key = _row_sample_key(pd.Series(row_out))
                    vl = row_out.get("variant_label", "original")
                    if sample_key:
                        m_a = objective_lookup.get((sample_key, vl, model_a), {})
                        m_b = objective_lookup.get((sample_key, vl, model_b), {})
                        base_a_val = m_a.get("pass_at_1")
                        base_b_val = m_b.get("pass_at_1")
                        plus_a_val = m_a.get("plus_pass_at_1")
                        plus_b_val = m_b.get("plus_pass_at_1")
                        has_correctness = (
                            base_a_val is not None and base_b_val is not None
                        )

                row_out["pairwise_correctness_data_available"] = has_correctness
                gated = False

                # Gate mode: one correct and other not -> auto-win
                if (
                    correctness_mode == "gate"
                    and has_correctness
                    and base_a_val is not None
                    and base_b_val is not None
                ):
                    a_correct = base_a_val > 0
                    b_correct = base_b_val > 0
                    if a_correct and not b_correct:
                        row_out["overall_winner"] = model_a
                        row_out["overall_winner_label"] = "model_a"
                        row_out["win_count_a"] = int(win_a)
                        row_out["win_count_b"] = int(win_b)
                        row_out["win_count_tie"] = int(win_tie)
                        records.append(row_out)
                        gated = True
                    elif b_correct and not a_correct:
                        row_out["overall_winner"] = model_b
                        row_out["overall_winner_label"] = "model_b"
                        row_out["win_count_a"] = int(win_a)
                        row_out["win_count_b"] = int(win_b)
                        row_out["win_count_tie"] = int(win_tie)
                        records.append(row_out)
                        gated = True

                if gated:
                    continue

                # Dimension mode: add correctness vote(s)
                if correctness_mode == "dimension" and has_correctness:
                    if base_a_val > base_b_val:
                        win_a += 1
                    elif base_b_val > base_a_val:
                        win_b += 1
                    else:
                        win_tie += 1

                    if (
                        include_plus_correctness
                        and plus_a_val is not None
                        and plus_b_val is not None
                    ):
                        if plus_a_val > plus_b_val:
                            win_a += 1
                        elif plus_b_val > plus_a_val:
                            win_b += 1
                        else:
                            win_tie += 1

                # Overall winner: majority of per-dimension winners (ties ignored).
                if win_a > win_b:
                    overall = model_a
                    overall_label = "model_a"
                elif win_b > win_a:
                    overall = model_b
                    overall_label = "model_b"
                else:
                    overall = None
                    overall_label = "tie"

                row_out["overall_winner"] = overall
                row_out["overall_winner_label"] = overall_label
                row_out["win_count_a"] = int(win_a)
                row_out["win_count_b"] = int(win_b)
                row_out["win_count_tie"] = int(win_tie)

                records.append(row_out)

    out_df = pd.DataFrame(records)
    if out_df.empty:
        raise PairwiseAnalysisError(
            "No synthetic pairwise rows could be built from indicator scores. "
            "Ensure at least two models have indicator outputs for overlapping samples."
        )
    return out_df


def run_pairwise_aggregation(
    pairwise_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    variant_filter: Optional[str] = None,
    judge_filter: Optional[str] = None,
    user_filter: Optional[str] = None,
    objective_df: Optional[pd.DataFrame] = None,
    objective_lookup: Optional[
        Dict[Tuple[str, str, str], Dict[str, Optional[float]]]
    ] = None,
) -> PairwiseAggregationBundle:
    """
    Execute full pairwise aggregation flow.

    Args:
        pairwise_df: DataFrame from load_pairwise_results().
        profiles_df: User profile metadata.
        variant_filter: Optional filter for variant type:
            - None: Aggregate across all variants (combined view)
            - "original": Only include original prompts
            - "personalized": Only include personalized prompts
        judge_filter: Optional filter for judge model:
            - None: Aggregate across all judge models
            - "<judge_name>": Only include results from this judge model
        user_filter: Optional filter for user/persona:
            - None: Aggregate across all users/personas
            - "<user_id>": Only include results from this user/persona
        objective_df: Optional objective metrics DataFrame used to compute
            pass@k (k in {1,5}) win rates per model pair; ties when equal.
        objective_lookup: Optional pre-built objective lookup dictionary. If provided,
            objective_df is ignored. This allows caching the lookup across multiple calls.

    Returns:
        PairwiseAggregationBundle: All derived analysis tables.

    Raises:
        PairwiseAnalysisError: If input data is empty or invalid.
    """
    if pairwise_df is None or pairwise_df.empty:
        raise PairwiseAnalysisError("Pairwise results DataFrame is empty.")

    # Attach user profile info if available
    sample_df = pairwise_df.copy()
    if not profiles_df.empty and "user_id" in profiles_df.columns:
        profile_cols = ["user_id"]
        if "user_profile_type" in profiles_df.columns:
            profile_cols.append("user_profile_type")
        sample_df = sample_df.merge(
            profiles_df[profile_cols].drop_duplicates(),
            on="user_id",
            how="left",
        )

    # Apply user/persona filter if specified
    if user_filter is not None and "user_id" in sample_df.columns:
        sample_df = sample_df[sample_df["user_id"] == user_filter].copy()
        if sample_df.empty:
            raise PairwiseAnalysisError(f"No data for user_filter='{user_filter}'.")

    # Apply judge filter if specified
    if judge_filter is not None and "judge_model_name" in sample_df.columns:
        sample_df = sample_df[sample_df["judge_model_name"] == judge_filter].copy()
        if sample_df.empty:
            raise PairwiseAnalysisError(f"No data for judge_filter='{judge_filter}'.")

    # Apply variant filter if specified
    if variant_filter is not None and "variant_label" in sample_df.columns:
        sample_df = sample_df[sample_df["variant_label"] == variant_filter].copy()
        if sample_df.empty:
            raise PairwiseAnalysisError(
                f"No data for variant_filter='{variant_filter}'."
            )

    # Compute aggregations
    # Don't group by variant/judge/user when aggregating across them
    group_by_variant = variant_filter is not None
    group_by_judge = judge_filter is not None
    group_by_user = user_filter is not None

    # Use pre-built lookup if provided, otherwise build from objective_df
    if objective_lookup is None:
        objective_lookup = (
            _build_objective_lookup(objective_df) if objective_df is not None else None
        )

    pair_summary = compute_pair_summary(
        sample_df,
        group_by_variant=group_by_variant,
        group_by_judge=group_by_judge,
        group_by_user=group_by_user,
        objective_lookup=objective_lookup,
    )
    dimension_summary = compute_dimension_win_rates(
        sample_df,
        group_by_variant=group_by_variant,
        group_by_judge=group_by_judge,
        group_by_user=group_by_user,
    )
    if objective_df is not None and not objective_df.empty:
        dimension_summary = augment_dimensions_with_objective(
            dimension_summary, pair_summary
        )
    user_pair_summary = compute_user_pair_summary(
        sample_df,
        group_by_variant=group_by_variant,
        group_by_judge=group_by_judge,
        group_by_user=group_by_user,
    )
    preference_matrix = build_preference_matrix(pair_summary)
    statistical_tests = compute_statistical_significance(
        sample_df,
        group_by_variant=group_by_variant,
        group_by_judge=group_by_judge,
        group_by_user=group_by_user,
    )

    return PairwiseAggregationBundle(
        sample_level=sample_df,
        pair_summary=pair_summary,
        dimension_summary=dimension_summary,
        user_pair_summary=user_pair_summary,
        preference_matrix=preference_matrix,
        statistical_tests=statistical_tests,
    )


def compute_pair_summary(
    pairwise_df: pd.DataFrame,
    group_by_variant: bool = False,
    group_by_judge: bool = False,
    group_by_user: bool = False,
    persona_importance_weights: Optional[Dict[str, float]] = None,
    use_persona_weighted_rates: bool = False,
    objective_lookup: Optional[
        Dict[Tuple[str, str, str], Dict[str, Optional[float]]]
    ] = None,
) -> pd.DataFrame:
    """
    Compute aggregated metrics per model pair.

    Args:
        pairwise_df: Sample-level pairwise comparison data.
        group_by_variant: If True, group by variant_label. If False, aggregate
            across all variants for a combined view.
        group_by_judge: If True, group by judge_model_name. If False, aggregate
            across all judge models.
        group_by_user: If True, group by user_id. If False, aggregate
            across all users/personas.
        persona_importance_weights: Optional mapping from user_id to an
            importance weight used for weighted aggregation.
        use_persona_weighted_rates: Whether to replace raw aggregate win rates
            with the weighted average of per-user win rates.
        objective_lookup: Optional lookup keyed by (task_id, variant_label, model_name)
            with pass@k metrics to compute objective win rates per pair.

    Returns:
        DataFrame with one row per model pair (or per pair+variant+judge+user), containing:
        - total_comparisons: Total number of comparisons
        - model_a_wins: Number of wins for model A
        - model_b_wins: Number of wins for model B
        - ties: Number of ties
        - model_a_win_rate: Win rate for model A (wins / total)
        - model_b_win_rate: Win rate for model B (wins / total)
        - tie_rate: Tie rate (ties / total)
        - model_a_win_prob: Win probability excluding ties
        - avg_bias_rate: Average position bias rate
    """
    if pairwise_df.empty:
        return pd.DataFrame()

    group_cols = ["model_a_name", "model_b_name", "model_pair"]
    if "pairwise_judgment_type" in pairwise_df.columns:
        group_cols.append("pairwise_judgment_type")
    # Only group by user_id if explicitly requested
    if group_by_user and "user_id" in pairwise_df.columns:
        group_cols.append("user_id")
    # Only group by judge_model_name if explicitly requested
    if group_by_judge and "judge_model_name" in pairwise_df.columns:
        group_cols.append("judge_model_name")
    # Only group by variant_label if explicitly requested
    if group_by_variant and "variant_label" in pairwise_df.columns:
        group_cols.append("variant_label")

    records: List[Dict[str, Any]] = []
    for keys, group in pairwise_df.groupby(group_cols, dropna=False):
        context = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))

        total = len(group)
        a_wins = (group["overall_winner_label"] == "model_a").sum()
        b_wins = (group["overall_winner_label"] == "model_b").sum()
        ties = (group["overall_winner_label"] == "tie").sum()

        # Win rates
        a_rate = a_wins / total if total > 0 else 0.0
        b_rate = b_wins / total if total > 0 else 0.0
        tie_rate = ties / total if total > 0 else 0.0

        # Win probability (excluding ties)
        non_tie = a_wins + b_wins
        a_prob = a_wins / non_tie if non_tie > 0 else 0.5

        # Bias rate
        avg_bias = group["bias_rate"].mean() if "bias_rate" in group.columns else 0.0

        record = {
            **context,
            "total_comparisons": total,
            "model_a_wins": int(a_wins),
            "model_b_wins": int(b_wins),
            "ties": int(ties),
            "model_a_win_rate": a_rate,
            "model_b_win_rate": b_rate,
            "tie_rate": tie_rate,
            "model_a_win_prob": a_prob,
            "avg_bias_rate": avg_bias,
            "persona_weighted": False,
            "persona_weighted_user_count": 0,
            "persona_weight_sum": 0.0,
        }

        if (
            use_persona_weighted_rates
            and persona_importance_weights
            and not group_by_user
            and "user_id" in group.columns
        ):
            user_rows: List[Tuple[float, float, float, float]] = []
            for user_id, user_group in group.groupby("user_id", dropna=False):
                user_key = str(user_id)
                weight = float(persona_importance_weights.get(user_key, 0.0))
                if weight <= 0:
                    continue
                user_total = len(user_group)
                if user_total <= 0:
                    continue
                user_a_wins = (user_group["overall_winner_label"] == "model_a").sum()
                user_b_wins = (user_group["overall_winner_label"] == "model_b").sum()
                user_ties = (user_group["overall_winner_label"] == "tie").sum()
                user_rows.append(
                    (
                        weight,
                        user_a_wins / user_total,
                        user_b_wins / user_total,
                        user_ties / user_total,
                    )
                )

            if user_rows:
                weight_sum = sum(row[0] for row in user_rows)
                if weight_sum > 0:
                    record["model_a_win_rate"] = (
                        sum(weight * rate for weight, rate, _, _ in user_rows)
                        / weight_sum
                    )
                    record["model_b_win_rate"] = (
                        sum(weight * rate for weight, _, rate, _ in user_rows)
                        / weight_sum
                    )
                    record["tie_rate"] = (
                        sum(weight * rate for weight, _, _, rate in user_rows)
                        / weight_sum
                    )
                    non_tie_weighted = (
                        record["model_a_win_rate"] + record["model_b_win_rate"]
                    )
                    record["model_a_win_prob"] = (
                        record["model_a_win_rate"] / non_tie_weighted
                        if non_tie_weighted > 0
                        else 0.5
                    )
                    record["persona_weighted"] = True
                    record["persona_weighted_user_count"] = len(user_rows)
                    record["persona_weight_sum"] = float(weight_sum)

        # Optional: objective pass@k win rates (k in {1,5})
        if objective_lookup:
            for k in (1, 5):
                outcome = _compute_objective_outcomes_for_group(
                    group, objective_lookup, k
                )
                record.update(outcome)
            # Also compute plus-test Pass@1
            plus_outcome = _compute_plus_pass_at_1_outcomes_for_group(
                group, objective_lookup
            )
            record.update(plus_outcome)
        records.append(record)

    return pd.DataFrame(records)


def compute_dimension_win_rates(
    pairwise_df: pd.DataFrame,
    group_by_variant: bool = False,
    group_by_judge: bool = False,
    group_by_user: bool = False,
) -> pd.DataFrame:
    """
    Compute per-dimension win rates with position bias information.

    Args:
        pairwise_df: Sample-level pairwise comparison data.
        group_by_variant: If True, group by variant_label. If False, aggregate
            across all variants for a combined view.
        group_by_judge: If True, group by judge_model_name. If False, aggregate
            across all judge models.
        group_by_user: If True, group by user_id. If False, aggregate
            across all users/personas.

    Returns:
        DataFrame with one row per (model_pair, dimension), containing:
        - dimension: Dimension name
        - model_a_wins: Number of wins for model A
        - model_b_wins: Number of wins for model B
        - ties: Number of ties
        - model_a_win_rate: Win rate for model A
        - position_bias_count: Number of comparisons with detected bias
        - position_bias_rate: Proportion of comparisons with bias
    """
    if pairwise_df.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    def _is_eval_error_rationale(value: object) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        lowered = text.lower()
        return lowered.startswith("evaluation error:") or lowered.startswith(
            "parse error:"
        )

    # Group by model pair
    group_cols = ["model_a_name", "model_b_name", "model_pair"]
    if "pairwise_judgment_type" in pairwise_df.columns:
        group_cols.append("pairwise_judgment_type")
    # Only group by user_id if explicitly requested
    if group_by_user and "user_id" in pairwise_df.columns:
        group_cols.append("user_id")
    # Only group by judge_model_name if explicitly requested
    if group_by_judge and "judge_model_name" in pairwise_df.columns:
        group_cols.append("judge_model_name")
    # Only group by variant_label if explicitly requested
    if group_by_variant and "variant_label" in pairwise_df.columns:
        group_cols.append("variant_label")

    for keys, group in pairwise_df.groupby(group_cols, dropna=False):
        context = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))

        for dim in PAIRWISE_DIMENSIONS:
            winner_col = f"dim_{dim}_winner_label"
            bias_col = f"dim_{dim}_bias_detected"
            conf_col = f"dim_{dim}_confidence"
            rationale_col = f"dim_{dim}_rationale"

            if winner_col not in group.columns:
                continue

            # Exclude evaluation/parsing errors for this dimension from denominators.
            # Stage 5B records these as ties with low confidence and an error rationale.
            if rationale_col in group.columns:
                valid_mask = ~group[rationale_col].apply(_is_eval_error_rationale)
            else:
                valid_mask = pd.Series([True] * len(group), index=group.index)

            valid_group = group[valid_mask]
            total = int(len(valid_group))
            a_wins = (valid_group[winner_col] == "model_a").sum()
            b_wins = (valid_group[winner_col] == "model_b").sum()
            ties = (valid_group[winner_col] == "tie").sum()

            bias_count = (
                valid_group[bias_col].sum() if bias_col in valid_group.columns else 0
            )

            # Confidence distribution
            conf_dist = {}
            if conf_col in group.columns:
                for level in ["low", "medium", "high"]:
                    conf_dist[f"conf_{level}_count"] = (
                        valid_group[conf_col] == level
                    ).sum()

            record = {
                **context,
                "dimension": dim,
                "total_comparisons": total,
                "model_a_wins": int(a_wins),
                "model_b_wins": int(b_wins),
                "ties": int(ties),
                "model_a_win_rate": a_wins / total if total > 0 else 0.0,
                "model_b_win_rate": b_wins / total if total > 0 else 0.0,
                "tie_rate": ties / total if total > 0 else 0.0,
                "position_bias_count": int(bias_count),
                "position_bias_rate": bias_count / total if total > 0 else 0.0,
                "evaluation_error_count": int((~valid_mask).sum()),
                **conf_dist,
            }
            records.append(record)

    return pd.DataFrame(records)


def compute_user_pair_summary(
    pairwise_df: pd.DataFrame,
    group_by_variant: bool = False,
    group_by_judge: bool = False,
    group_by_user: bool = False,
) -> pd.DataFrame:
    """
    Compute per-user per-pair summary statistics.

    Args:
        pairwise_df: Sample-level pairwise comparison data.
        group_by_variant: If True, group by variant_label. If False, aggregate
            across all variants for a combined view.
        group_by_judge: If True, group by judge_model_name. If False, aggregate
            across all judge models.
        group_by_user: If True, group by user_id (already included by default
            in this function). This parameter is accepted for consistency but
            has no additional effect since user_id is always in group columns.

    Returns:
        DataFrame with one row per (user_id, model_pair), containing
        win counts and rates for that user's comparisons.
    """
    if pairwise_df.empty:
        return pd.DataFrame()

    # Note: user_id is always included in this function's grouping
    group_cols = ["user_id", "model_a_name", "model_b_name", "model_pair"]
    if "pairwise_judgment_type" in pairwise_df.columns:
        group_cols.append("pairwise_judgment_type")
    # Add optional columns
    if "user_profile_type" in pairwise_df.columns:
        group_cols.append("user_profile_type")
    # Only group by judge_model_name if explicitly requested
    if group_by_judge and "judge_model_name" in pairwise_df.columns:
        group_cols.append("judge_model_name")
    # Only group by variant_label if explicitly requested
    if group_by_variant and "variant_label" in pairwise_df.columns:
        group_cols.append("variant_label")

    records: List[Dict[str, Any]] = []
    for keys, group in pairwise_df.groupby(group_cols, dropna=False):
        context = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))

        total = len(group)
        a_wins = (group["overall_winner_label"] == "model_a").sum()
        b_wins = (group["overall_winner_label"] == "model_b").sum()
        ties = (group["overall_winner_label"] == "tie").sum()

        record = {
            **context,
            "total_comparisons": total,
            "model_a_wins": int(a_wins),
            "model_b_wins": int(b_wins),
            "ties": int(ties),
            "model_a_win_rate": a_wins / total if total > 0 else 0.0,
            "model_b_win_rate": b_wins / total if total > 0 else 0.0,
            "tie_rate": ties / total if total > 0 else 0.0,
        }
        records.append(record)

    return pd.DataFrame(records)


def build_preference_matrix(pair_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build N x N preference matrix from pairwise summaries.

    The matrix entry (i, j) represents the win rate of model i against model j.
    Diagonal entries are set to 0.5 (neutral).

    Args:
        pair_summary: Output from compute_pair_summary().

    Returns:
        Square DataFrame with models as both index and columns,
        containing win rates. Returns empty DataFrame if insufficient data.
    """
    if pair_summary.empty:
        return pd.DataFrame()

    # Collect all unique models
    models_a = set(pair_summary["model_a_name"].unique())
    models_b = set(pair_summary["model_b_name"].unique())
    all_models = sorted(models_a | models_b)

    if len(all_models) < 2:
        return pd.DataFrame()

    # Initialize matrix with 0.5 (neutral)
    matrix = pd.DataFrame(
        0.5,
        index=all_models,
        columns=all_models,
    )

    # Fill in win rates using vectorized operations
    if not pair_summary.empty and all(
        col in pair_summary.columns
        for col in [
            "model_a_name",
            "model_b_name",
            "model_a_win_rate",
            "model_b_win_rate",
        ]
    ):
        # Filter to valid model pairs
        valid_mask = pair_summary["model_a_name"].isin(matrix.index) & pair_summary[
            "model_b_name"
        ].isin(matrix.columns)
        valid_pairs = pair_summary[valid_mask]

        if not valid_pairs.empty:
            # Vectorized assignment using .loc with indexer
            for model_a, model_b, a_rate, b_rate in zip(
                valid_pairs["model_a_name"],
                valid_pairs["model_b_name"],
                valid_pairs["model_a_win_rate"],
                valid_pairs["model_b_win_rate"],
            ):
                matrix.loc[model_a, model_b] = a_rate
                matrix.loc[model_b, model_a] = b_rate

    return matrix


def augment_dimensions_with_objective(
    dimension_summary: pd.DataFrame, pair_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Append objective pass@k win-rate rows to the dimension summary.

    Args:
        dimension_summary: Existing dimension-level summary.
        pair_summary: Pair-level summary containing objective pass@k win rates.

    Returns:
        DataFrame with added rows for objective pass@1, pass@5, and plus-test Pass@1
        (when present).
    """
    if pair_summary is None or pair_summary.empty:
        return dimension_summary

    # Vectorized approach: melt and filter in one go
    objective_rows_list: List[Dict[str, Any]] = []

    # Base columns that are always present
    base_cols = ["model_a_name", "model_b_name", "model_pair"]
    optional_cols = [
        "user_id",
        "judge_model_name",
        "variant_label",
        "pairwise_judgment_type",
    ]
    available_cols = [col for col in optional_cols if col in pair_summary.columns]
    all_base_cols = base_cols + available_cols

    # Process Pass@1 and Pass@5
    for k in (1, 5):
        total_col = f"obj_pass_at_{k}_comparisons"
        a_rate_col = f"obj_pass_at_{k}_model_a_win_rate"
        b_rate_col = f"obj_pass_at_{k}_model_b_win_rate"
        tie_rate_col = f"obj_pass_at_{k}_tie_rate"

        if not all(
            col in pair_summary.columns
            for col in [total_col, a_rate_col, b_rate_col, tie_rate_col]
        ):
            continue

        # Filter rows with valid data
        mask = (
            pair_summary[total_col].notna()
            & (pair_summary[total_col] > 0)
            & pair_summary[a_rate_col].notna()
            & pair_summary[b_rate_col].notna()
            & pair_summary[tie_rate_col].notna()
        )

        if not mask.any():
            continue

        valid_rows = pair_summary[mask]

        # Create rows using vectorized operations
        rows_dict = {
            "model_a_name": valid_rows["model_a_name"].values,
            "model_b_name": valid_rows["model_b_name"].values,
            "model_pair": valid_rows["model_pair"].values,
            "dimension": f"objective_pass_at_{k}",
            "total_comparisons": valid_rows[total_col].astype(int).values,
            "model_a_wins": (valid_rows[a_rate_col] * valid_rows[total_col])
            .round()
            .astype(int)
            .values,
            "model_b_wins": (valid_rows[b_rate_col] * valid_rows[total_col])
            .round()
            .astype(int)
            .values,
            "ties": (valid_rows[tie_rate_col] * valid_rows[total_col])
            .round()
            .astype(int)
            .values,
            "model_a_win_rate": valid_rows[a_rate_col].astype(float).values,
            "model_b_win_rate": valid_rows[b_rate_col].astype(float).values,
            "tie_rate": valid_rows[tie_rate_col].astype(float).values,
        }

        # Add optional columns
        for col in available_cols:
            rows_dict[col] = valid_rows[col].values

        # Convert to list of dicts
        n_rows = len(valid_rows)
        for i in range(n_rows):
            row_dict = {
                k: v[i] if isinstance(v, np.ndarray) else v
                for k, v in rows_dict.items()
            }
            objective_rows_list.append(row_dict)

    # Process plus-test Pass@1
    plus_total_col = "obj_plus_pass_at_1_comparisons"
    plus_a_rate_col = "obj_plus_pass_at_1_model_a_win_rate"
    plus_b_rate_col = "obj_plus_pass_at_1_model_b_win_rate"
    plus_tie_rate_col = "obj_plus_pass_at_1_tie_rate"

    if all(
        col in pair_summary.columns
        for col in [plus_total_col, plus_a_rate_col, plus_b_rate_col, plus_tie_rate_col]
    ):
        mask = (
            pair_summary[plus_total_col].notna()
            & (pair_summary[plus_total_col] > 0)
            & pair_summary[plus_a_rate_col].notna()
            & pair_summary[plus_b_rate_col].notna()
            & pair_summary[plus_tie_rate_col].notna()
        )

        if mask.any():
            valid_rows = pair_summary[mask]

            rows_dict = {
                "model_a_name": valid_rows["model_a_name"].values,
                "model_b_name": valid_rows["model_b_name"].values,
                "model_pair": valid_rows["model_pair"].values,
                "dimension": "objective_plus_pass_at_1",
                "total_comparisons": valid_rows[plus_total_col].astype(int).values,
                "model_a_wins": (
                    valid_rows[plus_a_rate_col] * valid_rows[plus_total_col]
                )
                .round()
                .astype(int)
                .values,
                "model_b_wins": (
                    valid_rows[plus_b_rate_col] * valid_rows[plus_total_col]
                )
                .round()
                .astype(int)
                .values,
                "ties": (valid_rows[plus_tie_rate_col] * valid_rows[plus_total_col])
                .round()
                .astype(int)
                .values,
                "model_a_win_rate": valid_rows[plus_a_rate_col].astype(float).values,
                "model_b_win_rate": valid_rows[plus_b_rate_col].astype(float).values,
                "tie_rate": valid_rows[plus_tie_rate_col].astype(float).values,
            }

            for col in available_cols:
                rows_dict[col] = valid_rows[col].values

            n_rows = len(valid_rows)
            for i in range(n_rows):
                row_dict = {
                    k: v[i] if isinstance(v, np.ndarray) else v
                    for k, v in rows_dict.items()
                }
                objective_rows_list.append(row_dict)

    if not objective_rows_list:
        return dimension_summary

    objective_df = pd.DataFrame(objective_rows_list)
    return pd.concat([objective_df, dimension_summary], ignore_index=True, sort=False)


def _build_objective_lookup(
    objective_df: pd.DataFrame,
) -> Dict[Tuple[str, str, str], Dict[str, Optional[float]]]:
    """
    Build lookup for objective pass@k metrics.

    Keying:
    - Prefer a stable *per-sample* key that includes the base task when variations exist:
      ``sample_key = f\"{task_id}::{variant_id}\"``.
      This avoids collapsing cases where variant_id is not globally unique (e.g. control_1/control_2).
    - Fall back to ``task_id`` when variant_id is missing (legacy artifacts).

    Prefers overall pass@k metrics, with base/plus fallbacks when overall is missing.
    Also stores plus pass@1 separately for plus-test Pass@1 analysis.
    """
    if objective_df is None or objective_df.empty:
        return {}

    # Vectorized approach: filter valid rows first
    if "model_name" not in objective_df.columns:
        return {}

    if "task_id" not in objective_df.columns:
        return {}
    has_variant_id = "variant_id" in objective_df.columns

    # Filter out rows with missing task_id or model_name
    mask = objective_df["task_id"].notna() & objective_df["model_name"].notna()
    if has_variant_id:
        mask = mask & objective_df["variant_id"].notna()
    valid_df = objective_df[mask].copy()

    if valid_df.empty:
        return {}

    # Fill variant_label with default
    if "variant_label" not in valid_df.columns:
        valid_df["variant_label"] = "original"
    else:
        valid_df["variant_label"] = valid_df["variant_label"].fillna("original")

    lookup: Dict[Tuple[str, str, str], Dict[str, Optional[float]]] = {}
    sample_keys = valid_df.apply(
        lambda row: _canonical_sample_key(
            task_id=row.get("task_id"),
            variant_id=row.get("variant_id") if has_variant_id else None,
        ),
        axis=1,
    )

    # Vectorized extraction of pass@k metrics
    for k in (1, 5):
        overall_key = f"obj_overall_pass_at_{k}"
        base_key = f"obj_base_pass_at_{k}"
        plus_key = f"obj_plus_pass_at_{k}"

        # Use vectorized operations with fallback logic
        if overall_key in valid_df.columns:
            pass_at_k = valid_df[overall_key].copy()
            if base_key in valid_df.columns:
                pass_at_k = pass_at_k.fillna(valid_df[base_key])
            if plus_key in valid_df.columns:
                pass_at_k = pass_at_k.fillna(valid_df[plus_key])
        elif base_key in valid_df.columns:
            pass_at_k = valid_df[base_key].copy()
            if plus_key in valid_df.columns:
                pass_at_k = pass_at_k.fillna(valid_df[plus_key])
        elif plus_key in valid_df.columns:
            pass_at_k = valid_df[plus_key].copy()
        else:
            pass_at_k = pd.Series([None] * len(valid_df), index=valid_df.index)

        for idx, (sample_key, model_name, variant_label) in enumerate(
            zip(sample_keys, valid_df["model_name"], valid_df["variant_label"])
        ):
            key = (str(sample_key), variant_label, model_name)
            if key not in lookup:
                lookup[key] = {}
            value = pass_at_k.iloc[idx]
            if pd.notna(value):
                lookup[key][f"pass_at_{k}"] = float(value)

    # Store plus pass@1 separately
    plus_pass_at_1_key = "obj_plus_pass_at_1"
    if plus_pass_at_1_key in valid_df.columns:
        for idx, (sample_key, model_name, variant_label) in enumerate(
            zip(sample_keys, valid_df["model_name"], valid_df["variant_label"])
        ):
            key = (str(sample_key), variant_label, model_name)
            if key not in lookup:
                lookup[key] = {}
            value = valid_df[plus_pass_at_1_key].iloc[idx]
            if pd.notna(value):
                lookup[key]["plus_pass_at_1"] = float(value)

    return lookup


def _canonical_sample_key(
    *,
    task_id: object,
    variant_id: object = None,
    raw_task_id: object = None,
) -> Optional[str]:
    """
    Build the canonical Stage-6 sample key used for objective/pairwise joins.

    Keys follow one simple rule:
    - ``original`` rows use ``task_id``
    - non-original rows use ``task_id::variant_id``

    If normalized fields are missing, this helper falls back to parsing
    ``raw_task_id`` values that use the ``::variation::`` encoding emitted by
    Stage 5B artifacts.
    """

    def _clean(v: object) -> Optional[str]:
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        return s

    t = _clean(task_id)
    v = _clean(variant_id)
    raw = _clean(raw_task_id)

    if (t is None or v is None) and raw:
        if "::variation::" in raw:
            raw_task, raw_variant = raw.split("::variation::", 1)
            t = t or _clean(raw_task)
            v = v or _clean(raw_variant)
        else:
            t = t or raw

    if t and v and v != t:
        return f"{t}::{v}"
    return t


def _row_sample_key(row: pd.Series) -> Optional[str]:
    """
    Return a stable per-sample key for a pairwise row.

    Prefer (task_id, variant_id) when available; otherwise use task_id.
    """
    return _canonical_sample_key(
        task_id=row.get("task_id"),
        variant_id=row.get("variant_id"),
        raw_task_id=row.get("raw_task_id"),
    )


def _compute_objective_outcomes_for_group(
    group: pd.DataFrame,
    objective_lookup: Dict[Tuple[str, str, str], Dict[str, Optional[float]]],
    k: int,
) -> Dict[str, Any]:
    """
    Compute objective pass@k win/tie rates for a grouped pairwise set.

    Treat equal scores as ties. Comparisons with missing metrics on either side are skipped.
    """
    total = 0
    a_wins = 0
    b_wins = 0
    ties = 0

    for _, row in group.iterrows():
        sample_key = _row_sample_key(row)
        variant_label = row.get("variant_label")
        if variant_label is None or (hasattr(pd, "isna") and pd.isna(variant_label)):
            variant_label = "original"
        model_a = row.get("model_a_name")
        model_b = row.get("model_b_name")
        if not sample_key or not model_a or not model_b:
            continue

        metrics_a = objective_lookup.get((sample_key, variant_label, model_a), {})
        metrics_b = objective_lookup.get((sample_key, variant_label, model_b), {})

        score_a = metrics_a.get(f"pass_at_{k}")
        score_b = metrics_b.get(f"pass_at_{k}")

        if score_a is None or score_b is None:
            continue

        total += 1
        if score_a > score_b:
            a_wins += 1
        elif score_b > score_a:
            b_wins += 1
        else:
            ties += 1

    def _rate(count: int) -> Optional[float]:
        return count / total if total > 0 else None

    return {
        f"obj_pass_at_{k}_comparisons": total,
        f"obj_pass_at_{k}_model_a_wins": a_wins,
        f"obj_pass_at_{k}_model_b_wins": b_wins,
        f"obj_pass_at_{k}_ties": ties,
        f"obj_pass_at_{k}_model_a_win_rate": _rate(a_wins),
        f"obj_pass_at_{k}_model_b_win_rate": _rate(b_wins),
        f"obj_pass_at_{k}_tie_rate": _rate(ties),
    }


def _compute_plus_pass_at_1_outcomes_for_group(
    group: pd.DataFrame,
    objective_lookup: Dict[Tuple[str, str, str], Dict[str, Optional[float]]],
) -> Dict[str, Any]:
    """
    Compute plus-test Pass@1 win/tie rates for a grouped pairwise set.

    Treat equal scores as ties. Comparisons with missing metrics on either side are skipped.
    """
    total = 0
    a_wins = 0
    b_wins = 0
    ties = 0

    for _, row in group.iterrows():
        sample_key = _row_sample_key(row)
        variant_label = row.get("variant_label")
        if variant_label is None or (hasattr(pd, "isna") and pd.isna(variant_label)):
            variant_label = "original"
        model_a = row.get("model_a_name")
        model_b = row.get("model_b_name")
        if not sample_key or not model_a or not model_b:
            continue

        metrics_a = objective_lookup.get((sample_key, variant_label, model_a), {})
        metrics_b = objective_lookup.get((sample_key, variant_label, model_b), {})

        score_a = metrics_a.get("plus_pass_at_1")
        score_b = metrics_b.get("plus_pass_at_1")

        if score_a is None or score_b is None:
            continue

        total += 1
        if score_a > score_b:
            a_wins += 1
        elif score_b > score_a:
            b_wins += 1
        else:
            ties += 1

    def _rate(count: int) -> Optional[float]:
        return count / total if total > 0 else None

    return {
        "obj_plus_pass_at_1_comparisons": total,
        "obj_plus_pass_at_1_model_a_wins": a_wins,
        "obj_plus_pass_at_1_model_b_wins": b_wins,
        "obj_plus_pass_at_1_ties": ties,
        "obj_plus_pass_at_1_model_a_win_rate": _rate(a_wins),
        "obj_plus_pass_at_1_model_b_win_rate": _rate(b_wins),
        "obj_plus_pass_at_1_tie_rate": _rate(ties),
    }


def compute_statistical_significance(
    pairwise_df: pd.DataFrame,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    group_by_variant: bool = False,
    group_by_judge: bool = False,
    group_by_user: bool = False,
) -> pd.DataFrame:
    """
    Compute statistical significance tests for pairwise comparisons.

    For each model pair, computes:
    - Binomial test p-value (H0: win probability = 0.5)
    - Sign test p-value
    - Bootstrap confidence interval for win rate

    Args:
        pairwise_df: Sample-level pairwise comparison data.
        confidence_level: Confidence level for intervals (default 0.95).
        n_bootstrap: Number of bootstrap samples (default 1000).
        group_by_variant: If True, group by variant_label. If False, aggregate
            across all variants for a combined view.
        group_by_judge: If True, group by judge_model_name. If False, aggregate
            across all judge models.
        group_by_user: If True, group by user_id. If False, aggregate
            across all users/personas.

    Returns:
        DataFrame with statistical test results per model pair.
    """
    if pairwise_df.empty:
        return pd.DataFrame()

    group_cols = ["model_a_name", "model_b_name", "model_pair"]
    if "pairwise_judgment_type" in pairwise_df.columns:
        group_cols.append("pairwise_judgment_type")
    # Only group by user_id if explicitly requested
    if group_by_user and "user_id" in pairwise_df.columns:
        group_cols.append("user_id")
    # Only group by judge_model_name if explicitly requested
    if group_by_judge and "judge_model_name" in pairwise_df.columns:
        group_cols.append("judge_model_name")
    # Only group by variant_label if explicitly requested
    if group_by_variant and "variant_label" in pairwise_df.columns:
        group_cols.append("variant_label")

    alpha = 1 - confidence_level
    records: List[Dict[str, Any]] = []

    for keys, group in pairwise_df.groupby(group_cols, dropna=False):
        context = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))

        # Count wins (excluding ties)
        a_wins = (group["overall_winner_label"] == "model_a").sum()
        b_wins = (group["overall_winner_label"] == "model_b").sum()
        n_total = len(group)

        # Binomial test (two-sided): H0 = p(A wins) = 0.5
        if n_total > 0:
            binom_result = stats.binomtest(a_wins, n_total, p=0.5)
            binom_pvalue = binom_result.pvalue
            binom_ci = binom_result.proportion_ci(confidence_level=confidence_level)
        else:
            binom_pvalue = 1.0
            binom_ci = (0.0, 1.0)

        # Sign test (equivalent to binomial for this case)
        sign_pvalue = binom_pvalue

        # Bootstrap CI for win rate
        win_rate = a_wins / n_total if n_total > 0 else 0.5
        ci_lower, ci_upper = _bootstrap_ci(
            a_wins, n_total, n_bootstrap, confidence_level
        )

        record = {
            **context,
            "n_comparisons_excl_ties": n_total,
            "model_a_wins": int(a_wins),
            "model_b_wins": int(b_wins),
            "model_a_win_prob": win_rate,
            "binomial_pvalue": binom_pvalue,
            "sign_test_pvalue": sign_pvalue,
            "significant_at_05": binom_pvalue < 0.05,
            "significant_at_01": binom_pvalue < 0.01,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "binom_ci_lower": binom_ci[0],
            "binom_ci_upper": binom_ci[1],
        }
        records.append(record)

    return pd.DataFrame(records)


def _bootstrap_ci(
    successes: int,
    n: int,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a proportion.

    Args:
        successes: Number of successes.
        n: Total number of trials.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if n == 0:
        return (0.0, 1.0)

    rng = np.random.default_rng(seed=42)
    p_hat = successes / n

    # Generate bootstrap samples
    bootstrap_props = rng.binomial(n, p_hat, size=n_bootstrap) / n

    # Compute percentile CI
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_props, 100 * alpha / 2)
    upper = np.percentile(bootstrap_props, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))


def compute_model_rankings(pair_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate model rankings from pairwise win rates.

    Uses average win rate across all pairs as the ranking metric.

    Args:
        pair_summary: Output from compute_pair_summary().

    Returns:
        DataFrame with one row per model, sorted by average win rate.
    """
    if pair_summary.empty:
        return pd.DataFrame()

    # Collect win rates for each model
    model_stats: Dict[str, List[float]] = {}

    for _, row in pair_summary.iterrows():
        model_a = row["model_a_name"]
        model_b = row["model_b_name"]
        a_rate = row["model_a_win_rate"]
        b_rate = row["model_b_win_rate"]

        if model_a not in model_stats:
            model_stats[model_a] = []
        if model_b not in model_stats:
            model_stats[model_b] = []

        model_stats[model_a].append(a_rate)
        model_stats[model_b].append(b_rate)

    # Compute average win rate per model
    records = []
    for model, rates in model_stats.items():
        records.append(
            {
                "model_name": model,
                "num_pairs": len(rates),
                "avg_win_rate": np.mean(rates) if rates else 0.5,
                "min_win_rate": np.min(rates) if rates else 0.5,
                "max_win_rate": np.max(rates) if rates else 0.5,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("avg_win_rate", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

    return df
