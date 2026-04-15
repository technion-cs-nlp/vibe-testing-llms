"""
Pairwise explorer statistics helpers.

This module contains pure functions used by interactive UIs (Streamlit) to:
- Build per-sample overall winner tables across judges.
- Compute agreement metrics (percent agreement, Cohen's kappa) between judges.

The functions are intentionally UI-agnostic and testable with pytest.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS
from src.vibe_testing.analysis.judge_utils import (
    HUMAN_JUDGE_PREFIXES,
    HUMAN_JUDGE_TOKEN_PREFIXES,
    is_human_judge_token,
    split_judges_by_group,
)
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_GENERAL_USER,
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
)

Outcome = str  # e.g. {"<model_a>","<model_b>","tie"} in explorer terms


@dataclass(frozen=True)
class JudgePairAgreement:
    """
    Agreement metrics for a judge pair.

    Attributes:
        judge_a: First judge name.
        judge_b: Second judge name.
        n_common: Number of samples where both judges have outcomes.
        percent_agreement: Proportion of identical outcomes (including ties).
        cohens_kappa: Cohen's kappa for 3-category outcomes (including ties).
        percent_agreement_excl_ties: Agreement over non-tie samples only (optional).
        cohens_kappa_excl_ties: Kappa over non-tie samples only (optional).
    """

    judge_a: str
    judge_b: str
    n_common: int
    percent_agreement: Optional[float]
    cohens_kappa: Optional[float]
    percent_agreement_excl_ties: Optional[float]
    cohens_kappa_excl_ties: Optional[float]


def compute_cohens_kappa(
    labels_a: Sequence[Outcome],
    labels_b: Sequence[Outcome],
    *,
    categories: Sequence[Outcome],
) -> Optional[float]:
    """
    Compute Cohen's kappa for categorical labels.

    Args:
        labels_a: Labels from rater A.
        labels_b: Labels from rater B.
        categories: All possible categories (order defines contingency layout).

    Returns:
        Cohen's kappa, or None when undefined (e.g., empty input).
    """

    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length.")
    n = int(len(labels_a))
    if n == 0:
        return None

    cat_to_idx = {str(c): i for i, c in enumerate(categories)}
    k = int(len(categories))
    if k < 2:
        return None

    mat = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        ia = cat_to_idx.get(str(a))
        ib = cat_to_idx.get(str(b))
        if ia is None or ib is None:
            raise ValueError(f"Unexpected label value: {a!r}, {b!r}")
        mat[ia, ib] += 1.0

    po = float(np.trace(mat) / n)
    row_marg = mat.sum(axis=1) / n
    col_marg = mat.sum(axis=0) / n
    pe = float(np.dot(row_marg, col_marg))

    denom = 1.0 - pe
    if denom <= 0:
        # Perfect expected agreement (degenerate); kappa undefined in practice.
        return None
    return float((po - pe) / denom)


def compute_percent_agreement(
    labels_a: Sequence[Outcome], labels_b: Sequence[Outcome]
) -> Optional[float]:
    """
    Compute percent agreement between two label sequences.

    Args:
        labels_a: Labels from rater A.
        labels_b: Labels from rater B.

    Returns:
        Agreement proportion, or None when undefined (empty).
    """

    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length.")
    n = int(len(labels_a))
    if n == 0:
        return None
    same = sum(1 for a, b in zip(labels_a, labels_b) if str(a) == str(b))
    return float(same / n)


def compute_judge_pair_agreement(
    outcomes_by_judge: Dict[str, Dict[str, Outcome]],
    *,
    task_ids: Sequence[str],
    categories: Sequence[Outcome] = ("X", "Y", "tie"),
) -> pd.DataFrame:
    """
    Compute agreement metrics for all judge pairs.

    Args:
        outcomes_by_judge: Mapping judge -> (task_id -> outcome).
        task_ids: Ordered task_ids defining the analysis set.
        categories: Outcome categories.

    Returns:
        DataFrame with one row per judge pair.
    """

    judges = sorted(outcomes_by_judge.keys())
    rows: List[Dict[str, Any]] = []

    for judge_a, judge_b in combinations(judges, 2):
        map_a = outcomes_by_judge.get(judge_a, {})
        map_b = outcomes_by_judge.get(judge_b, {})

        common: List[Tuple[str, Outcome, Outcome]] = []
        for tid in task_ids:
            if tid not in map_a or tid not in map_b:
                continue
            common.append((tid, map_a[tid], map_b[tid]))

        labels_a = [x[1] for x in common]
        labels_b = [x[2] for x in common]
        n_common = int(len(common))

        percent = compute_percent_agreement(labels_a, labels_b)
        kappa = compute_cohens_kappa(labels_a, labels_b, categories=categories)

        # Exclude ties (both sides must be non-tie)
        excl = [(a, b) for a, b in zip(labels_a, labels_b) if a != "tie" and b != "tie"]
        excl_a = [x[0] for x in excl]
        excl_b = [x[1] for x in excl]
        percent_excl = compute_percent_agreement(excl_a, excl_b)
        non_tie_categories = tuple(c for c in categories if str(c) != "tie")
        kappa_excl = (
            compute_cohens_kappa(excl_a, excl_b, categories=non_tie_categories)
            if excl and len(non_tie_categories) >= 2
            else None
        )

        rows.append(
            {
                "judge_a": judge_a,
                "judge_b": judge_b,
                "n_common": n_common,
                "percent_agreement": percent,
                "cohens_kappa": kappa,
                "n_common_excl_ties": int(len(excl)),
                "percent_agreement_excl_ties": percent_excl,
                "cohens_kappa_excl_ties": kappa_excl,
            }
        )

    return pd.DataFrame(rows)


def build_overall_winner_table(
    *,
    task_ids: Sequence[str],
    judges: Sequence[str],
    outcomes_by_judge: Dict[str, Dict[str, Outcome]],
) -> pd.DataFrame:
    """
    Build a wide table with one row per task and one column per judge outcome.

    Args:
        task_ids: Ordered task_ids.
        judges: Ordered list of judge names.
        outcomes_by_judge: Judge -> (task_id -> outcome).

    Returns:
        DataFrame with columns: task_id, n_judges_present, n_disagreeing, majority_winner, judge columns.
    """

    rows: List[Dict[str, Any]] = []
    for tid in task_ids:
        row: Dict[str, Any] = {"task_id": tid}
        present: List[Outcome] = []
        for judge in judges:
            outcome = outcomes_by_judge.get(judge, {}).get(tid)
            if outcome is None:
                row[judge] = None
            else:
                row[judge] = outcome
                present.append(outcome)

        unique_present = sorted({x for x in present})
        row["n_judges_present"] = int(len(present))
        row["n_disagreeing"] = int(max(0, len(unique_present) - 1))

        majority = None
        if present:
            counts: Dict[str, int] = {}
            for val in present:
                counts[str(val)] = counts.get(str(val), 0) + 1
            best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if best:
                # Tie for top count -> mark None to avoid misleading majority.
                top_count = best[0][1]
                if sum(1 for _, c in best if c == top_count) == 1:
                    majority = best[0][0]
        row["majority_winner"] = majority
        rows.append(row)

    df = pd.DataFrame(rows)
    # Stable column order: meta columns first, then judge columns.
    cols = ["task_id", "majority_winner", "n_judges_present", "n_disagreeing"] + list(
        judges
    )
    cols_existing = [c for c in cols if c in df.columns]
    return df[cols_existing]


def build_overall_winner_table_with_meta(
    *,
    item_ids: Sequence[str],
    judges: Sequence[str],
    outcomes_by_judge: Dict[str, Dict[str, Outcome]],
    item_meta_by_id: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a wide winners table over arbitrary item_ids, with per-item metadata.

    Args:
        item_ids: Ordered item identifiers.
        judges: Ordered judge names.
        outcomes_by_judge: Judge -> (item_id -> outcome) mapping.
        item_meta_by_id: item_id -> metadata dict.

    Returns:
        DataFrame with meta columns + item_id + majority_winner/n_judges_present/n_disagreeing + judge columns.
    """

    rows: List[Dict[str, Any]] = []
    for item_id in item_ids:
        meta = dict(item_meta_by_id.get(str(item_id), {}) or {})
        row: Dict[str, Any] = {"item_id": str(item_id), **meta}

        present: List[Outcome] = []
        for judge in judges:
            outcome = outcomes_by_judge.get(str(judge), {}).get(str(item_id))
            if outcome is None:
                row[str(judge)] = None
            else:
                row[str(judge)] = outcome
                present.append(outcome)

        unique_present = sorted({x for x in present})
        row["n_judges_present"] = int(len(present))
        row["n_disagreeing"] = int(max(0, len(unique_present) - 1))

        majority = None
        if present:
            counts: Dict[str, int] = {}
            for val in present:
                counts[str(val)] = counts.get(str(val), 0) + 1
            best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if best:
                top_count = best[0][1]
                if sum(1 for _, c in best if c == top_count) == 1:
                    majority = best[0][0]
        row["majority_winner"] = majority
        rows.append(row)

    df = pd.DataFrame(rows)

    meta_order = [
        "item_id",
        "persona",
        "generator_model",
        "filter_model",
        "prompt_type",
        "model_pair",
        "model_1",
        "model_2",
        "task_id",
        "majority_winner",
        "n_judges_present",
        "n_disagreeing",
    ]
    meta_existing = [c for c in meta_order if c in df.columns]
    other_meta = [
        c for c in df.columns if c not in meta_existing and c not in set(judges)
    ]
    cols = meta_existing + other_meta + [str(j) for j in judges if str(j) in df.columns]
    cols_existing = [c for c in cols if c in df.columns]
    return df[cols_existing]


def build_long_winner_table_with_meta(
    *,
    item_ids: Sequence[str],
    judges: Sequence[str],
    outcomes_by_judge: Dict[str, Dict[str, Outcome]],
    item_meta_by_id: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a long winners table with one row per (item_id, judge), with metadata.

    Args:
        item_ids: Ordered item identifiers.
        judges: Ordered judge names.
        outcomes_by_judge: Judge -> (item_id -> outcome) mapping.
        item_meta_by_id: item_id -> metadata dict.

    Returns:
        DataFrame with columns: item_id, meta..., judge, outcome.
    """

    rows: List[Dict[str, Any]] = []
    for item_id in item_ids:
        meta = dict(item_meta_by_id.get(str(item_id), {}) or {})
        for judge in judges:
            outcome = outcomes_by_judge.get(str(judge), {}).get(str(item_id))
            rows.append(
                {
                    "item_id": str(item_id),
                    **meta,
                    "judge": str(judge),
                    "outcome": outcome,
                }
            )
    df = pd.DataFrame(rows)

    meta_order = [
        "item_id",
        "persona",
        "generator_model",
        "filter_model",
        "prompt_type",
        "model_pair",
        "model_1",
        "model_2",
        "task_id",
        "judge",
        "outcome",
    ]
    cols = [c for c in meta_order if c in df.columns] + [
        c for c in df.columns if c not in meta_order
    ]
    cols_existing = [c for c in cols if c in df.columns]
    return df[cols_existing]


def compute_judge_pair_agreement_by_group(
    outcomes_by_judge: Dict[str, Dict[str, Outcome]],
    *,
    group_to_item_ids: Dict[str, Sequence[str]],
    categories: Sequence[Outcome] = ("X", "Y", "tie"),
    group_column_name: str = "model_pair",
) -> pd.DataFrame:
    """
    Compute judge-pair agreement metrics separately for each group.

    Args:
        outcomes_by_judge: Mapping judge -> (item_id -> outcome).
        group_to_item_ids: Mapping group key -> item_ids in the group.
        categories: Outcome categories.
        group_column_name: Name of the group column in the output.

    Returns:
        DataFrame with group column + judge-pair agreement metrics.
    """

    rows: List[Dict[str, Any]] = []
    for group, item_ids in sorted(group_to_item_ids.items(), key=lambda kv: str(kv[0])):
        df = compute_judge_pair_agreement(
            outcomes_by_judge, task_ids=list(item_ids), categories=categories
        )
        if df.empty:
            continue
        for rec in df.to_dict(orient="records"):
            rec[group_column_name] = str(group)
            rows.append(rec)

    if not rows:
        return pd.DataFrame(
            columns=[
                group_column_name,
                "judge_a",
                "judge_b",
                "n_common",
                "percent_agreement",
                "cohens_kappa",
                "n_common_excl_ties",
                "percent_agreement_excl_ties",
                "cohens_kappa_excl_ties",
            ]
        )

    out = pd.DataFrame(rows)
    col_order = [group_column_name] + [c for c in out.columns if c != group_column_name]
    return out[col_order]


# is_human_judge_token and split_judges_by_group are imported from
# src.vibe_testing.analysis.judge_utils and re-exported for backward compat.


def align_pairwise_judgment_types_to_human_scope(
    frame: pd.DataFrame,
    *,
    item_key_column: str,
    judge_column: str,
    judgment_type_column: str = "pairwise_judgment_type",
) -> pd.DataFrame:
    """
    Align per-item judgment types to the human-judged scope.

    Human rows default to ``persona`` unless their existing judgment type is
    explicitly ``general_user``. For each item key that has at least one human
    row, all matching rows are then aligned to the same human-derived judgment
    type so human-vs-LLM comparisons stay within one bucket.

    Args:
        frame: Row-level pairwise frame containing item keys, judge tokens, and
            optionally judgment types.
        item_key_column: Column whose values identify the same conceptual item
            across judges, excluding judgment type.
        judge_column: Column containing judge identifiers.
        judgment_type_column: Column containing the pairwise judgment type.

    Returns:
        pd.DataFrame: Copy of ``frame`` with aligned ``judgment_type_column``.

    Raises:
        ValueError: If required columns are missing or human rows for the same
            item key carry conflicting explicit judgment types.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])

    missing = [
        col
        for col in [item_key_column, judge_column]
        if col not in frame.columns
    ]
    if missing:
        raise ValueError(
            "align_pairwise_judgment_types_to_human_scope requires columns: "
            + ", ".join(missing)
        )

    out = frame.copy()
    if judgment_type_column not in out.columns:
        out[judgment_type_column] = PAIRWISE_JUDGMENT_TYPE_PERSONA

    def _normalize_type(value: object) -> str:
        token = str(value or "").strip()
        if not token:
            return PAIRWISE_JUDGMENT_TYPE_PERSONA
        return normalize_pairwise_judgment_type(token)

    out[judgment_type_column] = out[judgment_type_column].map(_normalize_type)
    out[item_key_column] = out[item_key_column].astype(str)
    out[judge_column] = out[judge_column].astype(str)

    human_mask = out[judge_column].map(is_human_judge_token)
    if not human_mask.any():
        return out

    # Human rows default to persona unless explicitly tagged as general_user.
    out.loc[
        human_mask
        & (out[judgment_type_column] != PAIRWISE_JUDGMENT_TYPE_GENERAL_USER),
        judgment_type_column,
    ] = PAIRWISE_JUDGMENT_TYPE_PERSONA

    human_rows = out.loc[human_mask, [item_key_column, judgment_type_column]].copy()
    if human_rows.empty:
        return out

    human_targets: Dict[str, str] = {}
    for item_key, group in human_rows.groupby(item_key_column, sort=False):
        types = sorted(set(group[judgment_type_column].astype(str).tolist()))
        if len(types) > 1:
            raise ValueError(
                "Conflicting human judgment types found for the same item key. "
                f"item_key={item_key!r} types={types!r}"
            )
        human_targets[str(item_key)] = str(types[0])

    target_series = out[item_key_column].map(human_targets)
    has_target = target_series.notna()
    out.loc[has_target, judgment_type_column] = target_series.loc[has_target].astype(str)
    return out


def filter_item_ids_to_human_annotated(
    item_ids: Sequence[str],
    *,
    outcomes_by_judge: Dict[str, Dict[str, Outcome]],
    human_judges: Sequence[str],
) -> List[str]:
    """
    Keep only item_ids that have at least one human-judge outcome.

    Args:
        item_ids: Candidate item ids.
        outcomes_by_judge: Judge -> (item_id -> outcome) mapping.
        human_judges: Judge tokens recognized as human annotators.

    Returns:
        List[str]: Item ids with at least one human judgment.
    """
    human_set = {str(j) for j in human_judges}
    if not human_set:
        return []

    filtered: List[str] = []
    for item_id in item_ids:
        item_key = str(item_id)
        has_human = False
        for judge in human_set:
            if outcomes_by_judge.get(judge, {}).get(item_key) is not None:
                has_human = True
                break
        if has_human:
            filtered.append(item_key)
    return filtered


def filter_item_ids_by_human_confidence(
    item_ids: Sequence[str],
    *,
    human_confidence_by_judge: Dict[str, Dict[str, str]],
    human_judges: Sequence[str],
    allowed_confidence_levels: Sequence[str],
) -> List[str]:
    """
    Keep only item_ids that have at least one human judgment with allowed confidence.

    Args:
        item_ids: Candidate item ids.
        human_confidence_by_judge: Judge -> (item_id -> human_overall_confidence).
        human_judges: Judge tokens recognized as human annotators.
        allowed_confidence_levels: Confidence labels to retain.

    Returns:
        List[str]: Item ids with at least one matching human confidence value.
    """
    human_set = {str(j) for j in human_judges}
    allowed_set = {str(x) for x in allowed_confidence_levels if str(x).strip()}
    if not human_set or not allowed_set:
        return []

    filtered: List[str] = []
    for item_id in item_ids:
        item_key = str(item_id)
        keep = False
        for judge in human_set:
            confidence = str(
                human_confidence_by_judge.get(judge, {}).get(item_key, "")
            ).strip()
            if confidence and confidence in allowed_set:
                keep = True
                break
        if keep:
            filtered.append(item_key)
    return filtered


def filter_agreement_df_to_within_judge_group(
    agreement_df: pd.DataFrame,
    *,
    judges_in_group: Sequence[str],
) -> pd.DataFrame:
    """
    Keep only agreement rows where both judges belong to the same target group.

    Args:
        agreement_df: Output of compute_judge_pair_agreement(...).
        judges_in_group: Judge tokens that define the group.

    Returns:
        pd.DataFrame: Filtered agreement rows.
    """
    if agreement_df is None or agreement_df.empty:
        return pd.DataFrame(
            columns=list(agreement_df.columns) if agreement_df is not None else []
        )

    group_set = {str(j) for j in judges_in_group}
    if not group_set:
        return agreement_df.iloc[0:0].copy()

    if "judge_a" not in agreement_df.columns or "judge_b" not in agreement_df.columns:
        raise ValueError("agreement_df must include 'judge_a' and 'judge_b' columns.")

    mask = agreement_df["judge_a"].astype(str).isin(group_set) & agreement_df[
        "judge_b"
    ].astype(str).isin(group_set)
    return agreement_df.loc[mask].copy()


def filter_agreement_df_between_judge_groups(
    agreement_df: pd.DataFrame,
    *,
    judges_in_group_a: Sequence[str],
    judges_in_group_b: Sequence[str],
) -> pd.DataFrame:
    """
    Keep only agreement rows where judges come from opposite target groups.

    Args:
        agreement_df: Output of compute_judge_pair_agreement(...).
        judges_in_group_a: Judge tokens that define the first group.
        judges_in_group_b: Judge tokens that define the second group.

    Returns:
        pd.DataFrame: Filtered agreement rows with one judge from each group.
    """
    if agreement_df is None or agreement_df.empty:
        return pd.DataFrame(
            columns=list(agreement_df.columns) if agreement_df is not None else []
        )

    group_a = {str(j) for j in judges_in_group_a}
    group_b = {str(j) for j in judges_in_group_b}
    if not group_a or not group_b:
        return agreement_df.iloc[0:0].copy()

    if "judge_a" not in agreement_df.columns or "judge_b" not in agreement_df.columns:
        raise ValueError("agreement_df must include 'judge_a' and 'judge_b' columns.")

    judge_a_series = agreement_df["judge_a"].astype(str)
    judge_b_series = agreement_df["judge_b"].astype(str)
    mask = (judge_a_series.isin(group_a) & judge_b_series.isin(group_b)) | (
        judge_a_series.isin(group_b) & judge_b_series.isin(group_a)
    )
    return agreement_df.loc[mask].copy()


def summarize_human_vs_llm_agreement_means(
    agreement_df: pd.DataFrame,
    *,
    human_judges: Sequence[str],
    llm_judges: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize mean agreement for each human annotator against all LLM judges.

    Args:
        agreement_df: Output of compute_judge_pair_agreement(...).
        human_judges: Judge tokens recognized as human annotators.
        llm_judges: Judge tokens recognized as LLM judges.

    Returns:
        pd.DataFrame: One row per human judge with mean/std metrics across
            all human-vs-LLM judge-pair rows.
    """
    expected_columns = [
        "human_judge",
        "percent_agreement_mean",
        "percent_agreement_excl_ties_mean",
        "cohens_kappa_mean",
        "cohens_kappa_excl_ties_mean",
        "percent_agreement_std",
        "percent_agreement_excl_ties_std",
        "cohens_kappa_std",
        "cohens_kappa_excl_ties_std",
        "percent_agreement_n_pairs",
        "percent_agreement_excl_ties_n_pairs",
        "cohens_kappa_n_pairs",
        "cohens_kappa_excl_ties_n_pairs",
        "total_items_used",
        "total_items_used_excl_ties",
    ]
    cross_group_df = filter_agreement_df_between_judge_groups(
        agreement_df,
        judges_in_group_a=human_judges,
        judges_in_group_b=llm_judges,
    )
    if cross_group_df.empty:
        return pd.DataFrame(columns=expected_columns)

    human_set = {str(j) for j in human_judges}
    normalized = cross_group_df.copy()
    normalized["human_judge"] = normalized.apply(
        lambda row: str(row["judge_a"])
        if str(row["judge_a"]) in human_set
        else str(row["judge_b"]),
        axis=1,
    )

    summary_df = _summarize_agreement_mean_std_by_group(
        normalized,
        group_columns=("human_judge",),
    )
    return summary_df[expected_columns]


def _dimension_names_from_sample_df(
    sample_df: pd.DataFrame,
    *,
    dimensions: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Infer dimension names from a Stage-6-style sample frame.

    Args:
        sample_df: Sample frame with `dim_<name>_winner_label` columns.
        dimensions: Optional explicit dimension ordering.

    Returns:
        List[str]: Ordered dimension names present in the frame.
    """
    if dimensions is not None:
        requested = [str(dim) for dim in dimensions]
        return [
            dim for dim in requested if f"dim_{dim}_winner_label" in sample_df.columns
        ]

    present = {
        str(col)[4:-13]
        for col in sample_df.columns
        if str(col).startswith("dim_") and str(col).endswith("_winner_label")
    }
    return [str(dim) for dim in PAIRWISE_DIMENSIONS if str(dim) in present]


def _summarize_agreement_mean_std_by_group(
    agreement_df: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize agreement metrics by one or more grouping columns.

    Args:
        agreement_df: Agreement rows containing percent agreement and kappa
            columns, including tie-excluded variants when available.
        group_columns: Columns that define each summary row.

    Returns:
        pd.DataFrame: Mean/std summaries grouped by the requested columns.
    """
    metric_specs = [
        ("percent_agreement", "percent_agreement_mean", "percent_agreement_std", "percent_agreement_n_pairs"),
        (
            "percent_agreement_excl_ties",
            "percent_agreement_excl_ties_mean",
            "percent_agreement_excl_ties_std",
            "percent_agreement_excl_ties_n_pairs",
        ),
        ("cohens_kappa", "cohens_kappa_mean", "cohens_kappa_std", "cohens_kappa_n_pairs"),
        (
            "cohens_kappa_excl_ties",
            "cohens_kappa_excl_ties_mean",
            "cohens_kappa_excl_ties_std",
            "cohens_kappa_excl_ties_n_pairs",
        ),
    ]
    mean_columns = [mean_col for _, mean_col, _, _ in metric_specs]
    std_columns = [std_col for _, _, std_col, _ in metric_specs]
    n_columns = [n_col for _, _, _, n_col in metric_specs]
    expected_columns = [
        *[str(col) for col in group_columns],
        *mean_columns,
        *std_columns,
        *n_columns,
        "total_items_used",
        "total_items_used_excl_ties",
    ]
    if agreement_df is None or agreement_df.empty:
        return pd.DataFrame(columns=expected_columns)

    missing = [str(col) for col in group_columns if str(col) not in agreement_df.columns]
    if missing:
        raise ValueError(
            "agreement_df is missing required grouping columns: "
            + ", ".join(missing)
        )

    rows: List[Dict[str, Any]] = []
    for keys, group_df in agreement_df.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = {
            str(col): value for col, value in zip(group_columns, keys)
        }

        def _mean_std(series: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
            n = int(len(series))
            if n <= 0:
                return None, None, 0
            mean = float(series.mean())
            std = float(series.std(ddof=1)) if n > 1 else 0.0
            return mean, std, n

        for source_col, mean_col, std_col, n_col in metric_specs:
            series = pd.to_numeric(group_df.get(source_col), errors="coerce").dropna()
            mean_value, std_value, n_value = _mean_std(series)
            row[mean_col] = mean_value
            row[std_col] = std_value
            row[n_col] = n_value
        row["total_items_used"] = int(
            pd.to_numeric(group_df.get("n_common"), errors="coerce").dropna().sum()
        )
        row["total_items_used_excl_ties"] = int(
            pd.to_numeric(group_df.get("n_common_excl_ties"), errors="coerce")
            .dropna()
            .sum()
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=expected_columns)


def compute_dimension_judge_pair_agreement(
    sample_df: pd.DataFrame,
    *,
    item_id_column: str = "item_id",
    judge_column: str = "judge_model_name",
    dimensions: Optional[Sequence[str]] = None,
    categories: Sequence[str] = ("model_a", "model_b", "tie"),
) -> pd.DataFrame:
    """
    Compute judge-pair agreement separately for each dimension.

    Args:
        sample_df: Stage-6-style sample frame with one row per (item, judge).
        item_id_column: Column that uniquely identifies an aggregate item.
        judge_column: Column storing judge identity.
        dimensions: Optional explicit dimension ordering.
        categories: Outcome categories for dimension winner labels.

    Returns:
        pd.DataFrame: Long agreement table with one row per
            `(dimension, judge_a, judge_b)`.
    """
    if sample_df is None or sample_df.empty:
        return pd.DataFrame()
    if item_id_column not in sample_df.columns or judge_column not in sample_df.columns:
        raise ValueError(
            f"sample_df must include {item_id_column!r} and {judge_column!r} columns."
        )

    dims_present = _dimension_names_from_sample_df(sample_df, dimensions=dimensions)
    if not dims_present:
        raise ValueError(
            "sample_df is missing per-dimension winner label columns "
            "(expected columns like 'dim_clarity_winner_label')."
        )

    task_ids = sorted(sample_df[item_id_column].dropna().astype(str).unique().tolist())
    rows: List[pd.DataFrame] = []
    for dim in dims_present:
        winner_col = f"dim_{dim}_winner_label"
        outcomes_by_judge: Dict[str, Dict[str, str]] = {}
        for _, row in sample_df[[item_id_column, judge_column, winner_col]].iterrows():
            item_id = str(row.get(item_id_column, "")).strip()
            judge = str(row.get(judge_column, "")).strip()
            winner = str(row.get(winner_col, "")).strip()
            if not item_id or not judge or not winner:
                continue
            judge_map = outcomes_by_judge.setdefault(judge, {})
            existing = judge_map.get(item_id)
            if existing is not None and existing != winner:
                raise ValueError(
                    "Conflicting dimension winner labels for the same item and judge. "
                    f"dimension={dim!r} item_id={item_id!r} judge={judge!r} "
                    f"existing={existing!r} new={winner!r}"
                )
            judge_map[item_id] = winner
        if not outcomes_by_judge:
            continue
        agreement_df = compute_judge_pair_agreement(
            outcomes_by_judge,
            task_ids=task_ids,
            categories=categories,
        )
        if agreement_df.empty:
            continue
        agreement_df.insert(0, "dimension", str(dim))
        rows.append(agreement_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_dimension_agreement_mean_std(
    agreement_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize dimension-level agreement by mean/std across judge pairs.

    Args:
        agreement_df: Output of `compute_dimension_judge_pair_agreement(...)`.

    Returns:
        pd.DataFrame: One row per dimension.
    """
    return _summarize_agreement_mean_std_by_group(
        agreement_df,
        group_columns=("dimension",),
    )


def summarize_dimension_human_vs_llm_means(
    agreement_df: pd.DataFrame,
    *,
    human_judges: Sequence[str],
    llm_judges: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize dimension-level agreement for each human annotator vs all LLM judges.

    Args:
        agreement_df: Output of `compute_dimension_judge_pair_agreement(...)`.
        human_judges: Judge tokens recognized as human annotators.
        llm_judges: Judge tokens recognized as LLM judges.

    Returns:
        pd.DataFrame: One row per `(dimension, human_judge)`.
    """
    expected_columns = [
        "dimension",
        "human_judge",
        "percent_agreement_mean",
        "percent_agreement_excl_ties_mean",
        "cohens_kappa_mean",
        "cohens_kappa_excl_ties_mean",
        "percent_agreement_std",
        "percent_agreement_excl_ties_std",
        "cohens_kappa_std",
        "cohens_kappa_excl_ties_std",
        "percent_agreement_n_pairs",
        "percent_agreement_excl_ties_n_pairs",
        "cohens_kappa_n_pairs",
        "cohens_kappa_excl_ties_n_pairs",
        "total_items_used",
        "total_items_used_excl_ties",
    ]
    cross_group_df = filter_agreement_df_between_judge_groups(
        agreement_df,
        judges_in_group_a=human_judges,
        judges_in_group_b=llm_judges,
    )
    if cross_group_df.empty:
        return pd.DataFrame(columns=expected_columns)

    human_set = {str(j) for j in human_judges}
    normalized = cross_group_df.copy()
    normalized["human_judge"] = normalized.apply(
        lambda row: str(row["judge_a"])
        if str(row["judge_a"]) in human_set
        else str(row["judge_b"]),
        axis=1,
    )
    summary_df = _summarize_agreement_mean_std_by_group(
        normalized,
        group_columns=("dimension", "human_judge"),
    )
    return summary_df[expected_columns]


def is_eval_or_parse_error_rationale(value: object) -> bool:
    """
    Return True if a rationale indicates an evaluation/parsing error.

    This mirrors Stage 6's rule used in `recompute_pairwise_overall_winner`.

    Args:
        value: Rationale text (or any object that can be stringified).

    Returns:
        bool: True if it starts with 'evaluation error:' or 'parse error:' (case-insensitive).
    """

    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    return lowered.startswith("evaluation error:") or lowered.startswith("parse error:")


def _winner_display_name(
    *,
    winner_model: Optional[object],
    winner_value: Optional[object],
    model_a_name: str,
    model_b_name: str,
) -> str:
    """
    Convert Stage-5b dimension winner fields into a display-friendly model name.

    Args:
        winner_model: Optional explicit winner model name (preferred when present).
        winner_value: Winner label ('A', 'B', or 'tie') when winner_model is missing.
        model_a_name: Name of model A for this comparison.
        model_b_name: Name of model B for this comparison.

    Returns:
        str: Winner model name or 'tie'.
    """

    if winner_model is not None and str(winner_model).strip():
        return str(winner_model)
    val = str(winner_value or "tie").strip()
    if val == "A":
        return model_a_name
    if val == "B":
        return model_b_name
    return "tie"


def _confidence_rank(value: object) -> int:
    """
    Map confidence label to an ordinal rank.

    Args:
        value: Confidence value (string-like).

    Returns:
        int: Rank where higher is more confident.
    """

    text = str(value or "low").strip().lower()
    if text == "high":
        return 3
    if text == "medium":
        return 2
    return 1


def recompute_dimension_winner(
    *,
    dim_result: Dict[str, Any],
    mode: str,
    model_a_name: str,
    model_b_name: str,
) -> Tuple[str, str]:
    """
    Recompute (winner_name, confidence) for a single dimension (Stage-5b payload).

    Modes:
    - stored: use stored winner fields (winner_model/winner).
    - strict: if original/swapped order results exist, require agreement; otherwise fall back to stored.
    - finegrained: if original/swapped disagree, break ties using confidence ranks (Stage-6-style).

    Args:
        dim_result: Dimension result payload from Stage 5b.
        mode: One of {'stored','strict','finegrained'}.
        model_a_name: Model A name (from record).
        model_b_name: Model B name (from record).

    Returns:
        Tuple[str, str]: (winner_name_or_tie, confidence_label)
    """

    mode_token = str(mode or "stored").strip().lower()
    if mode_token not in {"stored", "strict", "finegrained"}:
        raise ValueError(f"Unknown tie-breaker mode: {mode!r}")

    # Stored mode: use the record as-is.
    if mode_token == "stored":
        winner_model = dim_result.get("winner_model")
        winner_value = dim_result.get("winner")
        winner_name = _winner_display_name(
            winner_model=winner_model,
            winner_value=winner_value,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        confidence = str(dim_result.get("confidence", "low"))
        return winner_name, confidence

    original_order = dim_result.get("original_order_result")
    swapped_order = dim_result.get("swapped_order_result")
    if not isinstance(original_order, dict) or not isinstance(swapped_order, dict):
        # Without both orders, we cannot recompute; fall back to stored.
        winner_model = dim_result.get("winner_model")
        winner_value = dim_result.get("winner")
        winner_name = _winner_display_name(
            winner_model=winner_model,
            winner_value=winner_value,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        confidence = str(dim_result.get("confidence", "low"))
        return winner_name, confidence

    orig_pos_winner = str(original_order.get("position_winner", "tie"))
    swapped_pos_winner = str(swapped_order.get("position_winner", "tie"))
    orig_conf = str(original_order.get("confidence", "low"))
    swapped_conf = str(swapped_order.get("confidence", "low"))

    # Convert position winners to model winners.
    # Original order: position A = model_a, position B = model_b
    if orig_pos_winner == "A":
        orig_model_winner: Optional[str] = model_a_name
    elif orig_pos_winner == "B":
        orig_model_winner = model_b_name
    else:
        orig_model_winner = None

    # Swapped order: position A = model_b, position B = model_a
    if swapped_pos_winner == "A":
        swapped_model_winner: Optional[str] = model_b_name
    elif swapped_pos_winner == "B":
        swapped_model_winner = model_a_name
    else:
        swapped_model_winner = None

    # Agreement case.
    if orig_model_winner == swapped_model_winner:
        winner_name = orig_model_winner or "tie"
        return winner_name, orig_conf

    # Disagreement: strict mode always ties.
    if mode_token == "strict":
        return "tie", "low"

    # Fine-grained tie breaker (mirrors Stage 6 logic).
    orig_rank = _confidence_rank(orig_conf)
    swapped_rank = _confidence_rank(swapped_conf)

    # One order confident winner (>= medium) and the other is tie.
    if (
        orig_model_winner is not None
        and orig_rank >= 2
        and swapped_model_winner is None
    ):
        return orig_model_winner, orig_conf
    if (
        swapped_model_winner is not None
        and swapped_rank >= 2
        and orig_model_winner is None
    ):
        return swapped_model_winner, swapped_conf

    # Both ties.
    if orig_model_winner is None and swapped_model_winner is None:
        return "tie", "low"

    # One tie but other low-confidence winner: remain tie.
    if orig_model_winner is None or swapped_model_winner is None:
        return "tie", "low"

    # Both winners: pick higher confidence, else tie.
    if orig_rank > swapped_rank:
        return orig_model_winner, orig_conf
    if swapped_rank > orig_rank:
        return swapped_model_winner, swapped_conf
    return "tie", "low"


def recompute_overall_from_dimensions_with_controls(
    *,
    record: Dict[str, Any],
    mode: str,
    omit_pairwise_keys: Iterable[str],
    exclude_eval_parse_errors: bool,
    correctness_mode: str = "ignore",
    correctness_a: Optional[float] = None,
    correctness_b: Optional[float] = None,
    plus_correctness_a: Optional[float] = None,
    plus_correctness_b: Optional[float] = None,
    include_plus_correctness: bool = False,
) -> Tuple[str, int, int, int, bool]:
    """
    Recompute overall winner from Stage-5b dimension_results with omit/exclude controls.

    This is the legacy (raw-artifact) analogue of Stage 6's recomputation:
    - Omitted dimensions do not vote.
    - Error dimensions do not vote when `exclude_eval_parse_errors=True`.
    - If no usable dimensions remain, `valid_for_overall=False`.
    - Correctness can optionally act as a gate or additional dimension.

    Args:
        record: Stage-5b pairwise record.
        mode: Tie-breaker mode ('stored'|'strict'|'finegrained').
        omit_pairwise_keys: Dimension keys to omit (canonical).
        exclude_eval_parse_errors: If True, exclude error dimensions from vote.
        correctness_mode: ``"ignore"`` (default), ``"dimension"``, or ``"gate"``.
        correctness_a: Base pass@1 for model A (None if unavailable).
        correctness_b: Base pass@1 for model B (None if unavailable).
        plus_correctness_a: Plus pass@1 for model A (None if unavailable).
        plus_correctness_b: Plus pass@1 for model B (None if unavailable).
        include_plus_correctness: If True, include plus correctness as a vote.

    Returns:
        Tuple[str, int, int, int, bool]:
            (overall_winner_or_tie, a_wins, b_wins, ties, valid_for_overall)
    """

    model_a_name = str(record.get("model_a_name", "model_a"))
    model_b_name = str(record.get("model_b_name", "model_b"))
    dim_results = record.get("dimension_results") or {}
    if not isinstance(dim_results, dict) or not dim_results:
        return "tie", 0, 0, 0, False

    has_correctness = correctness_a is not None and correctness_b is not None

    # Gate mode: if one model correct and the other not, auto-win
    if correctness_mode == "gate" and has_correctness:
        a_correct = correctness_a > 0
        b_correct = correctness_b > 0
        if a_correct and not b_correct:
            return model_a_name, 0, 0, 0, True
        if b_correct and not a_correct:
            return model_b_name, 0, 0, 0, True

    omit_set = {str(x) for x in (omit_pairwise_keys or [])}

    a = 0
    b = 0
    t = 0
    usable = 0

    for dim_name, dim_payload in dim_results.items():
        dim_key = str(dim_name)
        if dim_key in omit_set:
            continue
        if not isinstance(dim_payload, dict):
            continue
        if exclude_eval_parse_errors and is_eval_or_parse_error_rationale(
            dim_payload.get("rationale", "")
        ):
            continue

        usable += 1
        winner_name, _conf = recompute_dimension_winner(
            dim_result=dim_payload,
            mode=mode,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        if winner_name == model_a_name:
            a += 1
        elif winner_name == model_b_name:
            b += 1
        else:
            t += 1

    # Add base correctness as a virtual dimension vote
    if correctness_mode == "dimension" and has_correctness:
        usable += 1
        if correctness_a > correctness_b:
            a += 1
        elif correctness_b > correctness_a:
            b += 1
        else:
            t += 1

        if (
            include_plus_correctness
            and plus_correctness_a is not None
            and plus_correctness_b is not None
        ):
            usable += 1
            if plus_correctness_a > plus_correctness_b:
                a += 1
            elif plus_correctness_b > plus_correctness_a:
                b += 1
            else:
                t += 1

    valid_for_overall = bool(usable > 0)
    if not valid_for_overall:
        return "tie", 0, 0, 0, False

    if a > b:
        overall = model_a_name
    elif b > a:
        overall = model_b_name
    else:
        overall = "tie"
    return overall, a, b, t, True


def recompute_overall_from_dimensions_with_controls_weighted(
    *,
    record: Dict[str, Any],
    mode: str,
    omit_pairwise_keys: Iterable[str],
    exclude_eval_parse_errors: bool,
    dimension_weights_by_user: Dict[str, Dict[str, float]],
    correctness_mode: str = "ignore",
    correctness_a: Optional[float] = None,
    correctness_b: Optional[float] = None,
    plus_correctness_a: Optional[float] = None,
    plus_correctness_b: Optional[float] = None,
    include_plus_correctness: bool = False,
) -> Tuple[str, float, float, float, bool]:
    """
    Recompute overall winner using persona-specific dimension weights.

    This is the legacy (raw-artifact) analogue of Stage 6's
    `recompute_pairwise_overall_winner_dimension_weighted(...)`:
    - Omitted dimensions do not vote.
    - Error dimensions do not vote when `exclude_eval_parse_errors=True`.
    - Per-dimension winners are computed using the same tie-breaker logic as the UI.
    - Overall winner compares weighted totals for model_a vs model_b (A==B => tie).
      The tie bucket weight does not break A/B ties.
    - Correctness can optionally act as a gate or additional weighted dimension.

    Args:
        record: Stage-5b pairwise record.
        mode: Tie-breaker mode ('stored'|'strict'|'finegrained').
        omit_pairwise_keys: Canonical pairwise dimension keys to omit.
        exclude_eval_parse_errors: If True, exclude error dimensions from vote.
        dimension_weights_by_user: Mapping user_id -> {pairwise_dim -> weight}.
        correctness_mode: ``"ignore"`` (default), ``"dimension"``, or ``"gate"``.
        correctness_a: Base pass@1 for model A (None if unavailable).
        correctness_b: Base pass@1 for model B (None if unavailable).
        plus_correctness_a: Plus pass@1 for model A (None if unavailable).
        plus_correctness_b: Plus pass@1 for model B (None if unavailable).
        include_plus_correctness: If True, include plus correctness as a weighted vote.

    Returns:
        Tuple[str, float, float, float, bool]:
            (overall_winner_or_tie, score_a, score_b, score_tie, valid_for_overall)

    Raises:
        ValueError: If the record has no user_id or the user's weights are missing/invalid.
    """
    from src.vibe_testing.analysis.dimension_omits import CORRECTNESS_DIMENSION_WEIGHT

    user_id = record.get("user_id")
    if user_id is None or (hasattr(pd, "isna") and pd.isna(user_id)):
        raise ValueError("Pairwise record is missing user_id; cannot apply weights.")
    user_key = str(user_id)
    if user_key not in dimension_weights_by_user:
        raise ValueError(
            f"Missing dimension weights for user_id='{user_key}'. "
            "Ensure the selected persona YAML configs cover all personas present."
        )
    dim_weights = dimension_weights_by_user[user_key]

    model_a_name = str(record.get("model_a_name", "model_a"))
    model_b_name = str(record.get("model_b_name", "model_b"))
    dim_results = record.get("dimension_results") or {}
    if not isinstance(dim_results, dict) or not dim_results:
        return "tie", 0.0, 0.0, 0.0, False

    has_correctness = correctness_a is not None and correctness_b is not None

    # Gate mode: if one model correct and the other not, auto-win
    if correctness_mode == "gate" and has_correctness:
        a_correct = correctness_a > 0
        b_correct = correctness_b > 0
        if a_correct and not b_correct:
            return (
                model_a_name,
                float(CORRECTNESS_DIMENSION_WEIGHT),
                0.0,
                0.0,
                True,
            )
        if b_correct and not a_correct:
            return (
                model_b_name,
                0.0,
                float(CORRECTNESS_DIMENSION_WEIGHT),
                0.0,
                True,
            )

    omit_set = {str(x) for x in (omit_pairwise_keys or [])}

    score_a = 0.0
    score_b = 0.0
    score_tie = 0.0
    usable = 0

    for dim in PAIRWISE_DIMENSIONS:
        dim_key = str(dim)
        if dim_key in omit_set:
            continue
        dim_payload = dim_results.get(dim_key)
        if not isinstance(dim_payload, dict):
            continue
        if exclude_eval_parse_errors and is_eval_or_parse_error_rationale(
            dim_payload.get("rationale", "")
        ):
            continue

        if dim_key not in dim_weights:
            raise ValueError(
                f"Dimension weights for user_id='{user_key}' are missing key '{dim_key}'."
            )
        w = float(dim_weights[dim_key])
        if w <= 0:
            raise ValueError(
                f"Non-positive dimension weight for user_id='{user_key}' dim='{dim_key}': {w}"
            )

        usable += 1
        winner_name, _conf = recompute_dimension_winner(
            dim_result=dim_payload,
            mode=mode,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        if winner_name == model_a_name:
            score_a += w
        elif winner_name == model_b_name:
            score_b += w
        else:
            score_tie += w

    # Add base correctness as a weighted virtual dimension
    if correctness_mode == "dimension" and has_correctness:
        cw = CORRECTNESS_DIMENSION_WEIGHT
        usable += 1
        if correctness_a > correctness_b:
            score_a += cw
        elif correctness_b > correctness_a:
            score_b += cw
        else:
            score_tie += cw

        if (
            include_plus_correctness
            and plus_correctness_a is not None
            and plus_correctness_b is not None
        ):
            plus_w = float(dim_weights.get("workflow_fit", 1.0))
            usable += 1
            if plus_correctness_a > plus_correctness_b:
                score_a += plus_w
            elif plus_correctness_b > plus_correctness_a:
                score_b += plus_w
            else:
                score_tie += plus_w

    total_weight = score_a + score_b + score_tie
    valid_for_overall = bool(usable > 0 and total_weight > 0)
    if not valid_for_overall:
        return "tie", 0.0, 0.0, 0.0, False

    if score_a > score_b:
        return model_a_name, float(score_a), float(score_b), float(score_tie), True
    if score_b > score_a:
        return model_b_name, float(score_a), float(score_b), float(score_tie), True
    return "tie", float(score_a), float(score_b), float(score_tie), True
