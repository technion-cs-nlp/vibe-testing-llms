"""Aggregation helpers for Stage 6 analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.vibe_testing.analysis.io import SUBJECTIVE_DIMENSIONS

# Module logger
logger = logging.getLogger(__name__)

# Deduplicate to avoid duplicates from backward compatibility mappings
# (e.g., both efficiency_score and workflow_fit_score map to subj_workflow_fit)
VIBE_DIMENSION_COLUMNS = list(dict.fromkeys(SUBJECTIVE_DIMENSIONS.values()))
OBJECTIVE_PREFIX = "obj_"

# Columns that define the configuration of a result
CONFIGURATION_DIMENSIONS = [
    "generator_model",
    "filter_model",
    "judge_model",
    "dataset_type",
]


class AnalysisDataError(RuntimeError):
    """Raised when required Stage 6 inputs are missing or invalid."""


@dataclass
class AggregationBundle:
    """
    Container that holds every derived table for Stage 6 outputs.
    """

    sample_level: pd.DataFrame
    user_model_variant: pd.DataFrame
    user_model_deltas: pd.DataFrame
    persona_summary: pd.DataFrame
    global_summary: pd.DataFrame
    ranking_reversals: pd.DataFrame


def run_full_aggregation(
    objective_df: pd.DataFrame,
    subjective_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> AggregationBundle:
    """
    Execute the full aggregation flow and return all relevant tables.

    Args:
        objective_df (pd.DataFrame): Sample-level static metrics.
        subjective_df (pd.DataFrame): Sample-level vibe metrics.
        profiles_df (pd.DataFrame): Persona metadata per ``user_id``.

    Returns:
        AggregationBundle: Structured outputs for downstream CSV + plots.
    """
    # Normalize user_id typing across all inputs before any merges/groupbys.
    # This prevents crashes like:
    # "ValueError: merge on float64 and object columns for key 'user_id'"
    objective_df = _normalize_user_id_column(objective_df, column="user_id")
    subjective_df = _normalize_user_id_column(subjective_df, column="user_id")
    profiles_df = _normalize_user_id_column(profiles_df, column="user_id")

    sample_df = prepare_sample_level_frame(
        objective_df=objective_df,
        subjective_df=subjective_df,
        profiles_df=profiles_df,
    )
    variant_summary = build_user_model_variant_summary(sample_df)
    delta_summary = compute_user_model_deltas(variant_summary)
    persona_summary = build_persona_summary(variant_summary, profiles_df)
    global_summary = build_global_summary(variant_summary)
    ranking_reversals = detect_ranking_reversals(variant_summary)

    return AggregationBundle(
        sample_level=sample_df,
        user_model_variant=variant_summary,
        user_model_deltas=delta_summary,
        persona_summary=persona_summary,
        global_summary=global_summary,
        ranking_reversals=ranking_reversals,
    )


def prepare_sample_level_frame(
    objective_df: pd.DataFrame,
    subjective_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align objective/static and subjective/vibe metrics at the sample level.

    Args:
        objective_df (pd.DataFrame): Objective metrics per sample.
        subjective_df (pd.DataFrame): Subjective metrics per sample.
        profiles_df (pd.DataFrame): User metadata for fallback inference.

    Returns:
        pd.DataFrame: Unified sample-level table with coverage indicators.

    Raises:
        AnalysisDataError: If both inputs are empty.
    """
    if (objective_df is None or objective_df.empty) and (
        subjective_df is None or subjective_df.empty
    ):
        raise AnalysisDataError("Objective and subjective inputs are both empty.")

    obj = objective_df.copy() if objective_df is not None else pd.DataFrame()
    subj = subjective_df.copy() if subjective_df is not None else pd.DataFrame()

    if obj.empty and subj.empty:
        raise AnalysisDataError("No data available after copying source frames.")

    for frame in (obj, subj):
        if not frame.empty:
            if "task_id" not in frame.columns:
                raise AnalysisDataError("task_id column is required for aggregation.")
            if "variant_id" not in frame.columns:
                frame["variant_id"] = frame["task_id"]
            if "variant_label" not in frame.columns:
                frame["variant_label"] = "original"
            else:
                frame["variant_label"] = frame["variant_label"].fillna("original")

    for frame in (obj, subj):
        if not frame.empty and "model_name" in frame.columns:
            frame["model_name"] = frame["model_name"].fillna("unknown_model")

    obj = _attach_user_ids(obj, subj, profiles_df)
    subj = _ensure_user_ids(subj, profiles_df)

    merge_keys = ["user_id", "task_id", "variant_id", "variant_label", "model_name"]
    # Do not force configuration dimensions into the merge keys.
    # Instead, we will coalesce them after the merge to handle partial metadata overlaps.

    if obj.empty:
        sample_df = subj.copy()
    elif subj.empty:
        sample_df = obj.copy()
    else:
        # Determine keys that actually exist in both frames for the merge
        join_keys = [
            key for key in merge_keys if key in obj.columns and key in subj.columns
        ]
        sample_df = pd.merge(
            obj,
            subj,
            how="outer",
            on=join_keys,
            suffixes=("_obj", ""),
        )

    # 1. Recover missing merge keys from either side
    for key in merge_keys:
        if key not in sample_df.columns:
            if key in obj.columns:
                sample_df[key] = obj[key]
            elif key in subj.columns:
                sample_df[key] = subj[key]

    # 2. Coalesce configuration dimensions (e.g. judge_model, dataset_type)
    # If a dimension exists in both inputs but wasn't used as a merge key (because we excluded it),
    # it will appear as {dim}_obj and {dim} (or just {dim} if only in one).
    # We want a single column {dim} that takes the non-null value.
    for dim in CONFIGURATION_DIMENSIONS:
        dim_obj = f"{dim}_obj"
        dim_subj = dim

        # Case A: Both columns exist (conflict resolution)
        if dim_obj in sample_df.columns and dim_subj in sample_df.columns:
            # Prefer subjective value if present, else objective (or vice versa, but subjective often has the judge metadata)
            sample_df[dim] = sample_df[dim_subj].fillna(sample_df[dim_obj])
            # Drop the _obj variant if we created a clean combined column
            if dim != dim_obj:
                sample_df.drop(columns=[dim_obj], inplace=True)

        # Case B: Only objective side exists
        elif dim_obj in sample_df.columns:
            sample_df[dim] = sample_df[dim_obj]
            sample_df.drop(columns=[dim_obj], inplace=True)

        # Case C: Only subjective side exists (already named {dim})
        elif dim in sample_df.columns:
            # Just ensure we fill NaNs if we have other sources? No, nothing to merge with.
            pass

        # Case D: Not in the main frame but might have been in source frames (e.g. if outer merge dropped it? No, merge keeps cols)
        # If the dimension was totally missing from the merge result, try to recover it from source frames based on index?
        # No, standard merge preserves columns.

        # Final cleanup: fill any remaining NaNs in config dimensions with "unknown" to avoid grouping issues later
        if dim in sample_df.columns:
            pass
            # Optional: sample_df[dim] = sample_df[dim].fillna("unknown")
            # But let's leave NaNs if they are genuinely missing to avoid noise,
            # or rely on groupby(dropna=False) which we added earlier.

    objective_cols = [
        col for col in sample_df.columns if col.startswith(OBJECTIVE_PREFIX)
    ]
    subjective_cols = [col for col in sample_df.columns if col.startswith("subj_")]
    if "combined_score" in sample_df.columns:
        subjective_cols.append("combined_score")
    sample_df["has_objective"] = (
        sample_df[objective_cols].notna().any(axis=1) if objective_cols else False
    )
    sample_df["has_subjective"] = (
        sample_df[subjective_cols].notna().any(axis=1) if subjective_cols else False
    )
    sample_df["has_both"] = sample_df["has_objective"] & sample_df["has_subjective"]
    sample_df["variant_label"] = sample_df["variant_label"].fillna("original")
    return sample_df


def build_user_model_variant_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per user/model/variant across tasks.

    Args:
        sample_df (pd.DataFrame): Unified sample-level table.

    Returns:
        pd.DataFrame: Aggregated metrics with counts, means, and standard
            deviations per ``(user_id, model_name, variant_label)``.
    """
    if sample_df.empty:
        return pd.DataFrame()

    group_cols = ["user_id", "model_name", "variant_label"]
    # dynamically add existing dimension columns to the grouping
    for dim in CONFIGURATION_DIMENSIONS:
        if dim in sample_df.columns:
            group_cols.append(dim)

    objective_cols = [
        col for col in sample_df.columns if col.startswith(OBJECTIVE_PREFIX)
    ]
    vibe_cols = [col for col in VIBE_DIMENSION_COLUMNS if col in sample_df.columns]

    summaries: List[Dict[str, float]] = []
    for keys, group in sample_df.groupby(group_cols, dropna=False):
        # keys matches the order of group_cols
        record = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))

        metrics: Dict[str, Any] = {
            **record,
            "sample_count": float(len(group)),
            "objective_count": float(group["has_objective"].sum()),
            "subjective_count": float(group["has_subjective"].sum()),
            "both_score_count": float(group["has_both"].sum()),
        }
        metrics["coverage_ratio"] = (
            metrics["both_score_count"] / metrics["sample_count"]
            if metrics["sample_count"]
            else 0.0
        )

        for col in objective_cols:
            metrics[f"{col}_mean"] = group[col].mean()

        if "subj_overall" in group.columns:
            metrics["subj_overall_mean"] = group["subj_overall"].mean()
            metrics["subj_overall_std"] = group["subj_overall"].std(ddof=0)
        if "combined_score" in group.columns:
            metrics["combined_score_mean"] = group["combined_score"].mean()
            metrics["combined_score_std"] = group["combined_score"].std(ddof=0)

        for col in vibe_cols:
            metrics[f"{col}_mean"] = group[col].mean()
            metrics[f"{col}_std"] = group[col].std(ddof=0)

        summaries.append(metrics)

    return pd.DataFrame(summaries)


def compute_user_model_deltas(
    variant_summary: pd.DataFrame,
    original_label: str = "original",
    personalized_label: str = "personalized",
) -> pd.DataFrame:
    """
    Compute personalization deltas per user/model pair.

    Args:
        variant_summary (pd.DataFrame): Output from
            :func:`build_user_model_variant_summary`.
        original_label (str): Label used for baseline/original prompts.
        personalized_label (str): Label used for personalized prompts.

    Returns:
        pd.DataFrame: Rows keyed by ``(user_id, model_name)`` containing
            personalized-minus-original deltas plus improvement indicators.
    """
    if variant_summary.empty:
        return pd.DataFrame()

    # Identify grouping columns (base keys + any config dimensions present)
    base_cols = ["user_id", "model_name"]
    extra_dims = [
        col for col in CONFIGURATION_DIMENSIONS if col in variant_summary.columns
    ]
    index_cols = base_cols + extra_dims

    personalized = (
        variant_summary[variant_summary["variant_label"] == personalized_label]
        .set_index(index_cols)
        .add_suffix(f"_{personalized_label}")
    )
    original = (
        variant_summary[variant_summary["variant_label"] == original_label]
        .set_index(index_cols)
        .add_suffix(f"_{original_label}")
    )

    merged = personalized.join(original, how="inner")
    metric_cols = [
        col
        for col in variant_summary.columns
        if col.endswith("_mean") and col not in ("coverage_ratio",)
    ]

    for base_col in metric_cols:
        personalized_col = f"{base_col}_{personalized_label}"
        original_col = f"{base_col}_{original_label}"
        if personalized_col in merged.columns and original_col in merged.columns:
            merged[f"{base_col}_delta"] = (
                merged[personalized_col] - merged[original_col]
            )

    objective_metric = _preferred_objective_metric(metric_cols)
    if objective_metric:
        delta_col = f"{objective_metric}_delta"
    else:
        delta_col = None

    subjective_delta_col = "subj_overall_mean_delta"
    subjective_delta = (
        merged[subjective_delta_col].fillna(0)
        if subjective_delta_col in merged.columns
        else pd.Series(0.0, index=merged.index)
    )
    merged["personalization_improves_subjective"] = subjective_delta > 0
    if delta_col:
        objective_delta = (
            merged[delta_col].fillna(0)
            if delta_col in merged.columns
            else pd.Series(0.0, index=merged.index)
        )
        merged["subjective_boost_no_objective_loss"] = (subjective_delta > 0) & (
            objective_delta >= 0
        )

    merged.reset_index(inplace=True)
    return merged


def build_persona_summary(
    variant_summary: pd.DataFrame, profiles_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate metrics per user_id, model, and variant.

    Args:
        variant_summary (pd.DataFrame): User/model/variant metrics.
        profiles_df (pd.DataFrame): Persona metadata (user_profile_type kept for reference).

    Returns:
        pd.DataFrame: User-level summary suitable for persona plots.
    """
    if variant_summary.empty:
        return pd.DataFrame()

    # Normalize user_id typing to avoid dtype mismatch merges (float vs object).
    vs = _normalize_user_id_column(variant_summary, column="user_id")
    profiles_df = _normalize_user_id_column(profiles_df, column="user_id")

    metric_cols = [
        col
        for col in vs.columns
        if col.endswith("_mean") and col not in ("coverage_ratio",)
    ]

    # Group by user_id + model + variant + any extra config dimensions
    # Note: user_profile_type is NOT used for grouping, only user_id
    group_cols = ["user_id", "model_name", "variant_label"]
    for dim in CONFIGURATION_DIMENSIONS:
        if dim in vs.columns:
            group_cols.append(dim)

    grouped = (
        vs.groupby(group_cols, dropna=False)
        .agg(
            {
                **{col: "mean" for col in metric_cols},
                "sample_count": "mean",
                "coverage_ratio": "mean",
            }
        )
        .reset_index()
    )

    # Optionally attach user_profile_type for reference (not for grouping)
    if "user_profile_type" in profiles_df.columns:
        persona_lookup = profiles_df[["user_id", "user_profile_type"]].drop_duplicates()
        grouped = grouped.merge(persona_lookup, on="user_id", how="left")
        grouped["user_profile_type"] = grouped["user_profile_type"].fillna("unknown")

    return grouped


def build_global_summary(variant_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics across all users for each model + variant.

    Args:
        variant_summary (pd.DataFrame): User/model/variant metrics.

    Returns:
        pd.DataFrame: Global overview with mean statistics.
    """
    if variant_summary.empty:
        return pd.DataFrame()

    metric_cols = [
        col
        for col in variant_summary.columns
        if col.endswith("_mean") and col not in ("coverage_ratio",)
    ]

    # Group by model + variant + any extra config dimensions
    group_cols = ["model_name", "variant_label"]
    for dim in CONFIGURATION_DIMENSIONS:
        if dim in variant_summary.columns:
            group_cols.append(dim)

    grouped = (
        variant_summary.groupby(group_cols, dropna=False)
        .agg(
            {
                **{col: "mean" for col in metric_cols},
                "sample_count": "sum",
                "coverage_ratio": "mean",
            }
        )
        .reset_index()
    )
    return grouped


def detect_ranking_reversals(
    variant_summary: pd.DataFrame,
    objective_metric: Optional[str] = None,
    subjective_metric: str = "subj_overall_mean",
    combined_metric: str = "combined_score_mean",
) -> pd.DataFrame:
    """
    Identify cases where the model ranking flips between static and vibe scores.

    Args:
        variant_summary (pd.DataFrame): Aggregated metrics.
        objective_metric (Optional[str]): Objective column to rank on. Defaults
            to Pass@1 mean if available.
        subjective_metric (str): Column representing subjective ranking.
        combined_metric (str): Column representing combined ranking.

    Returns:
        pd.DataFrame: One row per ``(user_id, variant_label)`` with best-model
            names and a ``ranking_reversal`` boolean.
    """
    if variant_summary.empty:
        return pd.DataFrame()

    objective_metric = objective_metric or _preferred_objective_metric(
        variant_summary.columns
    )

    # Group by user + variant + any extra config dimensions (except model_name)
    # We want to find the best model WITHIN a configuration for a user/variant
    group_cols = ["user_id", "variant_label"]
    for dim in CONFIGURATION_DIMENSIONS:
        if dim in variant_summary.columns:
            group_cols.append(dim)

    records: List[Dict[str, Any]] = []
    for keys, group in variant_summary.groupby(group_cols, dropna=False):
        # Reconstruct context
        context = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))

        record = context.copy()
        record["objective_best_model"] = _top_model(group, objective_metric)
        record["subjective_best_model"] = _top_model(group, subjective_metric)
        record["combined_best_model"] = _top_model(group, combined_metric)
        record["ranking_reversal"] = (
            record["objective_best_model"] != record["subjective_best_model"]
            if record["objective_best_model"] and record["subjective_best_model"]
            else False
        )
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attach_user_ids(
    objective_df: pd.DataFrame, subjective_df: pd.DataFrame, profiles_df: pd.DataFrame
) -> pd.DataFrame:
    """Ensure the objective frame has ``user_id`` values."""
    if objective_df.empty:
        return objective_df

    if "user_id" in objective_df.columns and objective_df["user_id"].notna().any():
        return objective_df

    user_lookup = (
        subjective_df[["task_id", "variant_id", "user_id"]].dropna().drop_duplicates()
        if not subjective_df.empty
        else pd.DataFrame(columns=["task_id", "variant_id", "user_id"])
    )
    merged = objective_df.merge(
        user_lookup,
        on=["task_id", "variant_id"],
        how="left",
        suffixes=("", "_from_subj"),
    )
    if "user_id" not in merged or merged["user_id"].isna().all():
        if not profiles_df.empty and profiles_df["user_id"].nunique() == 1:
            merged["user_id"] = profiles_df["user_id"].iloc[0]
    return merged


def _ensure_user_ids(
    subjective_df: pd.DataFrame, profiles_df: pd.DataFrame
) -> pd.DataFrame:
    """Fill missing ``user_id`` values in the subjective frame."""
    if subjective_df.empty:
        return subjective_df
    subj = subjective_df.copy()
    if "user_id" not in subj.columns:
        subj["user_id"] = None
    if subj["user_id"].isna().any() and not profiles_df.empty:
        if profiles_df["user_id"].nunique() == 1:
            subj["user_id"] = subj["user_id"].fillna(profiles_df["user_id"].iloc[0])
    return subj


def _preferred_objective_metric(columns: Sequence[str]) -> Optional[str]:
    """Choose the best available objective metric column."""
    priority = [
        "obj_overall_pass_at_1_mean",
        "obj_plus_pass_at_1_mean",
        "obj_base_pass_at_1_mean",
    ]
    for candidate in priority:
        if candidate in columns:
            return candidate
    for column in columns:
        if column.startswith(OBJECTIVE_PREFIX):
            return column
    return None


def _top_model(group: pd.DataFrame, metric: Optional[str]) -> Optional[str]:
    """Return the model with the highest value for ``metric``."""
    if not metric or metric not in group.columns:
        return None
    series = group[["model_name", metric]].dropna()
    if series.empty:
        return None
    top_row = series.sort_values(metric, ascending=False).head(1)
    return top_row["model_name"].iloc[0]


def _normalize_user_id_column(
    df: pd.DataFrame, column: str = "user_id"
) -> pd.DataFrame:
    """
    Normalize user_id typing to a canonical string identifier.

    This prevents merge/groupby failures when some inputs have numeric user_id
    (often float due to NaNs) and others have object/string user_id.

    Args:
        df (pd.DataFrame): Input frame.
        column (str): Column name to normalize.

    Returns:
        pd.DataFrame: Copy of df with a normalized `column` when present.
    """
    if df is None or df.empty or column not in df.columns:
        return df

    out = df.copy()
    before_dtype = out[column].dtype
    out[column] = out[column].map(_normalize_user_id_value).astype("string")
    after_dtype = out[column].dtype

    if before_dtype != after_dtype:
        logger.debug("Normalized %s dtype: %s -> %s", column, before_dtype, after_dtype)
    return out


def _normalize_user_id_value(value: Any) -> Any:
    """
    Normalize a single user_id value into a stable string key.

    Rules:
    - NaN/None -> <NA>
    - 3.0 -> "3" (integer-like floats)
    - 3.5 -> "3.5"
    - other -> stripped string form
    """
    if value is None or pd.isna(value):
        return pd.NA

    # Handle pandas/numpy scalars
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if text.lower() in {"nan", "none", ""}:
        return pd.NA

    # If someone already serialized a float-like integer ("3.0"), normalize to "3"
    try:
        parsed = float(text)
        if parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text
