"""CSV export utilities for Stage 6 analysis."""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS
from src.vibe_testing.analysis.judge_agreement import (
    JudgeAgreementError,
    compute_pooled_judge_agreement_metrics,
)
from src.vibe_testing.model_names import canonicalize_model_name, display_model_name

LOGGER = logging.getLogger(__name__)
# Primary identifier is user_id; user_profile_type is kept for reference only
ID_COLUMNS = ["user_id", "model_name", "variant_label"]
COUNT_COLUMNS = [
    "sample_count",
    "objective_count",
    "subjective_count",
    "both_score_count",
    "coverage_ratio",
]


def write_user_model_variant_summary(
    variant_df: pd.DataFrame,
    output_path: str,
    profiles_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Persist the per-user/model/variant aggregation table.

    Args:
        variant_df (pd.DataFrame): Output from
            :func:`build_user_model_variant_summary`.
        output_path (str): Destination CSV path.
        profiles_df (Optional[pd.DataFrame]): Persona metadata for enriching
            the table with ``user_profile_type``.

    Returns:
        Path: Path to the written CSV file.
    """
    enriched = _attach_persona_labels(variant_df, profiles_df)
    ordered_cols = _preferred_column_order(enriched.columns)
    _write_csv(enriched[ordered_cols], output_path)
    LOGGER.info("Wrote user/model variant summary to %s", output_path)
    return Path(output_path)


def write_user_model_deltas(
    delta_df: pd.DataFrame,
    output_path: str,
    profiles_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Persist personalized-minus-original deltas.

    Args:
        delta_df (pd.DataFrame): Output from :func:`compute_user_model_deltas`.
        output_path (str): Destination CSV path.
        profiles_df (Optional[pd.DataFrame]): Persona metadata for context.

    Returns:
        Path: Path to the written CSV file.
    """
    enriched = _attach_persona_labels(delta_df, profiles_df)
    ordered_cols = _preferred_column_order(enriched.columns, include_variants=False)
    _write_csv(enriched[ordered_cols], output_path)
    LOGGER.info("Wrote user/model delta summary to %s", output_path)
    return Path(output_path)


def write_model_overall_summary(
    global_df: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist the global per-model summary across all users.

    Args:
        global_df (pd.DataFrame): Output from :func:`build_global_summary`.
        output_path (str): Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    ordered_cols = _preferred_column_order(global_df.columns, include_user_cols=False)
    _write_csv(global_df[ordered_cols], output_path)
    LOGGER.info("Wrote model overall summary to %s", output_path)
    return Path(output_path)


def write_sample_level_flat(sample_df: pd.DataFrame, output_path: str) -> Path:
    """
    Persist the aligned sample-level table for ad-hoc analysis.

    Args:
        sample_df (pd.DataFrame): Output from :func:`prepare_sample_level_frame`.
        output_path (str): Destination CSV path.

    Returns:
        Path: Destination path.
    """
    sort_cols = [
        col
        for col in ["user_id", "model_name", "variant_label", "task_id", "variant_id"]
        if col in sample_df.columns
    ]
    frozen = sample_df.sort_values(sort_cols) if sort_cols else sample_df.copy()
    _write_csv(frozen, output_path)
    LOGGER.info("Wrote sample-level flat table to %s", output_path)
    return Path(output_path)


def _attach_persona_labels(
    df: pd.DataFrame, profiles_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Merge persona labels when available."""
    if profiles_df is None or profiles_df.empty or "user_id" not in df.columns:
        df = df.copy()
        if "user_profile_type" not in df.columns:
            df["user_profile_type"] = None
        return df

    # Normalize user_id typing to avoid pandas merge dtype errors (float vs object).
    left = df.copy()
    left["user_id"] = left["user_id"].map(_normalize_user_id_value).astype("string")
    persona_lookup = (
        profiles_df[["user_id", "user_profile_type"]].drop_duplicates().copy()
    )
    persona_lookup["user_id"] = (
        persona_lookup["user_id"].map(_normalize_user_id_value).astype("string")
    )

    enriched = left.merge(persona_lookup, on="user_id", how="left")
    if "user_profile_type" not in enriched.columns:
        enriched["user_profile_type"] = None
    enriched["user_profile_type"] = enriched["user_profile_type"].fillna(
        "unknown_persona"
    )
    return enriched


def _normalize_user_id_value(value: Any) -> Any:
    """
    Normalize a single user_id value into a stable string key.

    This mirrors the Stage 6 aggregation normalization so exports are robust even
    when upstream frames have numeric user_id due to dtype inference.
    """
    if value is None or pd.isna(value):
        return pd.NA

    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if text.lower() in {"nan", "none", ""}:
        return pd.NA

    try:
        parsed = float(text)
        if parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _preferred_column_order(
    columns: Iterable[str],
    include_variants: bool = True,
    include_user_cols: bool = True,
) -> List[str]:
    """Produce a consistent ordering for CSV columns."""
    existing = list(columns)
    ordered: List[str] = []

    # Primary identifier columns (user_id first, then optional user_profile_type for reference)
    if include_user_cols:
        ordered.extend([col for col in ["user_id"] if col in existing])
    ordered.extend([col for col in ["model_name"] if col in existing])
    if include_variants:
        ordered.extend([col for col in ["variant_label"] if col in existing])

    ordered.extend([col for col in COUNT_COLUMNS if col in existing])
    metric_means = sorted(
        [col for col in existing if col.endswith("_mean") and col not in COUNT_COLUMNS]
    )
    metric_stds = sorted([col for col in existing if col.endswith("_std")])
    delta_cols = sorted([col for col in existing if col.endswith("_delta")])

    remainder = [
        col
        for col in existing
        if col not in ordered + metric_means + metric_stds + delta_cols
    ]
    return ordered + metric_means + metric_stds + delta_cols + remainder


def _write_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write a DataFrame to CSV, ensuring the directory exists."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


# ---------------------------------------------------------------------------
# Pairwise comparison exporters
# ---------------------------------------------------------------------------


PAIRWISE_ID_COLUMNS = [
    "model_a_name",
    "model_b_name",
    "model_pair",
    "judge_model_name",
]


def write_pairwise_sample_level(
    sample_df: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist the sample-level pairwise comparison table.

    Args:
        sample_df: Sample-level pairwise data from PairwiseAggregationBundle.
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if sample_df.empty:
        LOGGER.warning("Pairwise sample-level data is empty, skipping write.")
        return Path(output_path)

    sort_cols = [
        col
        for col in [
            "user_id",
            "model_pair",
            "variant_label",
            "task_id",
            "variant_id",
            "raw_task_id",
        ]
        if col in sample_df.columns
    ]
    sorted_df = sample_df.sort_values(sort_cols) if sort_cols else sample_df.copy()
    _write_csv(sorted_df, output_path)
    LOGGER.info("Wrote pairwise sample-level table to %s", output_path)
    return Path(output_path)


def write_pairwise_pair_summary(
    pair_summary: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist the per-pair aggregation summary.

    Args:
        pair_summary: Output from compute_pair_summary().
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if pair_summary.empty:
        LOGGER.warning("Pairwise pair summary is empty, skipping write.")
        return Path(output_path)

    ordered_cols = _pairwise_column_order(pair_summary.columns)
    _write_csv(pair_summary[ordered_cols], output_path)
    LOGGER.info("Wrote pairwise pair summary to %s", output_path)
    return Path(output_path)


def write_pairwise_dimension_summary(
    dimension_summary: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist the per-dimension aggregation summary.

    Args:
        dimension_summary: Output from compute_dimension_win_rates().
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if dimension_summary.empty:
        LOGGER.warning("Pairwise dimension summary is empty, skipping write.")
        return Path(output_path)

    sort_cols = [
        col for col in ["model_pair", "dimension"] if col in dimension_summary.columns
    ]
    sorted_df = (
        dimension_summary.sort_values(sort_cols)
        if sort_cols
        else dimension_summary.copy()
    )
    _write_csv(sorted_df, output_path)
    LOGGER.info("Wrote pairwise dimension summary to %s", output_path)
    return Path(output_path)


def write_pairwise_user_summary(
    user_pair_summary: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist the per-user per-pair summary.

    Args:
        user_pair_summary: Output from compute_user_pair_summary().
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if user_pair_summary.empty:
        LOGGER.warning("Pairwise user summary is empty, skipping write.")
        return Path(output_path)

    sort_cols = [
        col for col in ["user_id", "model_pair"] if col in user_pair_summary.columns
    ]
    sorted_df = (
        user_pair_summary.sort_values(sort_cols)
        if sort_cols
        else user_pair_summary.copy()
    )
    _write_csv(sorted_df, output_path)
    LOGGER.info("Wrote pairwise user summary to %s", output_path)
    return Path(output_path)


def write_pairwise_preference_matrix(
    preference_matrix: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist the N x N preference matrix.

    Args:
        preference_matrix: Output from build_preference_matrix().
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if preference_matrix.empty:
        LOGGER.warning("Preference matrix is empty, skipping write.")
        return Path(output_path)

    # Include index (model names) as a column
    matrix_with_index = preference_matrix.copy()
    matrix_with_index.index.name = "model"
    matrix_with_index = matrix_with_index.reset_index()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix_with_index.to_csv(destination, index=False)
    LOGGER.info("Wrote preference matrix to %s", output_path)
    return Path(output_path)


def write_pairwise_statistical_tests(
    statistical_tests: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist the statistical significance test results.

    Args:
        statistical_tests: Output from compute_statistical_significance().
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if statistical_tests.empty:
        LOGGER.warning("Statistical tests results are empty, skipping write.")
        return Path(output_path)

    _write_csv(statistical_tests, output_path)
    LOGGER.info("Wrote pairwise statistical tests to %s", output_path)
    return Path(output_path)


def write_joint_preference_long_table(
    joint_long_df: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist a long-form joint preference table.

    This CSV is intended to contain enough information to recreate any joint
    preference matrix figure/table (wins, totals, win_rate, p_value, etc.).

    Args:
        joint_long_df: Long-form output from joint preference aggregation.
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if joint_long_df is None or joint_long_df.empty:
        LOGGER.warning("Joint preference long table is empty, skipping write.")
        return Path(output_path)

    ordered_cols = [
        col
        for col in [
            "persona",
            "prompt_type",
            "judge_model_name",
            "judge_model",
            "row_model",
            "col_model",
            "wins",
            "losses",
            "ties",
            "total",
            "tie_rate",
            "n_excl_ties",
            "win_rate",
            "p_value",
            "significant",
            "formatted_win_rate",
        ]
        if col in joint_long_df.columns
    ]
    frozen = joint_long_df.copy()
    if ordered_cols:
        remainder = [c for c in frozen.columns if c not in ordered_cols]

        # Judge agreement columns: keep overall agreement columns left of per-dimension.
        # This preserves readability for `joint_preference_long.csv` without affecting
        # any downstream merges (column names remain unchanged).
        judge_meta_cols = [
            c
            for c in ["judge_agreement_n_judges", "judge_agreement_n_items_total"]
            if c in remainder
        ]
        judge_overall_cols = sorted(
            [c for c in remainder if c.startswith("judge_agreement_overall_")]
        )
        judge_dim_cols = sorted(
            [c for c in remainder if c.startswith("judge_agreement_dim_")]
        )
        other_cols = sorted(
            [
                c
                for c in remainder
                if c not in set(judge_meta_cols + judge_overall_cols + judge_dim_cols)
            ]
        )

        frozen = frozen[
            ordered_cols
            + judge_meta_cols
            + judge_overall_cols
            + judge_dim_cols
            + other_cols
        ]
    _write_csv(frozen, output_path)
    LOGGER.info("Wrote joint preference long table to %s", output_path)
    return Path(output_path)


def write_joint_preference_rubric_detail_summary(
    rubric_summary_df: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist rubric-level mean/std summary aligned to joint preference conditions.

    This export is intended to help diagnose which rubric sub-features (within each
    vibe dimension) are associated with preference outcomes. The input DataFrame is
    expected to be produced by
    :func:`src.vibe_testing.analysis.pairwise.compute_pairwise_rubric_detail_summary`.

    Args:
        rubric_summary_df (pd.DataFrame): Long-form rubric summary table with columns
            including persona, prompt_type, row_model, col_model, vibe_dimension,
            rubric_name, mean_value, std_value, and n.
        output_path (str): Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if rubric_summary_df is None or rubric_summary_df.empty:
        LOGGER.warning("Rubric detail summary is empty, skipping write.")
        return Path(output_path)

    ordered_cols = [
        col
        for col in [
            "persona",
            "prompt_type",
            "judge_model_name",
            "row_model",
            "col_model",
            "vibe_dimension",
            "rubric_name",
            "row_mean",
            "row_std",
            "col_mean",
            "col_std",
            "n",
        ]
        if col in rubric_summary_df.columns
    ]
    frozen = rubric_summary_df.copy()
    if ordered_cols:
        remainder = [c for c in frozen.columns if c not in ordered_cols]
        frozen = frozen[ordered_cols + sorted(remainder)]
    _write_csv(frozen, output_path)
    LOGGER.info("Wrote joint preference rubric detail summary to %s", output_path)
    return Path(output_path)


def write_joint_preference_matrix(
    matrix: pd.DataFrame,
    output_path: str,
) -> Path:
    """
    Persist a joint preference matrix as a CSV.

    Args:
        matrix: Square matrix with models as index and columns.
        output_path: Destination CSV path.

    Returns:
        Path: Path to the written CSV file.
    """
    if matrix is None or matrix.empty:
        LOGGER.warning("Joint preference matrix is empty, skipping write.")
        return Path(output_path)

    matrix_with_index = matrix.copy()
    matrix_with_index.index.name = "model"
    matrix_with_index = matrix_with_index.reset_index()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix_with_index.to_csv(destination, index=False)
    LOGGER.info("Wrote joint preference matrix to %s", output_path)
    return Path(output_path)


def write_joint_preference_overall_latex(
    matrices: dict,
    output_path: str,
    personas: List[str],
    prompt_types: List[str],
    caption: str = "Pairwise win-rate preference matrices (joint across models).",
    label: str = "tab:pairwise-joint-preference",
) -> Path:
    """
    Write a paper-style LaTeX table showing win-rate and tie-rate per model pair.

    For each model pair, the table includes one row per prompt type and one column
    per persona. Each cell is formatted as:

        win_rate[*] (tie_rate)

    where ``*`` is a superscript indicating statistical significance.

    Args:
        matrices: Dict keyed by (persona, prompt_type) with values exposing
            a 'formatted_matrix' attribute (JointPreferenceMatrix).
        output_path: Destination .tex path.
        personas: Persona ordering (rows).
        prompt_types: Prompt type ordering (columns).
        caption: LaTeX caption string.
        label: LaTeX label for references.

    Returns:
        Path: Written .tex file.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _escape(text: str) -> str:
        # Minimal escaping for common model/persona identifiers
        return (
            str(text)
            .replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("^", "\\^{}")
            .replace("~", "\\~{}")
        )

    def _translate_model_name(model_name: str) -> str:
        """
        Translate a raw model token to a display-friendly name.

        This duplicates a small subset of figure translation logic to avoid
        importing plotting dependencies into exporters.
        """
        return display_model_name(model_name)

    def _translate_persona_name(persona_name: str) -> str:
        """Translate persona IDs to compact column headers."""
        if not persona_name:
            return persona_name
        raw = str(persona_name)
        normalized = raw.strip().lower().replace("-", "_")
        for token, display in [
            ("python_novice", "Beginner"),
            ("python_beginner", "Beginner"),
            ("python_intermediate", "Intermediate"),
            ("intermediate_learner", "Intermediate"),
            ("ai_researcher", "AI Researcher"),
            ("researcher_user", "AI Researcher"),
            ("advanced_developer", "Advanced"),
            ("python_advanced", "Advanced"),
        ]:
            if normalized.startswith(token) or token in normalized:
                return display
        return raw.replace("_", " ").title()

    def _format_cell(
        win_rate: Any,
        tie_rate: Any,
        significant: Any,
        opponent_win_rate: Any = None,
    ) -> str:
        """
        Format a single cell as: win_rate[*] (tie_rate).

        Bolding rule (paper-style):
        - Bold only if the cell is statistically significant AND the first model
          actually wins (its win-rate is higher than the opponent's win-rate).
        """
        if (
            win_rate is None
            or tie_rate is None
            or pd.isna(win_rate)
            or pd.isna(tie_rate)
        ):
            return "--"
        try:
            w = float(win_rate)
            t = float(tie_rate)
        except Exception:
            return "--"
        sig = bool(significant)
        win_txt = f"{w:.2f}"
        tie_txt = f"{t:.2f}"

        # Determine whether the first model wins, using the opponent direction when available.
        opp_w: float | None = None
        if opponent_win_rate is not None and not pd.isna(opponent_win_rate):
            try:
                opp_w = float(opponent_win_rate)
            except Exception:
                opp_w = None
        if opp_w is None:
            # Fallback: infer opponent win-rate from win/tie rates when possible.
            # Since total outcomes are (win + loss + tie) / total, we have:
            # opponent_win_rate = 1 - win_rate - tie_rate.
            try:
                opp_w = 1.0 - w - t
            except Exception:
                opp_w = None

        wins = (opp_w is not None) and (w > float(opp_w))
        should_bold = bool(sig and wins)
        if should_bold:
            win_txt = f"\\textbf{{{win_txt}}}"
        if sig:
            win_txt = f"{win_txt}\\textsuperscript{{*}}"
        return f"{win_txt} ({tie_txt})"

    def _has_valid_rates(win_rate: object, tie_rate: object) -> bool:
        """
        Return True iff win/tie are present and convertible to finite floats.

        This is used to avoid emitting model-pair blocks that contain only
        placeholder cells (i.e., the long-form dataframe includes the pair but
        has no actual comparison values).
        """
        if win_rate is None or tie_rate is None:
            return False
        try:
            if pd.isna(win_rate) or pd.isna(tie_rate):
                return False
        except Exception:
            # If pd.isna fails on an unexpected type, treat as invalid.
            return False
        try:
            w = float(win_rate)
            t = float(tie_rate)
        except Exception:
            return False
        if not (math.isfinite(w) and math.isfinite(t)):
            return False
        return True

    # Build a combined long-form table and a lookup.
    long_rows: List[pd.DataFrame] = []
    for (persona, pt), bundle in matrices.items():
        long_df = getattr(bundle, "pairwise_long", None)
        if long_df is None or long_df.empty:
            continue
        df = long_df.copy()
        df["persona"] = persona
        df["prompt_type"] = pt
        long_rows.append(df)

    combined_long = (
        pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    )
    if combined_long.empty:
        destination.write_text(
            "% No joint preference data available.\n", encoding="utf-8"
        )
        LOGGER.warning(
            "Joint preference LaTeX export skipped: no long-form data available."
        )
        return destination

    long_lookup: dict = {}
    for _, r in combined_long.iterrows():
        long_lookup[
            (
                r.get("persona"),
                r.get("prompt_type"),
                r.get("row_model"),
                r.get("col_model"),
            )
        ] = {
            "win_rate": r.get("win_rate"),
            "tie_rate": r.get("tie_rate"),
            "significant": r.get("significant"),
        }

    # Determine unique unordered model pairs across all slices.
    #
    # IMPORTANT: Do NOT include pairs that have no numeric values anywhere.
    # Some upstream runs may include rows for a model pair but leave win/tie
    # rates empty (NaN/None). Emitting these produces LaTeX blocks that are
    # entirely "--" placeholders, which we explicitly want to omit.
    model_pairs_set: set[tuple[str, str]] = set()
    for _, r in combined_long.iterrows():
        a = r.get("row_model")
        b = r.get("col_model")
        if not a or not b:
            continue
        a = str(a)
        b = str(b)
        if a == b:
            continue
        if not _has_valid_rates(r.get("win_rate"), r.get("tie_rate")):
            continue
        model_pairs_set.add(tuple(sorted([a, b])))

    unordered_pairs: List[tuple[str, str]] = sorted(
        model_pairs_set,
        key=lambda p: (_translate_model_name(p[0]), _translate_model_name(p[1])),
    )

    # Choose a stable direction for each unordered pair.
    #
    # Requirements:
    # - Prefer GPT-5.1 as the first model when paired with GPT-OSS-20B or GPT-4o.
    # - Prefer Qwen3-32B as the first model when paired with Qwen3-14B.
    # - Otherwise, fall back to the previous heuristic (mean win-rate >= 0.5).
    directed_pairs: List[tuple[str, str]] = []
    for a, b in unordered_pairs:
        disp_a = _translate_model_name(a)
        disp_b = _translate_model_name(b)

        preferred_order: tuple[str, str] | None = None
        if "GPT-5.1" in {disp_a, disp_b}:
            # Always put GPT-5.1 first when present.
            preferred_order = (a, b) if disp_a == "GPT-5.1" else (b, a)
        elif {disp_a, disp_b} == {"Qwen3-32B", "Qwen3-14B"}:
            preferred_order = (a, b) if disp_a == "Qwen3-32B" else (b, a)

        if preferred_order is not None:
            directed_pairs.append(preferred_order)
            continue

        ab_rates = combined_long[
            (combined_long["row_model"] == a) & (combined_long["col_model"] == b)
        ]["win_rate"].dropna()
        ba_rates = combined_long[
            (combined_long["row_model"] == b) & (combined_long["col_model"] == a)
        ]["win_rate"].dropna()
        mean_ab = float(ab_rates.mean()) if not ab_rates.empty else None
        mean_ba = float(ba_rates.mean()) if not ba_rates.empty else None
        if mean_ab is None and mean_ba is None:
            directed_pairs.append((a, b))
        elif mean_ab is None:
            directed_pairs.append((b, a))
        elif mean_ba is None:
            directed_pairs.append((a, b))
        else:
            directed_pairs.append((a, b) if mean_ab >= 0.5 else (b, a))

    # Build LaTeX output.
    lines: List[str] = []
    lines.append("% Auto-generated by stage_6_analyze_results.py")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{6pt}")
    lines.append("  \\renewcommand{\\arraystretch}{1.15}")

    # Persona columns are ordered left-to-right as:
    # Beginner, Intermediate, AI Researcher, Advanced (unknown personas at the end).
    persona_order = {
        "Beginner": 0,
        "Intermediate": 1,
        "AI Researcher": 2,
        "Advanced": 3,
    }
    personas_sorted = sorted(
        list(personas),
        key=lambda p: (
            persona_order.get(_translate_persona_name(p), 999),
            _translate_persona_name(p),
            str(p),
        ),
    )
    persona_headers = [_translate_persona_name(p) for p in personas_sorted]
    col_spec = "c l| " + " ".join(["c"] * len(personas_sorted))
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")
    lines.append("     &  &")
    lines.append(
        "    \\multicolumn{"
        + str(len(personas_sorted))
        + "}{c}{\\textbf{Win-rate (Tie-rate)}} \\\\"
    )
    lines.append("    \\cmidrule(lr){3-" + str(2 + len(personas_sorted)) + "}")
    lines.append(
        "    \\textbf{Model Pair} & \\textbf{Prompt Type} & "
        + " & ".join(f"\\textbf{{{_escape(h)}}}" for h in persona_headers)
        + " \\\\"
    )
    lines.append("    \\midrule")

    for pair_idx, (first, second) in enumerate(directed_pairs):
        first_disp = _escape(_translate_model_name(first))
        second_disp = _escape(_translate_model_name(second))
        model_pair_cell = (
            "\\begin{tabular}[c]{@{}c@{}}"
            f"\\texttt{{{first_disp}}} \\\\"
            "\\textit{vs.} \\\\"
            f"\\texttt{{{second_disp}}}"
            "\\end{tabular}"
        )

        for pt_i, pt in enumerate(prompt_types):
            prompt_label = _escape(str(pt).title())
            row_cells: List[str] = []
            for persona in personas_sorted:
                info = long_lookup.get((persona, pt, first, second))
                if not info:
                    row_cells.append("--")
                    continue
                opp = long_lookup.get((persona, pt, second, first))
                row_cells.append(
                    _format_cell(
                        info.get("win_rate"),
                        info.get("tie_rate"),
                        info.get("significant"),
                        opponent_win_rate=(opp.get("win_rate") if opp else None),
                    )
                )

            if pt_i == 0:
                lines.append(
                    f"    \\multirow{{{len(prompt_types)}}}{{*}}{{%\n"
                    f"      {model_pair_cell}%\n"
                    f"    }}\n"
                    f"      & {prompt_label}\n"
                    f"        & " + " & ".join(row_cells) + " \\\\"
                )
            else:
                lines.append(
                    f"      & {prompt_label}\n"
                    f"        & " + " & ".join(row_cells) + " \\\\"
                )

        if pair_idx < len(directed_pairs) - 1:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{")
    lines.append("    " + _escape(caption))
    lines.append("  }")
    lines.append("  \\label{" + _escape(label) + "}")
    lines.append("\\end{table*}")

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote joint preference overall LaTeX table to %s", output_path)
    return destination


_STREAMLIT_PROMPT_TYPE_LABELS = {
    "original": "Original",
    "control": "Control",
    "personalized": "Personalized",
}
_STREAMLIT_PROMPT_TYPE_ORDER = {
    "original": 0,
    "control": 1,
    "personalized": 2,
}
_STREAMLIT_JUDGE_PAIR_ROWS = [
    ("Overall", "overall"),
    ("LLM--LLM", "llm_llm"),
    ("Human--Human", "human_human"),
    ("Human--LLM", "human_llm"),
]
_STREAMLIT_DIMENSION_LABELS = {
    "clarity": "Clarity",
    "tone_style_fit": "Tone/Style Fit",
    "workflow_fit": "Workflow Fit",
    "cognitive_load": "Cognitive Load",
    "context_awareness": "Context Awareness",
    "persona_consistency": "Persona Consistency",
    "friction_loss_of_control": "Friction/Loss Of Control",
    "reliability_user_trust": "Reliability/User Trust",
    "anthropomorphism": "Anthropomorphism",
}


def _streamlit_import_agreement_helpers():
    """Import Streamlit agreement helpers lazily to avoid circular imports."""
    from src.vibe_testing.analysis.judge_utils import split_judges_by_group
    from src.vibe_testing.ui.pairwise_explorer_stats import (
        compute_dimension_judge_pair_agreement,
        compute_judge_pair_agreement,
        filter_item_ids_to_human_annotated,
        filter_agreement_df_between_judge_groups,
        filter_agreement_df_to_within_judge_group,
    )

    return {
        "compute_dimension_judge_pair_agreement": compute_dimension_judge_pair_agreement,
        "compute_judge_pair_agreement": compute_judge_pair_agreement,
        "filter_item_ids_to_human_annotated": filter_item_ids_to_human_annotated,
        "filter_agreement_df_between_judge_groups": (
            filter_agreement_df_between_judge_groups
        ),
        "filter_agreement_df_to_within_judge_group": (
            filter_agreement_df_to_within_judge_group
        ),
        "split_judges_by_group": split_judges_by_group,
    }


def _escape_latex_text(text: object) -> str:
    """Escape a value for safe LaTeX inline rendering."""
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("^", "\\^{}")
        .replace("~", "\\~{}")
    )


def _safe_str(value: object) -> str:
    """Convert a value to string while treating missing values as empty."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _streamlit_prompt_types(frame: pd.DataFrame) -> List[str]:
    """Return prompt types ordered like Streamlit/Stage 6."""
    if frame is None or frame.empty or "__prompt_type__" not in frame.columns:
        return []
    return [
        prompt_type
        for prompt_type in sorted(
            frame["__prompt_type__"].dropna().unique().tolist(),
            key=lambda value: (
                _STREAMLIT_PROMPT_TYPE_ORDER.get(str(value), 999),
                str(value),
            ),
        )
        if str(prompt_type).strip()
    ]


def _streamlit_item_id_from_row(row: pd.Series, prompt_type: str) -> str:
    """Build the same composite item id used by the Streamlit-style exporter."""
    return "||".join(
        [
            _safe_str(row.get("user_id")).strip(),
            str(prompt_type).strip().lower(),
            _safe_str(row.get("model_a_name")).strip(),
            _safe_str(row.get("model_b_name")).strip(),
            _safe_str(row.get("task_id")).strip(),
            _safe_str(row.get("variant_id")).strip(),
        ]
    )


def _winner_label_to_streamlit_outcome(
    winner: object,
    *,
    model_a_name: object,
    model_b_name: object,
) -> Optional[str]:
    """Map a Stage-6 overall or dimension winner label to model_a/model_b/tie."""
    winner_text = _safe_str(winner).strip()
    if not winner_text:
        return None
    winner_lower = winner_text.lower()
    if winner_lower == "tie":
        return "tie"

    model_a_text = _safe_str(model_a_name).strip()
    model_b_text = _safe_str(model_b_name).strip()
    if winner_lower in {"model_a", "row"} or winner_text == model_a_text:
        return "model_a"
    if winner_lower in {"model_b", "col"} or winner_text == model_b_text:
        return "model_b"
    if winner_text == display_model_name(model_a_text):
        return "model_a"
    if winner_text == display_model_name(model_b_text):
        return "model_b"
    raise ValueError(
        "Unsupported winner label for Streamlit-style agreement LaTeX: "
        f"{winner_text!r}"
    )


def _streamlit_subset_agreement_df(
    agreement_df: pd.DataFrame,
    *,
    pair_type: str,
    human_judges: List[str],
    llm_judges: List[str],
    helper_imports: Dict[str, object],
) -> pd.DataFrame:
    """Select the requested judge-pair subset using the same Streamlit rules."""
    if agreement_df is None or agreement_df.empty:
        return pd.DataFrame()
    if pair_type == "overall":
        return agreement_df.copy()
    if pair_type == "llm_llm":
        return helper_imports["filter_agreement_df_to_within_judge_group"](
            agreement_df,
            judges_in_group=llm_judges,
        )
    if pair_type == "human_human":
        return helper_imports["filter_agreement_df_to_within_judge_group"](
            agreement_df,
            judges_in_group=human_judges,
        )
    if pair_type == "human_llm":
        return helper_imports["filter_agreement_df_between_judge_groups"](
            agreement_df,
            judges_in_group_a=human_judges,
            judges_in_group_b=llm_judges,
        )
    raise ValueError(f"Unsupported judge pair type: {pair_type!r}")


def _streamlit_filter_item_ids_for_pair_type(
    item_ids: List[str],
    *,
    pair_type: str,
    outcomes_by_judge: Dict[str, Dict[str, str]],
    human_judges: List[str],
    helper_imports: Dict[str, object],
) -> List[str]:
    """
    Restrict item ids for specific pair types to the human-annotated subset.

    Per the requested behavior, `Overall` and `LLM--LLM` should only be computed
    on samples that include at least one human annotation. Other pair types keep
    the original item scope unchanged.
    """
    if pair_type not in {"overall", "llm_llm"}:
        return list(item_ids)
    return helper_imports["filter_item_ids_to_human_annotated"](
        list(item_ids),
        outcomes_by_judge=outcomes_by_judge,
        human_judges=human_judges,
    )


def _streamlit_mean_std(
    series: pd.Series,
) -> Tuple[Optional[float], Optional[float], int]:
    """Match Streamlit mean/std aggregation across judge-pair rows."""
    n = int(len(series))
    if n <= 0:
        return None, None, 0
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if n > 1 else 0.0
    return mean, std, n


def _streamlit_metric_summary(
    agreement_df: pd.DataFrame,
    *,
    value_column: str,
    item_count_column: str,
) -> Dict[str, object]:
    """Summarize one Streamlit metric family with counts and overlap totals."""
    if agreement_df is None or agreement_df.empty:
        return {
            "mean": None,
            "std": None,
            "n_pairs": 0,
            "n_items": 0,
        }
    value_series = pd.to_numeric(
        agreement_df.get(value_column),
        errors="coerce",
    ).dropna()
    mean, std, n_pairs = _streamlit_mean_std(value_series)
    n_items = int(
        pd.to_numeric(agreement_df.get(item_count_column), errors="coerce").dropna().sum()
    )
    return {
        "mean": mean,
        "std": std,
        "n_pairs": n_pairs,
        "n_items": n_items,
    }


def _format_pmstd(
    mean: Optional[float],
    std: Optional[float],
    *,
    decimals: int,
    scale: float = 1.0,
    bold_mean: bool = False,
    missing_text: str = "--",
) -> str:
    """Format a mean/std pair using the existing LaTeX macro."""
    if mean is None or std is None or pd.isna(mean) or pd.isna(std):
        return str(missing_text)
    mean_value = float(mean) * float(scale)
    std_value = float(std) * float(scale)
    mean_text = f"{mean_value:.{decimals}f}"
    if bold_mean:
        mean_text = f"\\textbf{{{mean_text}}}"
    return f"\\pmstd{{{mean_text}}}{{{std_value:.{decimals}f}}}"


def _streamlit_dimension_display_name(dimension: str) -> str:
    """Return a compact display label for a pairwise dimension key."""
    return _STREAMLIT_DIMENSION_LABELS.get(
        str(dimension),
        str(dimension).replace("_", " ").title(),
    )


def _streamlit_build_overall_outcomes(
    pairwise_df: pd.DataFrame,
    *,
    judge_column: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], Dict[str, set[str]]]:
    """Build prompt-normalized outcomes used by the overall Streamlit-style table."""
    frame = pairwise_df.copy()
    frame["__prompt_type__"] = frame["variant_label"].map(
        lambda value: _safe_str(value).strip().lower()
    )
    outcomes_by_judge: Dict[str, Dict[str, str]] = {}
    prompt_item_ids: Dict[str, set[str]] = {
        prompt_type: set() for prompt_type in _streamlit_prompt_types(frame)
    }

    for _, row in frame.iterrows():
        prompt_type = _safe_str(row.get("__prompt_type__")).strip().lower()
        if not prompt_type:
            continue
        judge = _safe_str(row.get(judge_column)).strip()
        if not judge:
            continue
        item_id = _streamlit_item_id_from_row(row, prompt_type)
        if not item_id.replace("|", ""):
            continue
        outcome = _winner_label_to_streamlit_outcome(
            row.get("overall_winner_label"),
            model_a_name=row.get("model_a_name"),
            model_b_name=row.get("model_b_name"),
        )
        if outcome is None:
            continue

        judge_map = outcomes_by_judge.setdefault(judge, {})
        existing = judge_map.get(item_id)
        if existing is not None and existing != outcome:
            raise ValueError(
                "Conflicting duplicate overall winner labels for the same item/judge "
                "when building Streamlit-style agreement LaTeX. "
                f"judge={judge!r} item_id={item_id!r} existing={existing!r} "
                f"new={outcome!r}"
            )
        judge_map[item_id] = outcome
        prompt_item_ids.setdefault(prompt_type, set()).add(item_id)

    return frame, outcomes_by_judge, prompt_item_ids


def _streamlit_build_dimension_sample_frame(
    pairwise_df: pd.DataFrame,
    *,
    judge_column: str,
    dimensions: List[str],
) -> pd.DataFrame:
    """Build a Stage-6-style sample frame for dimension agreement helpers."""
    frame = pairwise_df.copy()
    frame["__prompt_type__"] = frame["variant_label"].map(
        lambda value: _safe_str(value).strip().lower()
    )

    rows: List[Dict[str, object]] = []
    for _, row in frame.iterrows():
        prompt_type = _safe_str(row.get("__prompt_type__")).strip().lower()
        if not prompt_type:
            continue
        judge = _safe_str(row.get(judge_column)).strip()
        if not judge:
            continue

        item_id = _streamlit_item_id_from_row(row, prompt_type)
        if not item_id.replace("|", ""):
            continue

        base_row: Dict[str, object] = {
            "item_id": item_id,
            judge_column: judge,
            "__prompt_type__": prompt_type,
        }
        has_dimension = False
        for dimension in dimensions:
            winner_column = f"dim_{dimension}_winner_label"
            if winner_column not in frame.columns:
                continue
            outcome = _winner_label_to_streamlit_outcome(
                row.get(winner_column),
                model_a_name=row.get("model_a_name"),
                model_b_name=row.get("model_b_name"),
            )
            if outcome is None:
                continue
            base_row[winner_column] = outcome
            has_dimension = True
        if has_dimension:
            rows.append(base_row)

    return pd.DataFrame(rows)


def write_joint_preference_streamlit_agreement_latex(
    pairwise_df: pd.DataFrame,
    output_path: str,
    *,
    judge_column: str = "judge_model_name",
    disable_metrics: bool = False,
    caption: str = (
        "Human judgment validation results across prompt types and judge pair "
        "types. Agreement is reported as mean percentage agreement and Cohen's "
        "$\\kappa$, with standard deviations across judge pairs. The reported "
        "\\#Pairs and \\#Items follow the Streamlit agreement table semantics: "
        "\\#Pairs counts judge pairs with defined percent agreement, and "
        "\\#Items sums the per-pair overlap counts across those judge pairs."
    ),
    label: str = "tab:human_validation_agreement_streamlit",
) -> Path:
    """
    Write a Streamlit-style agreement LaTeX table for Stage 6 pairwise results.

    This export intentionally mirrors the agreement summary tables shown in the
    Streamlit pairwise explorer. It uses per-judge-pair agreement rows from
    ``compute_judge_pair_agreement(...)`` and summarizes them with mean/std
    across judge pairs, matching Streamlit's ``std(ddof=1)`` behavior and item
    counting semantics.

    Args:
        pairwise_df (pd.DataFrame): Canonical Stage-6 pairwise sample dataframe.
        output_path (str): Destination .tex path.
        judge_column (str): Column storing judge identifiers.
        disable_metrics (bool): If True, write the fixed table structure with
            placeholder values and an explanatory comment.
        caption (str): LaTeX caption.
        label (str): LaTeX label.

    Returns:
        Path: Written .tex file.

    Raises:
        ValueError: If required columns are missing or duplicate judge/item rows
            conflict.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    helper_imports = _streamlit_import_agreement_helpers()

    if pairwise_df is None:
        raise ValueError(
            "pairwise_df is None; cannot export Streamlit-style agreement LaTeX."
        )

    required = [
        "user_id",
        "variant_label",
        "task_id",
        "variant_id",
        "model_a_name",
        "model_b_name",
        "overall_winner_label",
    ]
    missing = [col for col in required if col not in pairwise_df.columns]
    if missing:
        raise ValueError(
            "pairwise_df is missing required columns for Streamlit-style "
            "agreement LaTeX: " + ", ".join(missing)
        )
    if judge_column not in pairwise_df.columns:
        raise ValueError(
            "pairwise_df is missing the judge column required for Streamlit-style "
            f"agreement LaTeX export: {judge_column}"
        )

    frame, outcomes_by_judge, prompt_item_ids = _streamlit_build_overall_outcomes(
        pairwise_df,
        judge_column=judge_column,
    )
    prompt_types = _streamlit_prompt_types(frame)
    if not prompt_types:
        raise ValueError(
            "pairwise_df has no prompt types; cannot export Streamlit-style "
            "agreement LaTeX."
        )

    judges = sorted(outcomes_by_judge.keys())
    if len(judges) < 2 and not disable_metrics:
        raise ValueError(
            "Need at least two judges to export Streamlit-style agreement LaTeX. "
            f"Found {len(judges)}."
        )

    human_judges, llm_judges = helper_imports["split_judges_by_group"](judges)

    def _compute_agreement_df(item_ids: List[str]) -> pd.DataFrame:
        if disable_metrics or len(judges) < 2 or not item_ids:
            return pd.DataFrame()
        return helper_imports["compute_judge_pair_agreement"](
            outcomes_by_judge,
            task_ids=item_ids,
            categories=("model_a", "model_b", "tie"),
        )

    prompt_summaries: List[Tuple[str, List[Tuple[str, Dict[str, object]]]]] = []
    total_usable_pair_rows = 0

    for prompt_type in prompt_types:
        item_ids = sorted(prompt_item_ids.get(prompt_type, set()))
        row_summaries: List[Tuple[str, Dict[str, object]]] = []
        for display_label, pair_type in _STREAMLIT_JUDGE_PAIR_ROWS:
            scoped_item_ids = _streamlit_filter_item_ids_for_pair_type(
                item_ids,
                pair_type=pair_type,
                outcomes_by_judge=outcomes_by_judge,
                human_judges=human_judges,
                helper_imports=helper_imports,
            )
            agreement_df = _compute_agreement_df(scoped_item_ids)
            subset_df = _streamlit_subset_agreement_df(
                agreement_df,
                pair_type=pair_type,
                human_judges=human_judges,
                llm_judges=llm_judges,
                helper_imports=helper_imports,
            )
            percent_summary = _streamlit_metric_summary(
                subset_df,
                value_column="percent_agreement",
                item_count_column="n_common",
            )
            kappa_summary = _streamlit_metric_summary(
                subset_df,
                value_column="cohens_kappa",
                item_count_column="n_common",
            )
            summary = {
                "percent_mean": percent_summary["mean"],
                "percent_std": percent_summary["std"],
                "kappa_mean": kappa_summary["mean"],
                "kappa_std": kappa_summary["std"],
                "n_pairs": percent_summary["n_pairs"],
                "n_items": percent_summary["n_items"],
            }
            if summary["n_pairs"]:
                total_usable_pair_rows += int(summary["n_pairs"])
            row_summaries.append((display_label, summary))
        prompt_summaries.append((prompt_type, row_summaries))

    overall_item_ids = sorted(
        {item_id for item_ids in prompt_item_ids.values() for item_id in item_ids}
    )
    overall_agreement_df = _compute_agreement_df(
        _streamlit_filter_item_ids_for_pair_type(
            overall_item_ids,
            pair_type="overall",
            outcomes_by_judge=outcomes_by_judge,
            human_judges=human_judges,
            helper_imports=helper_imports,
        )
    )
    overall_subset = _streamlit_subset_agreement_df(
        overall_agreement_df,
        pair_type="overall",
        human_judges=human_judges,
        llm_judges=llm_judges,
        helper_imports=helper_imports,
    )
    overall_percent_summary = _streamlit_metric_summary(
        overall_subset,
        value_column="percent_agreement",
        item_count_column="n_common",
    )
    overall_kappa_summary = _streamlit_metric_summary(
        overall_subset,
        value_column="cohens_kappa",
        item_count_column="n_common",
    )
    overall_summary = {
        "percent_mean": overall_percent_summary["mean"],
        "percent_std": overall_percent_summary["std"],
        "kappa_mean": overall_kappa_summary["mean"],
        "kappa_std": overall_kappa_summary["std"],
        "n_pairs": overall_percent_summary["n_pairs"],
        "n_items": overall_percent_summary["n_items"],
    }

    lines: List[str] = []
    lines.append("% Auto-generated by stage_6_analyze_results.py")
    if disable_metrics:
        lines.append("% Streamlit-style agreement metrics were disabled via CLI.")
    elif total_usable_pair_rows <= 0:
        lines.append(
            "% No usable judge-pair agreement rows were available; metrics are "
            "rendered as NA and counts as 0."
        )
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{7pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\begin{tabular}{@{}c|c|cc|cc@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Prompt Type} & \\textbf{Judge Pair Type} & "
        "\\textbf{Agreement (\\%)} & \\textbf{Cohen's $\\kappa$} & "
        "\\textbf{\\#Pairs} & \\textbf{\\#Items} \\\\"
    )
    lines.append("\\midrule")

    for prompt_index, (prompt_type, row_summaries) in enumerate(prompt_summaries):
        prompt_label = _STREAMLIT_PROMPT_TYPE_LABELS.get(prompt_type, prompt_type.title())
        for row_index, (pair_label, summary) in enumerate(row_summaries):
            prefix = ""
            if row_index == 0:
                prefix = (
                    f"\\multirow{{{len(row_summaries)}}}{{*}}{{\\textbf{{{_escape_latex_text(prompt_label)}}}}} "
                )
            percent_text = _format_pmstd(
                summary["percent_mean"],
                summary["percent_std"],
                decimals=1,
                scale=100.0,
                missing_text="NA",
            )
            kappa_text = _format_pmstd(
                summary["kappa_mean"],
                summary["kappa_std"],
                decimals=2,
                missing_text="NA",
            )
            n_pairs_text = str(int(summary["n_pairs"])) if int(summary["n_pairs"]) > 0 else "0"
            n_items_text = str(int(summary["n_items"])) if int(summary["n_items"]) > 0 else "0"
            if row_index == 0:
                lines.append(
                    f"{prefix}& {_escape_latex_text(pair_label)} & {percent_text} & {kappa_text} "
                    f"& {n_pairs_text} & {n_items_text} \\\\"
                )
            else:
                lines.append(
                    f"& {_escape_latex_text(pair_label)} & {percent_text} & {kappa_text} "
                    f"& {n_pairs_text} & {n_items_text} \\\\"
                )
        if prompt_index < len(prompt_summaries) - 1:
            lines.append("\\midrule")

    lines.append("\\midrule")
    lines.append("\\midrule")
    lines.append(
        "\\textbf{Overall} & All Samples "
        f"& {_format_pmstd(overall_summary['percent_mean'], overall_summary['percent_std'], decimals=1, scale=100.0, bold_mean=True, missing_text='NA')} "
        f"& {_format_pmstd(overall_summary['kappa_mean'], overall_summary['kappa_std'], decimals=2, bold_mean=True, missing_text='NA')} "
        f"& {int(overall_summary['n_pairs']) if int(overall_summary['n_pairs']) > 0 else 0} "
        f"& {int(overall_summary['n_items']) if int(overall_summary['n_items']) > 0 else 0} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{" + _escape_latex_text(caption) + "}")
    lines.append("\\label{" + _escape_latex_text(label) + "}")
    lines.append("\\end{table*}")

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote Streamlit-style agreement LaTeX table to %s", output_path)
    return destination


def write_joint_preference_streamlit_dimension_agreement_latex(
    pairwise_df: pd.DataFrame,
    output_path: str,
    *,
    judge_column: str = "judge_model_name",
    disable_metrics: bool = False,
    caption: str = (
        "Dimension-level human judgment validation results. Agreement is "
        "reported as mean percentage agreement and Cohen's $\\kappa$, with "
        "standard deviations across judge pairs."
    ),
    label: str = "tab:human_validation_dimension_agreement_streamlit",
) -> Path:
    """
    Write a Streamlit-style per-dimension agreement LaTeX table.

    This export mirrors Streamlit's dimension-level agreement calculations while
    keeping the existing overall Streamlit-style agreement table unchanged.

    Args:
        pairwise_df (pd.DataFrame): Canonical Stage-6 pairwise sample dataframe.
        output_path (str): Destination .tex path.
        judge_column (str): Column storing judge identifiers.
        disable_metrics (bool): If True, write the fixed table structure with
            placeholder values and an explanatory comment.
        caption (str): LaTeX caption.
        label (str): LaTeX label.

    Returns:
        Path: Written .tex file.

    Raises:
        ValueError: If required columns are missing or if no dimension winner
            columns exist.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    helper_imports = _streamlit_import_agreement_helpers()

    if pairwise_df is None:
        raise ValueError(
            "pairwise_df is None; cannot export Streamlit-style dimension "
            "agreement LaTeX."
        )

    required = [
        "user_id",
        "variant_label",
        "task_id",
        "variant_id",
        "model_a_name",
        "model_b_name",
    ]
    missing = [col for col in required if col not in pairwise_df.columns]
    if missing:
        raise ValueError(
            "pairwise_df is missing required columns for Streamlit-style "
            "dimension agreement LaTeX: " + ", ".join(missing)
        )
    if judge_column not in pairwise_df.columns:
        raise ValueError(
            "pairwise_df is missing the judge column required for Streamlit-style "
            f"dimension agreement LaTeX export: {judge_column}"
        )

    dimensions = [
        dimension
        for dimension in PAIRWISE_DIMENSIONS
        if f"dim_{dimension}_winner_label" in pairwise_df.columns
    ]
    if not dimensions:
        raise ValueError(
            "pairwise_df is missing per-dimension winner labels required for "
            "Streamlit-style dimension agreement LaTeX."
        )

    sample_df = _streamlit_build_dimension_sample_frame(
        pairwise_df,
        judge_column=judge_column,
        dimensions=dimensions,
    )
    prompt_types = [
        prompt_type
        for prompt_type in _streamlit_prompt_types(sample_df)
        if prompt_type in {"original", "personalized"}
    ]
    if not prompt_types:
        raise ValueError(
            "pairwise_df has no original/personalized prompt types with usable "
            "dimension winner labels; cannot export Streamlit-style dimension "
            "agreement LaTeX."
        )

    judges = (
        sorted(sample_df[judge_column].dropna().astype(str).unique().tolist())
        if judge_column in sample_df.columns
        else []
    )
    if len(judges) < 2 and not disable_metrics:
        raise ValueError(
            "Need at least two judges to export Streamlit-style dimension "
            f"agreement LaTeX. Found {len(judges)}."
        )
    human_judges, llm_judges = helper_imports["split_judges_by_group"](judges)

    prompt_dimension_summaries: List[
        Tuple[
            str,
            List[Tuple[str, List[Tuple[str, Dict[str, object]]]]],
            List[Tuple[str, Dict[str, object]]],
        ]
    ] = []
    total_usable_pair_rows = 0

    def _format_item_count_summary(
        counts: List[int],
    ) -> str:
        unique_counts = sorted({int(value) for value in counts})
        if not unique_counts:
            return "0"
        if len(unique_counts) == 1:
            return str(unique_counts[0])
        return f"{unique_counts[0]}--{unique_counts[-1]}"

    def _build_prompt_caption(
        prompt_type: str,
        dimension_rows: List[Tuple[str, List[Tuple[str, Dict[str, object]]]]],
        pooled_pair_rows: List[Tuple[str, Dict[str, object]]],
    ) -> str:
        prompt_label = _STREAMLIT_PROMPT_TYPE_LABELS.get(prompt_type, prompt_type.title())
        pair_type_summaries: List[str] = []
        range_needed = False
        for pair_label, _pair_type in _STREAMLIT_JUDGE_PAIR_ROWS:
            n_items_values: List[int] = []
            n_items_excl_values: List[int] = []
            for _dimension, pair_rows in dimension_rows:
                for candidate_label, summary in pair_rows:
                    if candidate_label != pair_label:
                        continue
                    n_items_values.append(int(summary["n_items"]))
                    n_items_excl_values.append(int(summary["n_items_excl"]))
            if len(set(n_items_values)) > 1 or len(set(n_items_excl_values)) > 1:
                range_needed = True
            pair_type_summaries.append(
                f"{pair_label}: items={_format_item_count_summary(n_items_values)}, "
                f"excl. ties={_format_item_count_summary(n_items_excl_values)}"
            )

        caption_parts = [
            f"{caption} Prompt type: {prompt_label}.",
            "Counts by judge pair type use the Streamlit overlap semantics "
            "(items from n_common; excl. ties from n_common_excl_ties).",
            " ".join(pair_type_summaries) + ".",
        ]
        if pooled_pair_rows:
            pooled_summaries = []
            for pair_label, summary in pooled_pair_rows:
                pooled_summaries.append(
                    f"{pair_label}: items={int(summary['n_items'])}, "
                    f"excl. ties={int(summary['n_items_excl'])}"
                )
            caption_parts.append(
                "Pooled counts treat each sample-dimension pair as one item. "
                "Pooled "
                + "; ".join(pooled_summaries)
                + "."
            )
        if range_needed:
            caption_parts.append(
                "Ranges indicate that the overlap count varies across dimensions."
            )
        return " ".join(caption_parts)

    def _compute_pair_type_summaries(
        outcomes_by_judge: Dict[str, Dict[str, str]],
        item_ids: List[str],
    ) -> List[Tuple[str, Dict[str, object]]]:
        nonlocal total_usable_pair_rows

        pair_rows: List[Tuple[str, Dict[str, object]]] = []
        for display_label, pair_type in _STREAMLIT_JUDGE_PAIR_ROWS:
            scoped_item_ids = _streamlit_filter_item_ids_for_pair_type(
                item_ids,
                pair_type=pair_type,
                outcomes_by_judge=outcomes_by_judge,
                human_judges=human_judges,
                helper_imports=helper_imports,
            )
            if disable_metrics or len(judges) < 2 or not scoped_item_ids:
                agreement_df = pd.DataFrame()
            else:
                agreement_df = helper_imports["compute_judge_pair_agreement"](
                    outcomes_by_judge,
                    task_ids=scoped_item_ids,
                    categories=("model_a", "model_b", "tie"),
                )
            subset_df = _streamlit_subset_agreement_df(
                agreement_df,
                pair_type=pair_type,
                human_judges=human_judges,
                llm_judges=llm_judges,
                helper_imports=helper_imports,
            )
            percent_summary = _streamlit_metric_summary(
                subset_df,
                value_column="percent_agreement",
                item_count_column="n_common",
            )
            kappa_summary = _streamlit_metric_summary(
                subset_df,
                value_column="cohens_kappa",
                item_count_column="n_common",
            )
            percent_excl_summary = _streamlit_metric_summary(
                subset_df,
                value_column="percent_agreement_excl_ties",
                item_count_column="n_common_excl_ties",
            )
            kappa_excl_summary = _streamlit_metric_summary(
                subset_df,
                value_column="cohens_kappa_excl_ties",
                item_count_column="n_common_excl_ties",
            )
            if percent_summary["n_pairs"]:
                total_usable_pair_rows += int(percent_summary["n_pairs"])
            pair_rows.append(
                (
                    display_label,
                    {
                        "percent_mean": percent_summary["mean"],
                        "percent_std": percent_summary["std"],
                        "kappa_mean": kappa_summary["mean"],
                        "kappa_std": kappa_summary["std"],
                        "n_pairs": percent_summary["n_pairs"],
                        "n_items": percent_summary["n_items"],
                        "percent_excl_mean": percent_excl_summary["mean"],
                        "percent_excl_std": percent_excl_summary["std"],
                        "kappa_excl_mean": kappa_excl_summary["mean"],
                        "kappa_excl_std": kappa_excl_summary["std"],
                        "n_pairs_excl": percent_excl_summary["n_pairs"],
                        "n_items_excl": percent_excl_summary["n_items"],
                    },
                )
            )
        return pair_rows

    for prompt_type in prompt_types:
        prompt_sample_df = sample_df[
            sample_df["__prompt_type__"].astype(str) == str(prompt_type)
        ].copy()
        if prompt_sample_df.empty:
            continue

        prompt_dimensions = [
            dimension
            for dimension in dimensions
            if f"dim_{dimension}_winner_label" in prompt_sample_df.columns
            and prompt_sample_df[f"dim_{dimension}_winner_label"]
            .dropna()
            .astype(str)
            .str.strip()
            .ne("")
            .any()
        ]
        if not prompt_dimensions:
            continue

        dimension_rows: List[Tuple[str, List[Tuple[str, Dict[str, object]]]]] = []
        pooled_outcomes_by_judge: Dict[str, Dict[str, str]] = {}
        for dimension in prompt_dimensions:
            winner_column = f"dim_{dimension}_winner_label"
            outcomes_by_judge_dim: Dict[str, Dict[str, str]] = {}
            for _, row in prompt_sample_df[["item_id", judge_column, winner_column]].iterrows():
                item_id = _safe_str(row.get("item_id")).strip()
                judge = _safe_str(row.get(judge_column)).strip()
                winner = _safe_str(row.get(winner_column)).strip()
                if not item_id or not judge or not winner:
                    continue
                judge_map = outcomes_by_judge_dim.setdefault(judge, {})
                existing = judge_map.get(item_id)
                if existing is not None and existing != winner:
                    raise ValueError(
                        "Conflicting dimension winner labels for the same item/judge "
                        "when building Streamlit-style dimension agreement LaTeX. "
                        f"dimension={dimension!r} judge={judge!r} item_id={item_id!r} "
                        f"existing={existing!r} new={winner!r}"
                    )
                judge_map[item_id] = winner

                pooled_item_id = f"{item_id}::{dimension}"
                pooled_judge_map = pooled_outcomes_by_judge.setdefault(judge, {})
                pooled_existing = pooled_judge_map.get(pooled_item_id)
                if pooled_existing is not None and pooled_existing != winner:
                    raise ValueError(
                        "Conflicting pooled dimension winner labels for the same "
                        "synthetic item/judge when building Streamlit-style dimension "
                        "agreement LaTeX. "
                        f"dimension={dimension!r} judge={judge!r} "
                        f"pooled_item_id={pooled_item_id!r} "
                        f"existing={pooled_existing!r} new={winner!r}"
                    )
                pooled_judge_map[pooled_item_id] = winner

            dimension_item_ids = sorted(
                {
                    _safe_str(value).strip()
                    for value in prompt_sample_df["item_id"].dropna().astype(str).tolist()
                    if _safe_str(value).strip()
                }
            )
            pair_rows = _compute_pair_type_summaries(
                outcomes_by_judge_dim,
                dimension_item_ids,
            )
            dimension_rows.append((dimension, pair_rows))

        pooled_item_ids = sorted(
            {
                item_id
                for judge_items in pooled_outcomes_by_judge.values()
                for item_id in judge_items.keys()
            }
        )
        pooled_pair_rows = _compute_pair_type_summaries(
            pooled_outcomes_by_judge,
            pooled_item_ids,
        )
        prompt_dimension_summaries.append((prompt_type, dimension_rows, pooled_pair_rows))

    if not prompt_dimension_summaries:
        raise ValueError(
            "No prompt types with usable dimension winner labels were available for "
            "Streamlit-style dimension agreement LaTeX."
        )

    lines: List[str] = []
    lines.append("% Auto-generated by stage_6_analyze_results.py")
    if disable_metrics:
        lines.append(
            "% Streamlit-style dimension agreement metrics were disabled via CLI."
        )
    elif total_usable_pair_rows <= 0:
        lines.append(
            "% No usable dimension-level judge-pair agreement rows were available; "
            "metrics are rendered as NA and counts as 0."
        )

    for prompt_index, (
        prompt_type,
        dimension_rows,
        pooled_pair_rows,
    ) in enumerate(prompt_dimension_summaries):
        prompt_caption = _build_prompt_caption(
            prompt_type,
            dimension_rows,
            pooled_pair_rows,
        )
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\setlength{\\tabcolsep}{4pt}")
        lines.append("\\renewcommand{\\arraystretch}{1.12}")
        lines.append("\\begin{tabular}{@{}c|c|cc|cc@{}}")
        lines.append("\\toprule")
        lines.append(
            "\\textbf{Dimension} & \\textbf{Pair Type} & "
            "\\textbf{Agr. (\\%)} & \\textbf{$\\kappa$} & "
            "\\textbf{Agr. excl. ties (\\%)} & "
            "\\textbf{$\\kappa$ excl. ties} \\\\"
        )
        lines.append("\\midrule")

        for dimension_index, (dimension, pair_rows) in enumerate(dimension_rows):
            dimension_label = _streamlit_dimension_display_name(dimension)
            for row_index, (pair_label, summary) in enumerate(pair_rows):
                columns: List[str] = []
                if row_index == 0:
                    columns.append(
                        f"\\multirow{{{len(pair_rows)}}}{{*}}{{\\textbf{{{_escape_latex_text(dimension_label)}}}}}"
                    )
                else:
                    columns.append("")

                columns.append(_escape_latex_text(pair_label))
                columns.append(
                    _format_pmstd(
                        summary["percent_mean"],
                        summary["percent_std"],
                        decimals=1,
                        scale=100.0,
                        missing_text="NA",
                    )
                )
                columns.append(
                    _format_pmstd(
                        summary["kappa_mean"],
                        summary["kappa_std"],
                        decimals=2,
                        missing_text="NA",
                    )
                )
                columns.append(
                    _format_pmstd(
                        summary["percent_excl_mean"],
                        summary["percent_excl_std"],
                        decimals=1,
                        scale=100.0,
                        missing_text="NA",
                    )
                )
                columns.append(
                    _format_pmstd(
                        summary["kappa_excl_mean"],
                        summary["kappa_excl_std"],
                        decimals=2,
                        missing_text="NA",
                    )
                )
                lines.append(" & ".join(columns) + " \\\\")

            if dimension_index < len(dimension_rows) - 1 or pooled_pair_rows:
                lines.append("\\cmidrule(lr){1-6}")

        if pooled_pair_rows:
            for row_index, (pair_label, summary) in enumerate(pooled_pair_rows):
                columns = []
                if row_index == 0:
                    columns.append("\\multirow{4}{*}{\\textbf{Pooled}}")
                else:
                    columns.append("")
                columns.append(_escape_latex_text(pair_label))
                columns.append(
                    _format_pmstd(
                        summary["percent_mean"],
                        summary["percent_std"],
                        decimals=1,
                        scale=100.0,
                        missing_text="NA",
                    )
                )
                columns.append(
                    _format_pmstd(
                        summary["kappa_mean"],
                        summary["kappa_std"],
                        decimals=2,
                        missing_text="NA",
                    )
                )
                columns.append(
                    _format_pmstd(
                        summary["percent_excl_mean"],
                        summary["percent_excl_std"],
                        decimals=1,
                        scale=100.0,
                        missing_text="NA",
                    )
                )
                columns.append(
                    _format_pmstd(
                        summary["kappa_excl_mean"],
                        summary["kappa_excl_std"],
                        decimals=2,
                        missing_text="NA",
                    )
                )
                lines.append(" & ".join(columns) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{" + _escape_latex_text(prompt_caption) + "}")
        lines.append(
            "\\label{"
            + _escape_latex_text(f"{label}_{_safe_str(prompt_type).strip()}")
            + "}"
        )
        lines.append("\\end{table}")
        if prompt_index < len(prompt_dimension_summaries) - 1:
            lines.append("")

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info(
        "Wrote Streamlit-style dimension agreement LaTeX table to %s",
        output_path,
    )
    return destination


def write_joint_preference_judge_agreement_latex(
    pairwise_df: pd.DataFrame,
    output_path: str,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 42,
    judge_column: str = "judge_model_name",
    disable_metrics: bool = False,
    caption: str = (
        "LLM-judge agreement for the subjective pairwise preference labels, "
        "reported over multiple pooled slices of the evaluation. For each subset, "
        "we compute condition-level judge agreement and then aggregate those "
        "condition metrics using sample-count weighting. For each condition, "
        "we compute (i) \\textbf{raw agreement} as the mean percentage of items on "
        "which two judges output the same label, and (ii) \\textbf{Fleiss' "
        "$\\kappa$}, which adjusts agreement for chance given the judges' marginal "
        "label distributions (note that when one model is selected as the winner "
        "in most items, the resulting label imbalance can lower $\\kappa$ despite "
        "high raw agreement). In the special case of exactly two judges, Fleiss' "
        "$\\kappa$ is equivalent to Cohen's $\\kappa$. Values are reported as "
        "\\(\\text{mean} \\pm \\text{std}\\), where the mean and standard "
        "deviation are computed across conditions and weighted by the number of "
        "samples in each condition."
    ),
    label: str = "tab:judge_judges_agreement",
) -> Path:
    """
    Write a paper-style LaTeX table summarizing pooled LLM-judge agreement.

    The exported layout follows the fixed paper-table structure requested by the
    user: model-pair rows, persona rows, prompt-type rows, judge-pair rows, and
    one overall row. Missing subsets are rendered as ``--`` placeholders instead
    of being silently omitted.

    Args:
        pairwise_df (pd.DataFrame): Canonical Stage-6 pairwise sample dataframe.
        output_path (str): Destination .tex path.
        n_bootstrap (int): Number of bootstrap resamples for pooled agreement
            summaries.
        seed (int): RNG seed for pooled agreement summaries.
        judge_column (str): Column storing judge identifiers.
        disable_metrics (bool): If True, do not compute metrics and instead write
            the fixed table structure with placeholder values plus explanatory
            comments.
        caption (str): LaTeX caption.
        label (str): LaTeX label.

    Returns:
        Path: Written .tex file.

    Raises:
        ValueError: If required input columns are missing.
        JudgeAgreementError: If pooled judge-agreement computation encounters
            invalid or ambiguous data.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if pairwise_df is None:
        raise ValueError("pairwise_df is None; cannot export judge agreement LaTeX.")

    if pairwise_df.empty:
        frame = pd.DataFrame()
    else:
        required = [
            "user_id",
            "variant_label",
            "task_id",
            "variant_id",
            "model_a_name",
            "model_b_name",
            "overall_winner_label",
        ]
        missing = [col for col in required if col not in pairwise_df.columns]
        if missing:
            raise ValueError(
                "pairwise_df is missing required columns for judge agreement LaTeX: "
                + ", ".join(missing)
            )
        if judge_column not in pairwise_df.columns:
            raise ValueError(
                "pairwise_df is missing the judge column required for judge agreement "
                f"LaTeX export: {judge_column}"
            )
        frame = pairwise_df.copy()

    def _escape(text: str) -> str:
        return (
            str(text)
            .replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("^", "\\^{}")
            .replace("~", "\\~{}")
        )

    preferred_pairs = {
        frozenset({"GPT-5.1", "GPT-OSS-20B"}): ("GPT-5.1", "GPT-OSS-20B"),
        frozenset({"GPT-5.1", "GPT-4o"}): ("GPT-5.1", "GPT-4o"),
        frozenset({"Gemini-3-Pro", "Gemma-3-4B"}): ("Gemini-3-Pro", "Gemma-3-4B"),
        frozenset({"Qwen3-32B", "Qwen3-14B"}): ("Qwen3-32B", "Qwen3-14B"),
    }
    prompt_type_labels = {
        "original": "Original",
        "control": "Control",
        "personalized": "Personalized",
    }

    def _translate_model_name(model_name: object) -> str:
        if model_name is None or pd.isna(model_name):
            return ""
        return display_model_name(model_name)

    def _translate_persona_name(persona_name: object) -> str:
        if persona_name is None or pd.isna(persona_name):
            return ""
        raw = str(persona_name).strip()
        normalized = raw.lower().replace("-", "_")
        for token in ["python_novice", "python_beginner", "novice_user"]:
            if normalized.startswith(token) or token in normalized:
                return "Beginner"
        for token in ["python_intermediate", "intermediate_learner"]:
            if normalized.startswith(token) or token in normalized:
                return "Intermediate"
        for token in ["researcher_user", "ai_researcher", "python_researcher"]:
            if normalized.startswith(token) or token in normalized:
                return "Researcher"
        for token in ["advanced_developer", "python_advanced"]:
            if normalized.startswith(token) or token in normalized:
                return "Advanced"
        return raw.replace("_", " ").title()

    def _safe_str(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value)

    def _ordered_display_pair(first: object, second: object) -> str:
        first_txt = _translate_model_name(first)
        second_txt = _translate_model_name(second)
        if not first_txt or not second_txt:
            return ""
        preferred = preferred_pairs.get(frozenset({first_txt, second_txt}))
        if preferred is not None:
            return f"{preferred[0]} vs {preferred[1]}"
        ordered = sorted([first_txt, second_txt])
        return f"{ordered[0]} vs {ordered[1]}"

    def _display_prompt_type(prompt_type: object) -> str:
        raw = _safe_str(prompt_type).strip().lower()
        return prompt_type_labels.get(raw, raw.title())

    def _winner_to_outcome(
        row: pd.Series,
        *,
        desired_pair: Optional[Tuple[str, str]] = None,
    ) -> Optional[str]:
        winner = _safe_str(row.get("overall_winner_label")).strip()
        if not winner:
            return None
        winner_lower = winner.lower()
        if winner_lower == "tie":
            return "tie"

        raw_model_a = _safe_str(row.get("model_a_name"))
        raw_model_b = _safe_str(row.get("model_b_name"))
        disp_model_a = _translate_model_name(raw_model_a)
        disp_model_b = _translate_model_name(raw_model_b)

        if winner_lower in {"model_a", "row"} or winner == raw_model_a:
            winner_model = disp_model_a
        elif winner_lower in {"model_b", "col"} or winner == raw_model_b:
            winner_model = disp_model_b
        elif _translate_model_name(winner) in {disp_model_a, disp_model_b}:
            winner_model = _translate_model_name(winner)
        else:
            raise JudgeAgreementError(
                "Unsupported winner label in pairwise_df for judge agreement LaTeX: "
                f"{winner}"
            )

        if desired_pair is None:
            return "row" if winner_model == disp_model_a else "col"

        first_model, second_model = desired_pair
        if winner_model == first_model:
            return "row"
        if winner_model == second_model:
            return "col"
        raise JudgeAgreementError(
            "Winner model does not match the requested fixed pair direction for "
            f"judge agreement LaTeX: winner={winner_model}, pair={desired_pair}"
        )

    def _filter_subset_rows(
        subset_df: pd.DataFrame,
        *,
        desired_judges: Optional[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        if subset_df is None or subset_df.empty:
            return pd.DataFrame()

        subset = subset_df.copy()
        if desired_judges is not None:
            desired = set(desired_judges)
            subset = subset[subset["display_judge"].astype(str).isin(desired)].copy()
        return subset

    def _item_id_for_row(row: pd.Series) -> str:
        pair_key = "||".join(
            sorted(
                [
                    _safe_str(row.get("model_a_name")),
                    _safe_str(row.get("model_b_name")),
                ]
            )
        )
        return "||".join(
            [
                _safe_str(row.get("user_id")),
                _safe_str(row.get("variant_label")),
                _safe_str(row.get("task_id")),
                _safe_str(row.get("variant_id")),
                pair_key,
            ]
        )

    def _condition_key_for_row(row: pd.Series, grouping: str) -> str:
        parts = [
            _safe_str(row.get(col)).strip()
            for col in _condition_columns_for_grouping(grouping)
            if _safe_str(row.get(col)).strip()
        ]
        return " / ".join(parts)

    def _build_item_judge_table(
        subset_df: pd.DataFrame,
        *,
        desired_pair: Optional[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        if subset_df is None or subset_df.empty:
            return pd.DataFrame()

        rows: List[Dict[str, str]] = []
        for _, r in subset_df.iterrows():
            raw_judge = _safe_str(r.get(judge_column)).strip()
            if not raw_judge:
                continue
            rows.append(
                {
                    "item_id": _item_id_for_row(r),
                    "judge": raw_judge,
                    "outcome": _winner_to_outcome(r, desired_pair=desired_pair),
                }
            )

        if not rows:
            return pd.DataFrame()

        tidy = pd.DataFrame(rows)
        if tidy.duplicated(subset=["item_id", "judge"], keep=False).any():
            dupes = tidy[
                tidy.duplicated(subset=["item_id", "judge"], keep=False)
            ].copy()
            ambiguous = (
                dupes.groupby(["item_id", "judge"])["outcome"]
                .nunique(dropna=False)
                .reset_index(name="n_outcomes")
            )
            if (ambiguous["n_outcomes"] > 1).any():
                raise JudgeAgreementError(
                    "Judge agreement LaTeX export encountered conflicting duplicate "
                    "ratings after display-name normalization."
                )
            tidy = tidy.drop_duplicates(subset=["item_id", "judge"], keep="first")

        return tidy.pivot(
            index="item_id", columns="judge", values="outcome"
        ).sort_index()

    def _condition_columns_for_grouping(grouping: str) -> List[str]:
        if grouping == "persona":
            return ["display_prompt_type", "display_model_pair"]
        if grouping == "prompt_type":
            return ["display_persona", "display_model_pair"]
        return ["display_persona", "display_prompt_type", "display_model_pair"]

    def _condition_labels_for_subset(
        subset_df: pd.DataFrame, grouping: str
    ) -> List[str]:
        if subset_df is None or subset_df.empty:
            return []

        cols = _condition_columns_for_grouping(grouping)
        available = [col for col in cols if col in subset_df.columns]
        if not available:
            return []

        distinct = subset_df[available].drop_duplicates().copy()
        if distinct.empty:
            return []

        labels: List[str] = []
        for _, row in distinct.iterrows():
            parts = [
                _safe_str(row.get(col)).strip()
                for col in available
                if _safe_str(row.get(col)).strip()
            ]
            if parts:
                labels.append(" / ".join(parts))
        return sorted(set(labels))

    def _empty_summary() -> Dict[str, object]:
        return {
            "agreement_mean": float("nan"),
            "agreement_std": float("nan"),
            "fleiss_mean": float("nan"),
            "fleiss_std": float("nan"),
            "n_judges": 0,
            "n_conditions": 0,
            "raw_judges": [],
            "condition_labels": [],
        }

    def _compute_subset_summary(
        subset_df: pd.DataFrame,
        grouping: str,
        *,
        desired_pair: Optional[Tuple[str, str]] = None,
        desired_judges: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, object]:
        filtered_subset = _filter_subset_rows(
            subset_df,
            desired_judges=desired_judges,
        )
        if filtered_subset.empty:
            return _empty_summary()

        table = _build_item_judge_table(
            filtered_subset,
            desired_pair=desired_pair,
        )
        raw_judges = (
            sorted(table.columns.astype(str).tolist())
            if table is not None and not table.empty
            else sorted(
                filtered_subset[judge_column].dropna().astype(str).unique().tolist()
            )
        )
        condition_labels = _condition_labels_for_subset(filtered_subset, grouping)

        summary: Dict[str, object] = {
            "agreement_mean": float("nan"),
            "agreement_std": float("nan"),
            "fleiss_mean": float("nan"),
            "fleiss_std": float("nan"),
            "n_judges": int(len(raw_judges)),
            "n_conditions": int(len(condition_labels)),
            "raw_judges": raw_judges,
            "condition_labels": condition_labels,
        }
        if disable_metrics or table.empty or table.shape[1] < 2:
            return summary

        item_condition_ids = (
            filtered_subset.assign(
                item_id=filtered_subset.apply(_item_id_for_row, axis=1),
                condition_id=filtered_subset.apply(
                    lambda row: _condition_key_for_row(row, grouping),
                    axis=1,
                ),
            )[["item_id", "condition_id"]]
            .drop_duplicates(subset=["item_id"], keep="first")
            .set_index("item_id")["condition_id"]
            .reindex(table.index)
        )

        pooled = compute_pooled_judge_agreement_metrics(
            table,
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
            condition_ids=item_condition_ids,
        )
        pooled_summary = pooled["summary"]
        summary.update(
            {
                "agreement_mean": float(
                    pooled_summary.get("percent_agreement_mean_pairs", float("nan"))
                ),
                "agreement_std": float(
                    pooled_summary.get("percent_agreement_std_pairs", float("nan"))
                ),
                "fleiss_mean": float(
                    pooled_summary.get("fleiss_kappa_mean_pairs", float("nan"))
                ),
                "fleiss_std": float(
                    pooled_summary.get("fleiss_kappa_std_pairs", float("nan"))
                ),
            }
        )
        return summary

    def _is_missing(value: object) -> bool:
        return value is None or pd.isna(value)

    def _format_count(value: object) -> str:
        if _is_missing(value):
            return "--"
        return str(int(value))

    def _format_scalar(
        mean: object,
        std: object,
        *,
        bold_mean: bool = False,
    ) -> str:
        if _is_missing(mean):
            return "--"
        mean_txt = f"{float(mean):.2f}"
        if bold_mean:
            mean_txt = f"\\textbf{{{mean_txt}}}"
        if _is_missing(std):
            return mean_txt
        return f"\\pmstd{{{mean_txt}}}{{{float(std):.2f}}}"

    model_pair_rows = [
        ("GPT-5.1", "GPT-OSS-20B"),
        ("GPT-5.1", "GPT-4o"),
        ("Gemini-3-Pro", "Gemma-3-4B"),
        ("Qwen3-32B", "Qwen3-14B"),
    ]
    persona_rows = ["Beginner", "Intermediate", "Researcher", "Advanced"]
    prompt_type_rows = [
        ("Original", "original"),
        ("Control", "control"),
        ("Personalized", "personalized"),
    ]
    judge_pair_rows = [
        ("GPT-5.1", "GPT-OSS-20B"),
        ("GPT-5.1", "Qwen3-14B"),
        ("Qwen3-14B", "GPT-OSS-20B"),
    ]

    comments = [
        "% Auto-generated by stage_6_analyze_results.py",
        "% Assumption: the fixed row labels and directions match the user-provided paper example exactly.",
        "% Assumption: rows missing from the current run are rendered as '--' placeholders instead of being omitted.",
        "% Assumption: opposite directions of the same model pair are normalized into the fixed direction shown in the table.",
        "% Assumption: known aliases (for example unsloth/openai paths, effort-specific names, and OLD_ prefixes) are canonicalized before judge counting and agreement aggregation.",
    ]
    if disable_metrics:
        comments.append(
            "% Judge-agreement metrics were disabled via CLI, so this table contains only placeholders."
        )

    if not frame.empty:
        frame = frame.copy()
        frame["model_a_name"] = frame["model_a_name"].map(canonicalize_model_name)
        frame["model_b_name"] = frame["model_b_name"].map(canonicalize_model_name)
        frame[judge_column] = frame[judge_column].map(canonicalize_model_name)
        frame["display_model_a"] = frame["model_a_name"].map(_translate_model_name)
        frame["display_model_b"] = frame["model_b_name"].map(_translate_model_name)
        frame["display_judge"] = frame[judge_column].map(_translate_model_name)
        frame["display_persona"] = frame["user_id"].map(_translate_persona_name)
        frame["display_prompt_type"] = frame["variant_label"].map(_display_prompt_type)
        frame["display_model_pair"] = frame.apply(
            lambda row: _ordered_display_pair(
                row.get("display_model_a"),
                row.get("display_model_b"),
            ),
            axis=1,
        )

    subset_summaries: Dict[Tuple[str, str], Dict[str, object]] = {}
    if frame.empty:
        subset_summaries = {}
    else:
        available_judges = sorted(
            frame[judge_column].dropna().astype(str).unique().tolist()
        )
        if len(available_judges) < 2:
            comments.append(
                "% Fewer than two raw judges were available, so agreement metrics could not be computed."
            )
        for first_model, second_model in model_pair_rows:
            subset = frame[
                (
                    (frame["display_model_a"] == first_model)
                    & (frame["display_model_b"] == second_model)
                )
                | (
                    (frame["display_model_a"] == second_model)
                    & (frame["display_model_b"] == first_model)
                )
            ].copy()
            subset_summaries[("model_pair", f"{first_model} vs {second_model}")] = (
                _compute_subset_summary(
                    subset,
                    "model_pair",
                    desired_pair=(first_model, second_model),
                )
            )

        for persona_label in persona_rows:
            subset = frame[frame["display_persona"] == persona_label].copy()
            subset_summaries[("persona", persona_label)] = _compute_subset_summary(
                subset,
                "persona",
            )

        for prompt_label, prompt_value in prompt_type_rows:
            subset = frame[
                frame["variant_label"].astype(str) == str(prompt_value)
            ].copy()
            subset_summaries[("prompt_type", prompt_label)] = _compute_subset_summary(
                subset,
                "prompt_type",
            )

        for judge_a, judge_b in judge_pair_rows:
            subset_summaries[("judge_pair", f"{judge_a} vs {judge_b}")] = (
                _compute_subset_summary(
                    frame,
                    "judge_pair",
                    desired_judges=(judge_a, judge_b),
                )
            )

        subset_summaries[("overall", "All Samples")] = _compute_subset_summary(
            frame,
            "overall",
        )

    def _summary_for(grouping: str, subset: str) -> Dict[str, object]:
        return subset_summaries.get((grouping, subset), _empty_summary())

    def _format_subset_line(
        grouping: str,
        subset: str,
        *,
        bold_mean: bool = False,
    ) -> str:
        metrics = _summary_for(grouping, subset)
        return (
            f"{_format_count(metrics['n_judges'])}"
            f" & {_format_count(metrics['n_conditions'])}"
            f" & {_format_scalar(metrics['agreement_mean'], metrics['agreement_std'], bold_mean=bold_mean)}"
            f" & {_format_scalar(metrics['fleiss_mean'], metrics['fleiss_std'], bold_mean=bold_mean)}"
        )

    lines: List[str] = list(comments)
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{7pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\begin{tabular}{@{}c|c|cc|cc@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Grouping} & \\textbf{Subset} & \\textbf{\\# Judges} & "
        "\\textbf{\\# Conditions} & \\textbf{Agreement (\\%)} & "
        "\\textbf{Fleiss's $\\kappa$} \\\\"
    )
    lines.append("\\midrule")
    lines.append("")
    lines.append(f"\\multirow{{{len(model_pair_rows)}}}{{*}}{{\\textbf{{Model pair}}}} ")
    for index, (first_model, second_model) in enumerate(model_pair_rows):
        prefix = "&" if index > 0 else "&"
        label_txt = (
            f"\\texttt{{{_escape(first_model)}}} \\textit{{vs.}} "
            f"\\texttt{{{_escape(second_model)}}}"
        )
        lines.append(
            f"{prefix} {label_txt} & "
            + _format_subset_line("model_pair", f"{first_model} vs {second_model}")
            + " \\\\"
        )
    lines.append("\\midrule")
    lines.append("")
    lines.append("\\multirow{4}{*}{\\textbf{Persona}} ")
    for persona_label in persona_rows:
        lines.append(
            f"& {_escape(persona_label)} & "
            + _format_subset_line("persona", persona_label)
            + " \\\\"
        )
    lines.append("\\midrule")
    lines.append("")
    lines.append("\\multirow{3}{*}{\\textbf{Prompt type}} ")
    for prompt_label, _prompt_value in prompt_type_rows:
        lines.append(
            f"& {_escape(prompt_label)} & "
            + _format_subset_line("prompt_type", prompt_label)
            + " \\\\"
        )
    lines.append("\\midrule")
    lines.append("")
    lines.append("\\multirow{3}{*}{\\textbf{Judge pair}} ")
    for judge_a, judge_b in judge_pair_rows:
        judge_label = (
            f"\\texttt{{{_escape(judge_a)}}} \\textit{{vs.}} "
            f"\\texttt{{{_escape(judge_b)}}}"
        )
        lines.append(
            f"& {judge_label} & "
            + _format_subset_line("judge_pair", f"{judge_a} vs {judge_b}")
            + " \\\\"
        )
    lines.append("\\midrule")
    lines.append("\\midrule")
    lines.append(
        "\\textbf{Overall} & All Samples & "
        + _format_subset_line("overall", "All Samples", bold_mean=True)
        + " \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + _escape(label) + "}")
    lines.append("\\end{table*}")

    row_comment_order: List[Tuple[str, str, str]] = []
    row_comment_order.extend(
        [
            ("Model pair", "model_pair", f"{first} vs {second}")
            for first, second in model_pair_rows
        ]
    )
    row_comment_order.extend(
        [("Persona", "persona", persona_label) for persona_label in persona_rows]
    )
    row_comment_order.extend(
        [
            ("Prompt type", "prompt_type", prompt_label)
            for prompt_label, _ in prompt_type_rows
        ]
    )
    row_comment_order.extend(
        [
            ("Judge pair", "judge_pair", f"{judge_a} vs {judge_b}")
            for judge_a, judge_b in judge_pair_rows
        ]
    )
    row_comment_order.append(("Overall", "overall", "All Samples"))

    lines.append("%")
    lines.append(
        "% Row provenance (normalized conditions used for each displayed row):"
    )
    for grouping_display, grouping_key, subset_label in row_comment_order:
        summary = _summary_for(grouping_key, subset_label)
        canonical_judges = summary.get("raw_judges") or []
        canonical_judge_text = (
            "; ".join(str(value) for value in canonical_judges)
            if canonical_judges
            else "none"
        )
        condition_labels = summary.get("condition_labels") or []
        condition_text = (
            "; ".join(str(value) for value in condition_labels)
            if condition_labels
            else "none"
        )
        lines.append(f"% - {grouping_display} | {subset_label}")
        lines.append(
            f"%   canonical_judges_used ({int(summary.get('n_judges', 0))}): {canonical_judge_text}"
        )
        lines.append(
            f"%   conditions_used ({int(summary.get('n_conditions', 0))}): {condition_text}"
        )

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote joint preference judge agreement LaTeX table to %s", output_path)
    return destination


def _pairwise_column_order(columns: Iterable[str]) -> List[str]:
    """Produce a consistent ordering for pairwise CSV columns."""
    existing = list(columns)
    ordered: List[str] = []

    # Model pair columns first
    ordered.extend([col for col in PAIRWISE_ID_COLUMNS if col in existing])

    # Variant and user columns
    ordered.extend([col for col in ["variant_label", "user_id"] if col in existing])

    # Count and rate columns
    count_cols = [
        col for col in existing if "count" in col.lower() or "total" in col.lower()
    ]
    rate_cols = [
        col for col in existing if "rate" in col.lower() or "prob" in col.lower()
    ]
    ordered.extend(sorted([col for col in count_cols if col not in ordered]))
    ordered.extend(sorted([col for col in rate_cols if col not in ordered]))

    # Remaining columns
    remainder = [col for col in existing if col not in ordered]
    return ordered + sorted(remainder)
