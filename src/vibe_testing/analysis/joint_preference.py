"""Joint (multi-model) preference matrix computation for Stage 6 analysis.

This module aggregates Stage 5B pairwise comparison outcomes into joint
model-by-model win-rate matrices per (persona, prompt type).

Key ideas:
- Each pairwise comparison row corresponds to one "sample".
- For each sample and model pair, we use the per-sample overall winner
  (already aggregated across vibe dimensions by Stage 5B / AnalysisInputLoader).
- Win rates include ties in the denominator: win_rate = wins / total.
- Statistical significance is computed using a two-sided binomial test on
  non-tie outcomes (H0: p = 0.5). For display, we mark only the cell for the
  winner (win rate > 0.5) with '*'.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)

PROMPT_TYPES = ("original", "personalized", "control")
OBJECTIVE_PASS_AT_1_COLUMNS = (
    "obj_overall_pass_at_1",
    "obj_plus_pass_at_1",
    "obj_base_pass_at_1",
)


class JointPreferenceError(RuntimeError):
    """Raised when joint preference matrix computation encounters invalid data."""


class ObjectiveConsistencyError(RuntimeError):
    """Raised when objective consistency/flip analysis encounters invalid data."""


def compute_objective_accuracy_consistency_metrics(
    objective_df: pd.DataFrame,
    *,
    original_label: str = "original",
    treatment_labels: Optional[Iterable[str]] = None,
    baseline_user_id: Optional[str] = None,
    pass_at_1_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute objective accuracy consistency (flip / reverse-flip) vs original prompts.

    This quantifies how often a model's binary correctness outcome changes between the
    original condition and a treatment condition (e.g., personalized/control).

    Correctness rule:
        correct iff pass@1 >= ``pass_at_1_threshold``.

    Flip definitions (per base task):
        - flip: original_correct=1 and treatment_correct=0
        - reverse_flip: original_correct=0 and treatment_correct=1

    Args:
        objective_df (pd.DataFrame): Stage-4 objective metrics at sample-level. Must
            include columns: user_id, model_name, task_id, variant_label, and one of
            the pass@1 columns in ``OBJECTIVE_PASS_AT_1_COLUMNS``.
        original_label (str): Variant label for the baseline condition.
        treatment_labels (Optional[Iterable[str]]): Labels to compute consistency for.
            Defaults to ("personalized", "control").
        baseline_user_id (Optional[str]): If provided, use the original-condition
            objective rows for this user_id as the baseline for *all* users. This is
            useful when Stage 4 was optimized to run `original` only for a reference
            persona, but `personalized` was run per persona.
        pass_at_1_threshold (float): Threshold for converting pass@1 into binary
            correctness (default: 0.5).

    Returns:
        pd.DataFrame: One row per (persona, prompt_type, model_name) with columns:
            - persona
            - prompt_type
            - model_name
            - objective_flip_rate_vs_original
            - objective_reverse_flip_rate_vs_original

    Raises:
        ObjectiveConsistencyError: If required columns are missing/invalid, pass@1
            cannot be located, or original rows are missing.
    """
    if objective_df is None or objective_df.empty:
        raise ObjectiveConsistencyError(
            "objective_df is empty; cannot compute objective accuracy consistency metrics."
        )

    required_cols = ["user_id", "model_name", "task_id", "variant_label"]
    missing = [c for c in required_cols if c not in objective_df.columns]
    if missing:
        raise ObjectiveConsistencyError(
            "objective_df is missing required columns: " + ", ".join(missing)
        )

    threshold = float(pass_at_1_threshold)
    if threshold <= 0.0:
        raise ObjectiveConsistencyError(
            f"pass_at_1_threshold must be > 0. Got: {pass_at_1_threshold}"
        )

    pass_col = next(
        (c for c in OBJECTIVE_PASS_AT_1_COLUMNS if c in objective_df.columns), None
    )
    if pass_col is None:
        raise ObjectiveConsistencyError(
            "objective_df is missing a pass@1 column. Expected one of: "
            + ", ".join(OBJECTIVE_PASS_AT_1_COLUMNS)
        )

    treat_labels = (
        list(treatment_labels)
        if treatment_labels is not None
        else [pt for pt in PROMPT_TYPES if pt != str(original_label)]
    )
    treat_labels = [str(x) for x in treat_labels if str(x)]
    if not treat_labels:
        raise ObjectiveConsistencyError(
            "treatment_labels resolved to an empty set; nothing to compute."
        )

    df = objective_df.copy()

    null_required = [c for c in required_cols if df[c].isna().any()]
    if null_required:
        counts = {c: int(df[c].isna().sum()) for c in null_required}
        raise ObjectiveConsistencyError(
            "objective_df has nulls in required columns. Null counts: " + str(counts)
        )

    df["user_id"] = df["user_id"].astype(str)
    df["model_name"] = df["model_name"].astype(str)
    df["task_id"] = df["task_id"].astype(str)
    df["variant_label"] = df["variant_label"].astype(str)

    pass_at_1 = pd.to_numeric(df[pass_col], errors="coerce")
    if pass_at_1.isna().all():
        sample_vals = df[pass_col].head(5).tolist()
        raise ObjectiveConsistencyError(
            f"Pass@1 column '{pass_col}' could not be parsed as numeric (all NaN). "
            f"Sample values: {sample_vals}"
        )
    df["_is_correct"] = pass_at_1 >= threshold

    original = df[df["variant_label"] == str(original_label)].copy()
    if original.empty:
        raise ObjectiveConsistencyError(
            f"No rows found for original_label='{original_label}'."
        )
    if baseline_user_id is not None:
        original = original[original["user_id"] == str(baseline_user_id)].copy()
        if original.empty:
            raise ObjectiveConsistencyError(
                "baseline_user_id was provided, but no original rows were found for "
                f"user_id='{baseline_user_id}'."
            )

    original = (
        original[["user_id", "model_name", "task_id", "_is_correct"]]
        .drop_duplicates(subset=["user_id", "model_name", "task_id"], keep="last")
        .rename(columns={"_is_correct": "_orig_correct"})
    )

    treatment = df[df["variant_label"].isin(set(treat_labels))].copy()
    if treatment.empty:
        return pd.DataFrame(
            columns=[
                "persona",
                "prompt_type",
                "model_name",
                "objective_flip_rate_vs_original",
                "objective_reverse_flip_rate_vs_original",
            ]
        )

    treatment = treatment[
        ["user_id", "model_name", "task_id", "variant_label", "_is_correct"]
    ].rename(columns={"variant_label": "prompt_type", "_is_correct": "_treat_correct"})

    treatment = treatment.drop_duplicates(
        subset=["user_id", "model_name", "task_id", "prompt_type"], keep="last"
    )

    if baseline_user_id is None:
        merged = treatment.merge(
            original, on=["user_id", "model_name", "task_id"], how="inner"
        )
    else:
        # Join treatment rows for each persona against the baseline persona's original rows
        # (original prompt assumed persona-independent when this mode is used).
        merged = treatment.merge(
            original[["model_name", "task_id", "_orig_correct"]],
            on=["model_name", "task_id"],
            how="inner",
        )
    if merged.empty:
        raise ObjectiveConsistencyError(
            "No comparable objective rows found after joining treatment rows to original "
            "rows. This usually indicates that original objective evaluations are missing "
            "for the tasks present in treatment conditions."
        )

    merged["_flip"] = merged["_orig_correct"] & (~merged["_treat_correct"])
    merged["_reverse_flip"] = (~merged["_orig_correct"]) & merged["_treat_correct"]

    out = (
        merged.groupby(["user_id", "prompt_type", "model_name"], dropna=False)
        .agg(
            objective_flip_rate_vs_original=("_flip", "mean"),
            objective_reverse_flip_rate_vs_original=("_reverse_flip", "mean"),
        )
        .reset_index()
    )
    out = out.rename(columns={"user_id": "persona"})
    return out


@dataclass(frozen=True)
class JointPreferenceMatrix:
    """
    Joint preference matrix outputs for a single (persona, prompt type) slice.

    Attributes:
        persona: Persona/user identifier.
        prompt_type: Prompt type (variant label).
        models: Stable model ordering used for matrices.
        win_rate_matrix: Square matrix of win rates (wins / total incl. ties).
        formatted_matrix: Square matrix of formatted strings (e.g., '0.62*').
        pairwise_long: Long-form table with enough info to recreate figures/tables.
    """

    persona: str
    prompt_type: str
    models: List[str]
    win_rate_matrix: pd.DataFrame
    formatted_matrix: pd.DataFrame
    pairwise_long: pd.DataFrame


def compute_joint_preference_matrices(
    pairwise_sample_df: pd.DataFrame,
    personas: Optional[Iterable[str]] = None,
    prompt_types: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
) -> Dict[Tuple[str, str], JointPreferenceMatrix]:
    """
    Compute joint preference matrices for each (persona, prompt_type).

    Args:
        pairwise_sample_df: Sample-level pairwise DataFrame from
            AnalysisInputLoader.load_pairwise_results(). Must include:
            - user_id
            - variant_label
            - model_a_name, model_b_name
            - overall_winner_label in {'model_a','model_b','tie'}
        personas: Optional iterable of personas to include. If None, uses all.
        prompt_types: Optional iterable of prompt types to include. If None, uses
            the canonical set (original, personalized, control).
        alpha: Significance threshold for '*' markers (two-sided binomial test).

    Returns:
        Dict mapping (persona, prompt_type) -> JointPreferenceMatrix.

    Raises:
        JointPreferenceError: If required columns are missing or alpha invalid.
    """
    if pairwise_sample_df is None or pairwise_sample_df.empty:
        return {}

    if not (0.0 < float(alpha) < 1.0):
        raise JointPreferenceError(f"alpha must be in (0,1). Got: {alpha}")

    required_cols = [
        "user_id",
        "variant_label",
        "model_a_name",
        "model_b_name",
        "overall_winner_label",
    ]
    missing = [c for c in required_cols if c not in pairwise_sample_df.columns]
    if missing:
        raise JointPreferenceError(
            "pairwise_sample_df is missing required columns: " + ", ".join(missing)
        )

    prompt_types_final = (
        list(prompt_types) if prompt_types is not None else list(PROMPT_TYPES)
    )
    personas_final = (
        list(personas)
        if personas is not None
        else sorted(pairwise_sample_df["user_id"].dropna().unique().tolist())
    )

    results: Dict[Tuple[str, str], JointPreferenceMatrix] = {}
    for persona in personas_final:
        persona_df = pairwise_sample_df[pairwise_sample_df["user_id"] == persona].copy()
        if persona_df.empty:
            continue
        for prompt_type in prompt_types_final:
            slice_df = persona_df[persona_df["variant_label"] == prompt_type].copy()
            if slice_df.empty:
                continue
            matrix = compute_joint_preference_matrix_for_slice(
                slice_df,
                persona=persona,
                prompt_type=prompt_type,
                alpha=alpha,
            )
            if matrix is not None:
                results[(persona, prompt_type)] = matrix
    return results


def compute_weighted_joint_preference_matrices(
    pairwise_sample_df: pd.DataFrame,
    persona_importance_weights: Dict[str, float],
    personas: Optional[Iterable[str]] = None,
    prompt_types: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
) -> Dict[Tuple[str, str], JointPreferenceMatrix]:
    """
    Compute persona-importance-weighted joint preference matrices.

    Args:
        pairwise_sample_df: Sample-level pairwise DataFrame.
        persona_importance_weights: Mapping from persona/user_id to non-negative
            aggregation weights.
        personas: Optional personas to include. Defaults to all available personas.
        prompt_types: Optional prompt types to include. Defaults to canonical prompt
            types.
        alpha: Significance threshold forwarded to per-persona matrix computation.

    Returns:
        Dict[Tuple[str, str], JointPreferenceMatrix]: Weighted matrices keyed by
        ``("weighted_persona_importance", prompt_type)``.
    """
    base = compute_joint_preference_matrices(
        pairwise_sample_df,
        personas=personas,
        prompt_types=prompt_types,
        alpha=alpha,
    )
    if not base:
        return {}

    prompt_types_final = (
        list(prompt_types) if prompt_types is not None else list(PROMPT_TYPES)
    )
    weighted_results: Dict[Tuple[str, str], JointPreferenceMatrix] = {}
    for prompt_type in prompt_types_final:
        bundles = [
            bundle
            for (persona, pt), bundle in base.items()
            if pt == prompt_type
            and float(persona_importance_weights.get(str(persona), 0.0)) > 0
        ]
        if not bundles:
            continue

        models = sorted({model for bundle in bundles for model in bundle.models})
        win_rate_matrix = pd.DataFrame(0.5, index=models, columns=models, dtype=float)
        pairwise_rows: List[Dict[str, Any]] = []

        for row_model in models:
            for col_model in models:
                if row_model == col_model:
                    continue
                numerator = 0.0
                denominator = 0.0
                for bundle in bundles:
                    weight = float(
                        persona_importance_weights.get(str(bundle.persona), 0.0)
                    )
                    if weight <= 0:
                        continue
                    if (
                        row_model not in bundle.win_rate_matrix.index
                        or col_model not in bundle.win_rate_matrix.columns
                    ):
                        continue
                    numerator += weight * float(
                        bundle.win_rate_matrix.loc[row_model, col_model]
                    )
                    denominator += weight
                if denominator > 0:
                    value = numerator / denominator
                    win_rate_matrix.loc[row_model, col_model] = value
                    pairwise_rows.append(
                        {
                            "persona": "weighted_persona_importance",
                            "prompt_type": prompt_type,
                            "row_model": row_model,
                            "col_model": col_model,
                            "win_rate": value,
                            "formatted_win_rate": f"{value:.2f}",
                        }
                    )

        formatted_matrix = win_rate_matrix.apply(
            lambda column: column.map(lambda value: f"{float(value):.2f}")
        )
        pairwise_long = pd.DataFrame(pairwise_rows)
        weighted_results[("weighted_persona_importance", prompt_type)] = (
            JointPreferenceMatrix(
                persona="weighted_persona_importance",
                prompt_type=prompt_type,
                models=models,
                win_rate_matrix=win_rate_matrix,
                formatted_matrix=formatted_matrix,
                pairwise_long=pairwise_long,
            )
        )
    return weighted_results


def compute_joint_preference_long_by_judge(
    pairwise_sample_df: pd.DataFrame,
    personas: Optional[Iterable[str]] = None,
    prompt_types: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    judge_column: str = "judge_model_name",
) -> pd.DataFrame:
    """
    Compute a long-form joint preference table split by judge model.

    This helper mirrors :func:`compute_joint_preference_matrices`, but instead of
    returning matrices, it concatenates the per-slice long-form rows and adds a
    judge identifier column so Stage 6 can export a judge-separated CSV.

    Args:
        pairwise_sample_df: Sample-level pairwise DataFrame from
            AnalysisInputLoader.load_pairwise_results().
        personas: Optional iterable of personas to include. If None, uses all.
        prompt_types: Optional iterable of prompt types to include. If None, uses
            the canonical set (original, personalized, control).
        alpha: Significance threshold for '*' markers (two-sided binomial test).
        judge_column: Name of the judge model column in ``pairwise_sample_df``.
            Defaults to ``judge_model_name`` (Stage 5B schema).

    Returns:
        pd.DataFrame: Concatenated long-form table with an added ``judge_column``.

    Raises:
        JointPreferenceError: If ``judge_column`` is missing or ``alpha`` is invalid.
    """
    if pairwise_sample_df is None or pairwise_sample_df.empty:
        return pd.DataFrame()

    if judge_column not in pairwise_sample_df.columns:
        raise JointPreferenceError(
            f"pairwise_sample_df is missing judge column '{judge_column}'. "
            "Cannot compute judge-separated joint preference table."
        )

    judges = sorted(pairwise_sample_df[judge_column].dropna().unique().tolist())
    if not judges:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for judge in judges:
        judge_df = pairwise_sample_df[pairwise_sample_df[judge_column] == judge].copy()
        if judge_df.empty:
            continue

        matrices = compute_joint_preference_matrices(
            judge_df, personas=personas, prompt_types=prompt_types, alpha=alpha
        )
        for bundle in matrices.values():
            if bundle.pairwise_long is None or bundle.pairwise_long.empty:
                continue
            df = bundle.pairwise_long.copy()
            df[judge_column] = judge
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    sort_cols = [
        col
        for col in [judge_column, "persona", "prompt_type", "row_model", "col_model"]
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def compute_joint_preference_matrix_for_slice(
    pairwise_slice_df: pd.DataFrame,
    persona: str,
    prompt_type: str,
    alpha: float = 0.05,
) -> Optional[JointPreferenceMatrix]:
    """
    Compute joint matrices for one persona + prompt type slice.

    Args:
        pairwise_slice_df: Filtered pairwise sample-level DataFrame for one
            persona and one prompt type.
        persona: Persona identifier (for metadata).
        prompt_type: Prompt type/variant label (for metadata).
        alpha: Significance threshold.

    Returns:
        JointPreferenceMatrix, or None if fewer than 2 models are present.
    """
    models = _collect_models(pairwise_slice_df)
    if len(models) < 2:
        LOGGER.info(
            "Skipping joint preference matrix for persona='%s', prompt_type='%s': "
            "need >=2 models, found %d",
            persona,
            prompt_type,
            len(models),
        )
        return None

    win_rate_matrix = pd.DataFrame(
        np.nan,
        index=models,
        columns=models,
        dtype=float,
    )
    formatted_matrix = pd.DataFrame(
        "",
        index=models,
        columns=models,
        dtype=object,
    )

    # Diagonal: not meaningful; keep numeric at 0.5 for color centering, but display as "--"
    for m in models:
        win_rate_matrix.loc[m, m] = 0.5
        formatted_matrix.loc[m, m] = "--"

    long_records: List[Dict[str, object]] = []
    for i, model_x in enumerate(models):
        for j, model_y in enumerate(models):
            if i == j:
                continue
            outcomes = _compute_pair_outcomes(
                pairwise_slice_df, model_x, model_y, alpha=alpha
            )
            long_records.append(outcomes)

            total = int(outcomes["total"])
            win_rate = outcomes["win_rate"]
            win_rate_matrix.loc[model_x, model_y] = win_rate if total > 0 else np.nan
            formatted_matrix.loc[model_x, model_y] = outcomes["formatted_win_rate"]

    long_df = pd.DataFrame(long_records)
    # Provide a stable ordering for downstream merges / exports
    if not long_df.empty:
        long_df = long_df.sort_values(
            ["persona", "prompt_type", "row_model", "col_model"]
        ).reset_index(drop=True)

    return JointPreferenceMatrix(
        persona=persona,
        prompt_type=prompt_type,
        models=models,
        win_rate_matrix=win_rate_matrix,
        formatted_matrix=formatted_matrix,
        pairwise_long=long_df,
    )


def _collect_models(pairwise_slice_df: pd.DataFrame) -> List[str]:
    """Collect a stable model ordering from a slice."""
    models_a = set(pairwise_slice_df["model_a_name"].dropna().unique().tolist())
    models_b = set(pairwise_slice_df["model_b_name"].dropna().unique().tolist())
    return sorted(models_a | models_b)


def _compute_pair_outcomes(
    pairwise_slice_df: pd.DataFrame,
    row_model: str,
    col_model: str,
    alpha: float,
) -> Dict[str, object]:
    """
    Compute wins/total/win-rate and p-value for (row_model vs col_model).

    Args:
        pairwise_slice_df: One persona + one prompt type slice.
        row_model: Row model (X).
        col_model: Column model (Y).
        alpha: Significance threshold.

    Returns:
        Dict[str, object]: Record with counts, win rate, p-value, and formatted string.
    """
    mask_forward_all = (pairwise_slice_df["model_a_name"] == row_model) & (
        pairwise_slice_df["model_b_name"] == col_model
    )
    mask_reverse_all = (pairwise_slice_df["model_a_name"] == col_model) & (
        pairwise_slice_df["model_b_name"] == row_model
    )
    subset = pairwise_slice_df[mask_forward_all | mask_reverse_all].copy()

    total = int(len(subset))
    wins = 0
    losses = 0
    ties = 0

    if total > 0:
        # Recompute masks within subset to avoid pandas reindex warnings.
        mask_forward = (subset["model_a_name"] == row_model) & (
            subset["model_b_name"] == col_model
        )
        mask_reverse = (subset["model_a_name"] == col_model) & (
            subset["model_b_name"] == row_model
        )
        forward = subset[mask_forward]
        reverse = subset[mask_reverse]

        # Forward rows: overall_winner_label refers to model_a/model_b in that row
        wins += int((forward["overall_winner_label"] == "model_a").sum())
        losses += int((forward["overall_winner_label"] == "model_b").sum())
        ties += int((forward["overall_winner_label"] == "tie").sum())

        # Reverse rows: row_model is model_b in that row
        wins += int((reverse["overall_winner_label"] == "model_b").sum())
        losses += int((reverse["overall_winner_label"] == "model_a").sum())
        ties += int((reverse["overall_winner_label"] == "tie").sum())

    win_rate = (wins / total) if total > 0 else np.nan
    n_excl_ties = wins + losses
    if total > 0:
        p_value = float(stats.binomtest(wins, total, p=0.5).pvalue)
    else:
        p_value = 1.0

    # Display convention: mark any statistically significant deviation from 0.5,
    # regardless of direction (so both 0.95* and 0.05* get a marker).
    # Direction is conveyed by the numeric value itself.
    significant = bool(p_value < alpha and n_excl_ties > 0 and wins != losses)
    formatted = _format_win_rate(win_rate, significant=significant, total=total)

    persona = str(pairwise_slice_df["user_id"].iloc[0])
    prompt_type = str(pairwise_slice_df["variant_label"].iloc[0])

    return {
        "persona": persona,
        "prompt_type": prompt_type,
        "row_model": row_model,
        "col_model": col_model,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "total": int(total),
        "tie_rate": (float(ties) / float(total)) if total > 0 else np.nan,
        "n_excl_ties": int(n_excl_ties),
        "win_rate": float(win_rate) if total > 0 else np.nan,
        "p_value": float(p_value),
        "significant": significant,
        "formatted_win_rate": formatted,
    }


def _format_win_rate(win_rate: float, significant: bool, total: int) -> str:
    """Format a win rate for tables/figures."""
    if (
        total <= 0
        or win_rate is None
        or (isinstance(win_rate, float) and np.isnan(win_rate))
    ):
        return ""
    base = f"{float(win_rate):.2f}"
    return f"{base}*" if significant else base


# ---------------------------------------------------------------------------
# Cluster-aware paired tests (Original vs Treatment vs Control)
# ---------------------------------------------------------------------------


class JointPreferencePairedTestError(RuntimeError):
    """Raised when paired joint preference tests encounter invalid or missing data."""


def format_paired_delta_mark(
    delta: object,
    *,
    p_value: object = None,
    q_value: object = None,
    alpha: float = 0.05,
) -> str:
    """
    Format a significance mark (↑/↓/•) for a paired delta test.

    Decision rule:
    - Use q_value when provided and numeric; otherwise fall back to p_value.
    - If (q or p) <= alpha:
      - return '↑' if delta > 0
      - return '↓' if delta < 0
      - return '•' if delta == 0 (significant but no direction)
    - Otherwise return ''.

    Args:
        delta (object): Effect size (mean Δ).
        p_value (object): Raw permutation p-value (fallback when q is absent).
        q_value (object): Adjusted q-value (preferred when present).
        alpha (float): Significance cutoff.

    Returns:
        str: '', '↑', '↓', or '•'.
    """
    try:
        a = float(alpha)
        if not (0.0 < a < 1.0):
            raise ValueError
    except Exception:
        raise JointPreferencePairedTestError(f"alpha must be in (0,1). Got: {alpha}")

    def _to_float(x: object) -> Optional[float]:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return None

    d = _to_float(delta)
    q = _to_float(q_value)
    p = _to_float(p_value)
    pv = q if q is not None else p
    if d is None or pv is None:
        return ""
    if float(pv) > a:
        return ""
    if float(d) > 0:
        return "↑"
    if float(d) < 0:
        return "↓"
    return "•"


def compute_cluster_aware_paired_tests_for_joint_preference(
    pairwise_sample_df: pd.DataFrame,
    *,
    treatment_label: str = "personalized",
    control_label: str = "control",
    original_label: str = "original",
    n_permutations: int = 10_000,
    n_bootstrap: int = 2_000,
    alpha: float = 0.05,
    seed: int = 42,
    equivalence_win_margin: float = 0.02,
    equivalence_score_margin: float = 0.05,
    two_sided: bool = True,
    apply_bh_fdr: bool = True,
    group_by_judge: bool = False,
    judge_column: str = "judge_model_name",
) -> pd.DataFrame:
    """
    Compute cluster-aware paired tests for joint preference outcomes.

    This implements paired, item-level tests that treat the base item (task_id)
    as the unit of independence. Treatment/control variants are aggregated within
    each base item, then compared via paired differences to the base/original.

    Endpoints (from row_model's perspective, for each sample):
    - Binary win indicator: W=1 if row_model wins else 0 (ties/losses are 0).
    - Ordinal win/tie/loss score: S=+1 win, 0 tie, -1 loss.

    For each ordered model pair (row_model vs col_model), and optionally per judge,
    we compute deltas per base item:
    - Δ_treat = mean(metric | treatment) - mean(metric | original)
    - Δ_ctrl  = mean(metric | control)   - mean(metric | original)

    For each delta vector we report:
    - effect size (mean Δ)
    - bootstrap CI for mean Δ
    - paired sign-flip permutation p-value
    - BH-FDR q-values (optional)
    - CI-based equivalence for control vs original (optional)

    Args:
        pairwise_sample_df (pd.DataFrame): Pairwise sample-level DataFrame. Must include:
            - user_id
            - task_id (base item id; should link original/treatment/control)
            - variant_label (condition)
            - variant_id (variant instance id; used only for within-item aggregation)
            - model_a_name, model_b_name
            - overall_winner_label in {'model_a','model_b','tie'}
            - judge_column when group_by_judge is True
        treatment_label (str): Condition label for treatment (default: 'personalized').
        control_label (str): Condition label for control (default: 'control').
        original_label (str): Condition label for original (default: 'original').
        n_permutations (int): Number of sign-flip permutations for p-values.
        n_bootstrap (int): Number of bootstrap resamples for CI.
        alpha (float): Significance level for CIs (CI level = 1 - alpha).
        seed (int): RNG seed for permutation/bootstrap reproducibility.
        equivalence_win_margin (float): ±ε margin for win-rate delta equivalence.
        equivalence_score_margin (float): ±ε margin for ordinal score delta equivalence.
        two_sided (bool): Whether permutation p-values are two-sided.
        apply_bh_fdr (bool): Whether to compute BH-FDR q-values for permutation p-values.
        group_by_judge (bool): If True, compute stats separately per judge model.
        judge_column (str): Judge column name in the input data (default: 'judge_model_name').

    Returns:
        pd.DataFrame: One row per (persona, row_model, col_model[, judge_model_name])
        containing paired test results for treatment vs original and control vs original.

    Raises:
        JointPreferencePairedTestError: If required columns are missing or parameters invalid.
    """
    if pairwise_sample_df is None or pairwise_sample_df.empty:
        return pd.DataFrame()

    if not (0.0 < float(alpha) < 1.0):
        raise JointPreferencePairedTestError(f"alpha must be in (0,1). Got: {alpha}")
    if int(n_permutations) <= 0:
        raise JointPreferencePairedTestError(
            f"n_permutations must be > 0. Got: {n_permutations}"
        )
    if int(n_bootstrap) <= 0:
        raise JointPreferencePairedTestError(
            f"n_bootstrap must be > 0. Got: {n_bootstrap}"
        )
    if float(equivalence_win_margin) <= 0:
        raise JointPreferencePairedTestError(
            f"equivalence_win_margin must be > 0. Got: {equivalence_win_margin}"
        )
    if float(equivalence_score_margin) <= 0:
        raise JointPreferencePairedTestError(
            f"equivalence_score_margin must be > 0. Got: {equivalence_score_margin}"
        )

    required_cols = [
        "user_id",
        "task_id",
        "variant_label",
        "variant_id",
        "model_a_name",
        "model_b_name",
        "overall_winner_label",
    ]
    if group_by_judge:
        required_cols.append(judge_column)
    missing = [c for c in required_cols if c not in pairwise_sample_df.columns]
    if missing:
        raise JointPreferencePairedTestError(
            "pairwise_sample_df is missing required columns for paired tests: "
            + ", ".join(missing)
        )

    # Enforce non-null required identifiers. If these are missing, downstream grouping
    # can silently create "nan" string groups which is very difficult to debug.
    null_required = [c for c in required_cols if pairwise_sample_df[c].isna().any()]
    if null_required:
        counts = {c: int(pairwise_sample_df[c].isna().sum()) for c in null_required}
        raise JointPreferencePairedTestError(
            "pairwise_sample_df has nulls in required columns for paired tests. "
            f"Null counts: {counts}"
        )

    # Comment: Build a directional, row_model-centric outcome frame (W,S).
    directional = _build_directional_outcome_frame_for_joint_preference(
        pairwise_sample_df,
        judge_column=judge_column if group_by_judge else None,
    )
    if directional.empty:
        return pd.DataFrame()

    # Filter to the three conditions we can compare.
    conditions = {str(original_label), str(treatment_label), str(control_label)}
    directional = directional[
        directional["variant_label"].astype(str).isin(conditions)
    ].copy()
    if directional.empty:
        return pd.DataFrame()

    group_keys = ["user_id", "row_model", "col_model"]
    if group_by_judge:
        group_keys.append(judge_column)

    # Comment: Aggregate within each base item (task_id) and condition (variant_label).
    item_means = (
        directional.groupby(group_keys + ["task_id", "variant_label"], dropna=False)
        .agg(
            w_bar=("W", "mean"),
            s_bar=("S", "mean"),
            n_obs=("W", "size"),
        )
        .reset_index()
    )
    if item_means.empty:
        return pd.DataFrame()

    # Pivot to wide per base item so we can compute paired deltas.
    w_wide = item_means.pivot_table(
        index=group_keys + ["task_id"],
        columns="variant_label",
        values="w_bar",
        aggfunc="mean",
    )
    s_wide = item_means.pivot_table(
        index=group_keys + ["task_id"],
        columns="variant_label",
        values="s_bar",
        aggfunc="mean",
    )

    # Ensure the expected columns exist even when missing in data.
    for lbl in [original_label, treatment_label, control_label]:
        if lbl not in w_wide.columns:
            w_wide[lbl] = np.nan
        if lbl not in s_wide.columns:
            s_wide[lbl] = np.nan

    w_wide = w_wide.reset_index()
    s_wide = s_wide.reset_index()

    merged = w_wide.merge(s_wide, on=group_keys + ["task_id"], suffixes=("_w", "_s"))
    if merged.empty:
        return pd.DataFrame()

    ci_level = 1.0 - float(alpha)
    records: List[Dict[str, object]] = []
    for keys, grp in merged.groupby(group_keys, dropna=False):
        context = dict(zip(group_keys, keys if isinstance(keys, tuple) else [keys]))
        group_seed_base = _stable_int_seed(
            int(seed), [str(v) for v in context.values()]
        )

        def _extract_deltas(
            prefix: str,  # "_w" or "_s"
            left_label: str,
            right_label: str,
        ) -> Tuple[np.ndarray, Dict[str, int]]:
            left = grp.get(f"{left_label}{prefix}")
            right = grp.get(f"{right_label}{prefix}")
            if left is None or right is None:
                return np.array([], dtype=float), {
                    "n_base_items": 0,
                    "n_with_left": 0,
                    "n_with_right": 0,
                }
            left = pd.to_numeric(left, errors="coerce")
            right = pd.to_numeric(right, errors="coerce")
            n_with_left = int(left.notna().sum())
            n_with_right = int(right.notna().sum())
            paired_mask = left.notna() & right.notna()
            deltas = (left[paired_mask] - right[paired_mask]).astype(float).to_numpy()
            return deltas, {
                "n_base_items": int(paired_mask.sum()),
                "n_with_left": n_with_left,
                "n_with_right": n_with_right,
            }

        # Treatment vs Original
        deltas_w_treat, counts_w_treat = _extract_deltas(
            "_w", treatment_label, original_label
        )
        deltas_s_treat, counts_s_treat = _extract_deltas(
            "_s", treatment_label, original_label
        )

        # Control vs Original
        deltas_w_ctrl, counts_w_ctrl = _extract_deltas(
            "_w", control_label, original_label
        )
        deltas_s_ctrl, counts_s_ctrl = _extract_deltas(
            "_s", control_label, original_label
        )

        # Treatment vs Control
        deltas_w_treat_ctrl, counts_w_treat_ctrl = _extract_deltas(
            "_w", treatment_label, control_label
        )
        deltas_s_treat_ctrl, counts_s_treat_ctrl = _extract_deltas(
            "_s", treatment_label, control_label
        )

        # Compute test stats per delta vector.
        treat_win = _compute_paired_delta_stats(
            deltas_w_treat,
            rng=np.random.default_rng(
                _stable_int_seed(group_seed_base, ["treat", "win"])
            ),
            n_permutations=int(n_permutations),
            n_bootstrap=int(n_bootstrap),
            confidence_level=ci_level,
            two_sided=bool(two_sided),
        )
        treat_score = _compute_paired_delta_stats(
            deltas_s_treat,
            rng=np.random.default_rng(
                _stable_int_seed(group_seed_base, ["treat", "score"])
            ),
            n_permutations=int(n_permutations),
            n_bootstrap=int(n_bootstrap),
            confidence_level=ci_level,
            two_sided=bool(two_sided),
        )
        ctrl_win = _compute_paired_delta_stats(
            deltas_w_ctrl,
            rng=np.random.default_rng(
                _stable_int_seed(group_seed_base, ["ctrl", "win"])
            ),
            n_permutations=int(n_permutations),
            n_bootstrap=int(n_bootstrap),
            confidence_level=ci_level,
            two_sided=bool(two_sided),
        )
        ctrl_score = _compute_paired_delta_stats(
            deltas_s_ctrl,
            rng=np.random.default_rng(
                _stable_int_seed(group_seed_base, ["ctrl", "score"])
            ),
            n_permutations=int(n_permutations),
            n_bootstrap=int(n_bootstrap),
            confidence_level=ci_level,
            two_sided=bool(two_sided),
        )
        treat_ctrl_win = _compute_paired_delta_stats(
            deltas_w_treat_ctrl,
            rng=np.random.default_rng(
                _stable_int_seed(group_seed_base, ["treat_vs_ctrl", "win"])
            ),
            n_permutations=int(n_permutations),
            n_bootstrap=int(n_bootstrap),
            confidence_level=ci_level,
            two_sided=bool(two_sided),
        )
        treat_ctrl_score = _compute_paired_delta_stats(
            deltas_s_treat_ctrl,
            rng=np.random.default_rng(
                _stable_int_seed(group_seed_base, ["treat_vs_ctrl", "score"])
            ),
            n_permutations=int(n_permutations),
            n_bootstrap=int(n_bootstrap),
            confidence_level=ci_level,
            two_sided=bool(two_sided),
        )

        record: Dict[str, object] = {**context}

        # Treatment vs Original: win-rate endpoint
        record.update(
            {
                "paired_treat_vs_orig_n_base_items": counts_w_treat["n_base_items"],
                "paired_treat_vs_orig_n_with_treatment": counts_w_treat["n_with_left"],
                "paired_treat_vs_orig_n_with_original": counts_w_treat["n_with_right"],
                "paired_treat_vs_orig_delta_win_mean": treat_win["mean_delta"],
                "paired_treat_vs_orig_delta_win_ci_lower": treat_win["ci_lower"],
                "paired_treat_vs_orig_delta_win_ci_upper": treat_win["ci_upper"],
                "paired_treat_vs_orig_delta_win_p_value": treat_win["p_value"],
            }
        )
        # Treatment vs Original: ordinal endpoint
        record.update(
            {
                "paired_treat_vs_orig_delta_score_mean": treat_score["mean_delta"],
                "paired_treat_vs_orig_delta_score_ci_lower": treat_score["ci_lower"],
                "paired_treat_vs_orig_delta_score_ci_upper": treat_score["ci_upper"],
                "paired_treat_vs_orig_delta_score_p_value": treat_score["p_value"],
            }
        )

        # Control vs Original: win-rate endpoint
        record.update(
            {
                "paired_ctrl_vs_orig_n_base_items": counts_w_ctrl["n_base_items"],
                "paired_ctrl_vs_orig_n_with_control": counts_w_ctrl["n_with_left"],
                "paired_ctrl_vs_orig_n_with_original": counts_w_ctrl["n_with_right"],
                "paired_ctrl_vs_orig_delta_win_mean": ctrl_win["mean_delta"],
                "paired_ctrl_vs_orig_delta_win_ci_lower": ctrl_win["ci_lower"],
                "paired_ctrl_vs_orig_delta_win_ci_upper": ctrl_win["ci_upper"],
                "paired_ctrl_vs_orig_delta_win_p_value": ctrl_win["p_value"],
                "paired_ctrl_vs_orig_delta_win_equiv_margin": float(
                    equivalence_win_margin
                ),
                "paired_ctrl_vs_orig_delta_win_equivalent": _equivalence_from_ci(
                    ci_lower=ctrl_win["ci_lower"],
                    ci_upper=ctrl_win["ci_upper"],
                    margin=float(equivalence_win_margin),
                ),
            }
        )
        # Control vs Original: ordinal endpoint
        record.update(
            {
                "paired_ctrl_vs_orig_delta_score_mean": ctrl_score["mean_delta"],
                "paired_ctrl_vs_orig_delta_score_ci_lower": ctrl_score["ci_lower"],
                "paired_ctrl_vs_orig_delta_score_ci_upper": ctrl_score["ci_upper"],
                "paired_ctrl_vs_orig_delta_score_p_value": ctrl_score["p_value"],
                "paired_ctrl_vs_orig_delta_score_equiv_margin": float(
                    equivalence_score_margin
                ),
                "paired_ctrl_vs_orig_delta_score_equivalent": _equivalence_from_ci(
                    ci_lower=ctrl_score["ci_lower"],
                    ci_upper=ctrl_score["ci_upper"],
                    margin=float(equivalence_score_margin),
                ),
            }
        )

        # Treatment vs Control: win-rate endpoint
        record.update(
            {
                "paired_treat_vs_ctrl_n_base_items": counts_w_treat_ctrl[
                    "n_base_items"
                ],
                "paired_treat_vs_ctrl_n_with_treatment": counts_w_treat_ctrl[
                    "n_with_left"
                ],
                "paired_treat_vs_ctrl_n_with_control": counts_w_treat_ctrl[
                    "n_with_right"
                ],
                "paired_treat_vs_ctrl_delta_win_mean": treat_ctrl_win["mean_delta"],
                "paired_treat_vs_ctrl_delta_win_ci_lower": treat_ctrl_win["ci_lower"],
                "paired_treat_vs_ctrl_delta_win_ci_upper": treat_ctrl_win["ci_upper"],
                "paired_treat_vs_ctrl_delta_win_p_value": treat_ctrl_win["p_value"],
            }
        )
        # Treatment vs Control: ordinal endpoint
        record.update(
            {
                "paired_treat_vs_ctrl_delta_score_mean": treat_ctrl_score["mean_delta"],
                "paired_treat_vs_ctrl_delta_score_ci_lower": treat_ctrl_score[
                    "ci_lower"
                ],
                "paired_treat_vs_ctrl_delta_score_ci_upper": treat_ctrl_score[
                    "ci_upper"
                ],
                "paired_treat_vs_ctrl_delta_score_p_value": treat_ctrl_score["p_value"],
            }
        )

        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        return out

    # Add BH-FDR q-values for each p-value column, grouped by persona and (optional) judge.
    if apply_bh_fdr:
        q_groups = ["user_id"]
        if group_by_judge:
            q_groups.append(judge_column)
        for p_col, q_col, tag in [
            (
                "paired_treat_vs_orig_delta_win_p_value",
                "paired_treat_vs_orig_delta_win_q_value",
                "treat_win",
            ),
            (
                "paired_treat_vs_orig_delta_score_p_value",
                "paired_treat_vs_orig_delta_score_q_value",
                "treat_score",
            ),
            (
                "paired_ctrl_vs_orig_delta_win_p_value",
                "paired_ctrl_vs_orig_delta_win_q_value",
                "ctrl_win",
            ),
            (
                "paired_ctrl_vs_orig_delta_score_p_value",
                "paired_ctrl_vs_orig_delta_score_q_value",
                "ctrl_score",
            ),
            (
                "paired_treat_vs_ctrl_delta_win_p_value",
                "paired_treat_vs_ctrl_delta_win_q_value",
                "treat_vs_ctrl_win",
            ),
            (
                "paired_treat_vs_ctrl_delta_score_p_value",
                "paired_treat_vs_ctrl_delta_score_q_value",
                "treat_vs_ctrl_score",
            ),
        ]:
            out[q_col] = np.nan
            for g_keys, g_df in out.groupby(q_groups, dropna=False):
                idx = g_df.index
                pvals = pd.to_numeric(g_df[p_col], errors="coerce").to_numpy()
                qvals = _bh_fdr_qvalues(pvals)
                out.loc[idx, q_col] = qvals
            LOGGER.info(
                "Computed BH-FDR q-values for '%s' grouped by %s (%s rows)",
                tag,
                q_groups,
                len(out),
            )

    # Rename user_id to persona for convenient joins to joint_preference long tables.
    out = out.rename(columns={"user_id": "persona"})
    if group_by_judge and judge_column != "judge_model_name":
        out = out.rename(columns={judge_column: "judge_model_name"})
    elif group_by_judge and judge_column == "judge_model_name":
        # Ensure stable naming for downstream joins.
        pass
    return out


def _build_directional_outcome_frame_for_joint_preference(
    pairwise_sample_df: pd.DataFrame,
    *,
    judge_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a directional outcome frame with W/S outcomes from row_model's perspective.

    Each input comparison row contributes two rows:
    - (row_model=model_a, col_model=model_b) with outcome from model_a's perspective
    - (row_model=model_b, col_model=model_a) with outcome from model_b's perspective

    Args:
        pairwise_sample_df (pd.DataFrame): Pairwise sample DataFrame.
        judge_column (Optional[str]): Judge column to carry through when not None.

    Returns:
        pd.DataFrame: Directional outcome frame with columns:
            user_id, task_id, variant_label, variant_id, row_model, col_model, W, S
            plus judge_column when provided.

    Raises:
        JointPreferencePairedTestError: If `overall_winner_label` has unexpected values.
    """
    df = pairwise_sample_df.copy()

    winner = df["overall_winner_label"].astype(str)
    valid = {"model_a", "model_b", "tie"}
    bad = sorted(set(winner.unique().tolist()) - valid)
    if bad:
        raise JointPreferencePairedTestError(
            "overall_winner_label contains unexpected values: " + ", ".join(bad)
        )

    base_cols = ["user_id", "task_id", "variant_label", "variant_id"]
    if judge_column:
        base_cols.append(judge_column)

    forward = df[
        base_cols + ["model_a_name", "model_b_name", "overall_winner_label"]
    ].copy()
    forward = forward.rename(
        columns={"model_a_name": "row_model", "model_b_name": "col_model"}
    )
    forward_winner = forward["overall_winner_label"].astype(str)
    forward["W"] = (forward_winner == "model_a").astype(float)
    forward["S"] = np.where(
        forward_winner == "model_a",
        1.0,
        np.where(forward_winner == "model_b", -1.0, 0.0),
    )

    reverse = df[
        base_cols + ["model_a_name", "model_b_name", "overall_winner_label"]
    ].copy()
    reverse = reverse.rename(
        columns={"model_b_name": "row_model", "model_a_name": "col_model"}
    )
    reverse_winner = reverse["overall_winner_label"].astype(str)
    reverse["W"] = (reverse_winner == "model_b").astype(float)
    reverse["S"] = np.where(
        reverse_winner == "model_b",
        1.0,
        np.where(reverse_winner == "model_a", -1.0, 0.0),
    )

    out = pd.concat([forward, reverse], ignore_index=True)
    # Normalize identifier typing for stable joins/exports.
    for col in [
        "user_id",
        "task_id",
        "variant_label",
        "variant_id",
        "row_model",
        "col_model",
    ]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if judge_column and judge_column in out.columns:
        out[judge_column] = out[judge_column].astype(str)
    return out


def _compute_paired_delta_stats(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
    n_bootstrap: int,
    confidence_level: float,
    two_sided: bool,
) -> Dict[str, float]:
    """
    Compute effect size, CI, and permutation p-value for paired deltas.

    Args:
        deltas (np.ndarray): Vector of per-base-item paired differences.
        rng (np.random.Generator): Random generator (seeded upstream).
        n_permutations (int): Number of sign-flip permutations.
        n_bootstrap (int): Number of bootstrap resamples.
        confidence_level (float): Confidence level for the bootstrap CI.
        two_sided (bool): Whether to compute a two-sided p-value.

    Returns:
        Dict[str, float]: mean_delta, ci_lower, ci_upper, p_value (NaN when undefined).
    """
    if deltas is None or len(deltas) == 0:
        return {
            "mean_delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
        }

    x = np.asarray(deltas, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {
            "mean_delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
        }

    mean_delta = float(np.mean(x))
    ci_lower, ci_upper = _bootstrap_mean_ci(
        x,
        rng=rng,
        n_bootstrap=int(n_bootstrap),
        confidence_level=float(confidence_level),
    )
    p_value = _paired_sign_flip_permutation_p_value(
        x,
        rng=rng,
        n_permutations=int(n_permutations),
        two_sided=bool(two_sided),
    )
    return {
        "mean_delta": float(mean_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
    }


def _paired_sign_flip_permutation_p_value(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
    two_sided: bool,
    chunk_size: int = 2_000,
) -> float:
    """
    Paired sign-flip permutation test for mean(delta)=0.

    Args:
        deltas (np.ndarray): Paired differences at the base-item level.
        rng (np.random.Generator): RNG for sign draws.
        n_permutations (int): Number of permutations.
        two_sided (bool): If True, uses |mean| for a two-sided p-value.
        chunk_size (int): Permutations per chunk (memory guard).

    Returns:
        float: Permutation p-value.
    """
    x = np.asarray(deltas, dtype=float)
    x = x[~np.isnan(x)]
    n = int(len(x))
    if n <= 0:
        return float("nan")
    if np.allclose(x, 0.0):
        return 1.0

    obs = float(np.mean(x))
    obs_stat = abs(obs) if two_sided else obs

    more_extreme = 0
    done = 0
    while done < int(n_permutations):
        take = min(int(chunk_size), int(n_permutations) - done)
        # Draw ±1 signs.
        signs = rng.integers(0, 2, size=(take, n)).astype(float)
        signs = signs * 2.0 - 1.0
        perm_means = (signs * x.reshape(1, -1)).mean(axis=1)
        if two_sided:
            perm_stats = np.abs(perm_means)
            more_extreme += int((perm_stats >= obs_stat).sum())
        else:
            # One-sided: consider direction of observed mean.
            if obs >= 0:
                more_extreme += int((perm_means >= obs_stat).sum())
            else:
                more_extreme += int((perm_means <= obs_stat).sum())
        done += take

    # Add-one smoothing for exactness.
    return float((more_extreme + 1) / (int(n_permutations) + 1))


def _bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence_level: float,
) -> Tuple[float, float]:
    """
    Bootstrap percentile CI for the mean of deltas.

    Args:
        deltas (np.ndarray): Paired differences.
        rng (np.random.Generator): RNG for resampling.
        n_bootstrap (int): Number of bootstrap resamples.
        confidence_level (float): Confidence level (e.g. 0.95).

    Returns:
        Tuple[float, float]: (ci_lower, ci_upper)
    """
    x = np.asarray(deltas, dtype=float)
    x = x[~np.isnan(x)]
    n = int(len(x))
    if n <= 0:
        return (float("nan"), float("nan"))
    if n == 1:
        v = float(x[0])
        return (v, v)

    means = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        means[i] = float(np.mean(x[idx]))

    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def _equivalence_from_ci(
    ci_lower: float, ci_upper: float, margin: float
) -> Optional[bool]:
    """
    CI-based equivalence decision for a mean delta.

    Args:
        ci_lower (float): CI lower bound.
        ci_upper (float): CI upper bound.
        margin (float): Equivalence margin ±ε.

    Returns:
        Optional[bool]: True/False when CI is defined, otherwise None.
    """
    try:
        if ci_lower is None or ci_upper is None:
            return None
        if np.isnan(float(ci_lower)) or np.isnan(float(ci_upper)):
            return None
        m = float(margin)
        return bool(float(ci_lower) >= -m and float(ci_upper) <= m)
    except Exception:
        return None


def _bh_fdr_qvalues(p_values: np.ndarray) -> np.ndarray:
    """
    Compute Benjamini-Hochberg FDR-adjusted q-values.

    Args:
        p_values (np.ndarray): Array of raw p-values (NaNs are ignored and preserved).

    Returns:
        np.ndarray: q-values aligned with p_values.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    if not mask.any():
        return out

    pv = p[mask]
    m = int(len(pv))
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * (m / (np.arange(1, m + 1, dtype=float)))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    # Reassign to original positions.
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    out_idx = np.where(mask)[0]
    out[out_idx] = q[inv]
    return out


def _stable_int_seed(seed: int, parts: List[str]) -> int:
    """
    Produce a deterministic 32-bit seed from an integer seed and string parts.

    Args:
        seed (int): Base seed.
        parts (List[str]): Components identifying a deterministic sub-stream.

    Returns:
        int: A non-negative integer suitable for `np.random.default_rng`.
    """
    payload = ("|".join([str(seed)] + [str(p) for p in parts])).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)
