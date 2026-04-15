"""Helpers for omitting specific vibe dimensions from Stage 6 analysis outputs.

This module centralizes the normalization logic used by the Stage 6 script to:
- Drop subjective per-dimension columns (and optionally strip dimension keys from
  subjective_metadata dict payloads)
- Drop pairwise per-dimension columns (dim_<dimension>_*) from Stage 5B frames
- Incorporate pass@1 correctness into per-sample pairwise winner determination

Note: This is a *presentation/output* filter by default. Stage 6 can optionally
recompute pairwise overall winners after applying dimension omissions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS

logger = logging.getLogger(__name__)

# Valid correctness handling modes for pairwise winner determination.
CORRECTNESS_MODES = ("ignore", "dimension", "gate")

# Fixed weight assigned to the base-correctness virtual dimension when
# dimension-weighted winner recomputation is active.
CORRECTNESS_DIMENSION_WEIGHT = 5.0

# Canonical pairwise dimension keys
PAIRWISE_CANONICAL = {
    "clarity",
    "tone_style_fit",
    "workflow_fit",
    "cognitive_load",
    "context_awareness",
    "persona_consistency",
    "friction_loss_of_control",
    "reliability_user_trust",
    "anthropomorphism",
}

# Common aliases used elsewhere in the codebase / legacy naming
ALIASES = {
    "frustration": "friction_loss_of_control",
    "friction": "friction_loss_of_control",
    "efficiency": "workflow_fit",
}


def normalize_omit_dimensions(dimensions: Iterable[str]) -> Tuple[set[str], set[str]]:
    """
    Normalize user-supplied tokens into (pairwise_keys, subjective_col_keys).

    Args:
        dimensions: Iterable of dimension tokens to omit (e.g. 'frustration',
            'reliability_user_trust', 'subj_reliability_user_trust').

    Returns:
        Tuple[set[str], set[str]]:
            - pairwise dimension keys (canonical, e.g. 'reliability_user_trust')
            - subjective column keys (e.g. 'subj_reliability_user_trust')
    """
    pairwise_omits: set[str] = set()
    subjective_omits: set[str] = set()

    for raw in dimensions:
        if raw is None:
            continue
        token = str(raw).strip().lower().replace("-", "_")
        if not token:
            continue

        if token.startswith("subj_"):
            subjective_omits.add(token)
            base = token.replace("subj_", "", 1)
            base = ALIASES.get(base, base)
            if base in PAIRWISE_CANONICAL:
                pairwise_omits.add(base)
            continue

        token = ALIASES.get(token, token)
        if token in PAIRWISE_CANONICAL:
            pairwise_omits.add(token)
            subjective_omits.add(f"subj_{token}")

    return pairwise_omits, subjective_omits


def apply_subjective_dimension_omits(
    subjective_df: pd.DataFrame,
    omit_pairwise_keys: set[str],
    omit_subjective_cols: set[str],
    *,
    strip_subjective_metadata: bool = True,
) -> pd.DataFrame:
    """
    Drop subjective per-dimension columns and optionally strip keys from subjective_metadata.

    Args:
        subjective_df: Frame returned by AnalysisInputLoader.load_subjective_results().
        omit_pairwise_keys: Canonical dimension keys to omit (e.g. 'reliability_user_trust').
        omit_subjective_cols: Subjective column keys to omit (e.g. 'subj_reliability_user_trust').
        strip_subjective_metadata: If True, removes the omitted dimension keys from the
            'subjective_metadata' dict column (when present).

    Returns:
        New DataFrame with omissions applied.
    """
    if subjective_df is None or subjective_df.empty:
        return subjective_df

    df = subjective_df.copy()

    # Drop raw subjective dimension columns (subj_*)
    if omit_subjective_cols:
        drop_cols = [c for c in df.columns if c in omit_subjective_cols]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    # Also drop already-aggregated forms if present (e.g. subj_*_mean)
    if omit_subjective_cols:
        mean_cols = [f"{c}_mean" for c in omit_subjective_cols]
        drop_mean_cols = [c for c in df.columns if c in mean_cols]
        if drop_mean_cols:
            df = df.drop(columns=drop_mean_cols)

    if (
        strip_subjective_metadata
        and "subjective_metadata" in df.columns
        and omit_pairwise_keys
    ):
        df["subjective_metadata"] = df["subjective_metadata"].apply(
            lambda meta: _strip_dimension_keys_from_metadata(meta, omit_pairwise_keys)
        )

    return df


def apply_pairwise_dimension_omits(
    pairwise_df: pd.DataFrame,
    omit_pairwise_keys: set[str],
) -> pd.DataFrame:
    """
    Drop pairwise per-dimension columns (dim_<dimension>_*) for omitted dimensions.

    Args:
        pairwise_df: Frame returned by AnalysisInputLoader.load_pairwise_results().
        omit_pairwise_keys: Canonical dimension keys to omit.

    Returns:
        New DataFrame with per-dimension columns removed.
    """
    if pairwise_df is None or pairwise_df.empty or not omit_pairwise_keys:
        return pairwise_df

    df = pairwise_df.copy()
    prefixes = tuple(f"dim_{d}_" for d in omit_pairwise_keys)
    drop_cols = [c for c in df.columns if c.startswith(prefixes)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


ObjectiveLookup = Dict[Tuple[str, str, str], Dict[str, Optional[float]]]


def _row_sample_key(row: pd.Series) -> Optional[str]:
    """
    Return a stable per-sample key for a pairwise row.

    Prefer (task_id, variant_id) when available; otherwise use task_id.
    Mirrors ``pairwise._row_sample_key``.

    Args:
        row: A pandas Series representing a single pairwise comparison row.

    Returns:
        Optional[str]: A string key for objective lookup, or None.
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

    raw_task_id = _clean(row.get("raw_task_id"))
    t = _clean(row.get("task_id"))
    v = _clean(row.get("variant_id"))

    if raw_task_id and raw_task_id != t:
        return raw_task_id
    if t and v and v != t:
        return f"{t}::{v}"
    return t


def _lookup_sample_correctness(
    row: pd.Series,
    objective_lookup: ObjectiveLookup,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Look up base and plus pass@1 values for both models in a pairwise row.

    Args:
        row: A pandas Series with at least task_id, variant_label,
            model_a_name, model_b_name.
        objective_lookup: Pre-built objective lookup from
            ``pairwise._build_objective_lookup``.

    Returns:
        Tuple of (base_pass_at_1_a, base_pass_at_1_b,
                  plus_pass_at_1_a, plus_pass_at_1_b).
        Any value may be None when data is unavailable.
    """
    sample_key = _row_sample_key(row)
    if sample_key is None:
        return None, None, None, None

    variant_label = row.get("variant_label")
    if variant_label is None or (hasattr(pd, "isna") and pd.isna(variant_label)):
        variant_label = "original"

    model_a = row.get("model_a_name")
    model_b = row.get("model_b_name")
    if not model_a or not model_b:
        return None, None, None, None

    metrics_a = objective_lookup.get((sample_key, variant_label, model_a), {})
    metrics_b = objective_lookup.get((sample_key, variant_label, model_b), {})

    base_a = metrics_a.get("pass_at_1")
    base_b = metrics_b.get("pass_at_1")
    plus_a = metrics_a.get("plus_pass_at_1")
    plus_b = metrics_b.get("plus_pass_at_1")

    return base_a, base_b, plus_a, plus_b


def _validate_correctness_mode(
    correctness_mode: str,
    objective_lookup: Optional[ObjectiveLookup],
) -> None:
    """
    Validate that the correctness mode and lookup are consistent.

    Args:
        correctness_mode: One of CORRECTNESS_MODES.
        objective_lookup: Objective lookup dict.

    Raises:
        ValueError: If the mode is unknown or requires missing data.
    """
    if correctness_mode not in CORRECTNESS_MODES:
        raise ValueError(
            f"Unknown correctness_mode={correctness_mode!r}; "
            f"must be one of {CORRECTNESS_MODES}."
        )
    if correctness_mode != "ignore" and (
        objective_lookup is None or not objective_lookup
    ):
        raise ValueError(
            f"correctness_mode={correctness_mode!r} requires a non-empty "
            "objective_lookup, but none was provided. Either set "
            "correctness_mode='ignore' or provide objective results."
        )


def recompute_pairwise_overall_winner(
    pairwise_df: pd.DataFrame,
    *,
    omit_pairwise_keys: set[str] | None = None,
    exclude_evaluation_error_dimensions: bool = True,
    correctness_mode: str = "ignore",
    objective_lookup: Optional[ObjectiveLookup] = None,
    include_plus_correctness: bool = False,
) -> pd.DataFrame:
    """
    Recompute overall pairwise winners after dimension omissions.

    This is used in Stage 6 analysis to ensure that:
    - Omitted dimensions do not influence per-sample overall outcomes.
    - Per-dimension evaluation errors do not contribute as "ties" when desired.
    - Correctness (pass@1) can optionally influence per-sample winners.

    This function operates purely on the Stage-6 DataFrame produced by
    ``AnalysisInputLoader.load_pairwise_results``.

    Args:
        pairwise_df: Pairwise sample-level DataFrame.
        omit_pairwise_keys: Canonical dimension keys to omit when recomputing.
        exclude_evaluation_error_dimensions: If True, exclude dimensions whose
            rationale indicates an evaluation/parsing error from the vote count.
        correctness_mode: How to incorporate pass@1 correctness.
            - ``"ignore"``: Do not use correctness (default, preserves legacy behaviour).
            - ``"dimension"``: Treat base pass@1 as an additional dimension vote.
            - ``"gate"``: If exactly one model is correct (pass@1 > 0), it wins
              automatically and dimension votes are skipped.
        objective_lookup: Pre-built objective lookup (required when
            ``correctness_mode != "ignore"``).
        include_plus_correctness: When True and ``correctness_mode == "dimension"``,
            add a second virtual dimension using plus-test pass@1.

    Returns:
        A new DataFrame with recomputed columns:
        - overall_winner
        - overall_winner_label
        - win_count_a / win_count_b / win_count_tie (based on included dims)
        - pairwise_valid_for_overall (False when no non-error dims are available)
        - pairwise_eval_error_dim_count (number of excluded error dimensions)
        - pairwise_correctness_data_available (per-row flag)

    Raises:
        ValueError: If correctness_mode is invalid or requires missing data.
    """
    if pairwise_df is None or pairwise_df.empty:
        return pairwise_df

    _validate_correctness_mode(correctness_mode, objective_lookup)

    omit_pairwise_keys = omit_pairwise_keys or set()
    included_dims = [d for d in PAIRWISE_DIMENSIONS if d not in omit_pairwise_keys]
    if not included_dims:
        raise ValueError(
            "No pairwise dimensions remain after omission; cannot recompute winners."
        )

    df = pairwise_df.copy()

    required_cols = {"model_a_name", "model_b_name"}
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            "pairwise_df missing required columns for recomputing overall winner: "
            + ", ".join(sorted(missing_required))
        )

    winner_cols = [f"dim_{d}_winner_label" for d in included_dims]
    rationale_cols = [f"dim_{d}_rationale" for d in included_dims]
    missing_dim_cols = [
        c for c in (winner_cols + rationale_cols) if c not in df.columns
    ]
    if missing_dim_cols:
        raise ValueError(
            "pairwise_df missing per-dimension columns required for recomputing "
            "overall winner: " + ", ".join(sorted(missing_dim_cols))
        )

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

    use_correctness = correctness_mode != "ignore"

    overall_labels: list[str] = []
    overall_winners: list[object] = []
    win_count_a: list[int] = []
    win_count_b: list[int] = []
    win_count_tie: list[int] = []
    valid_for_overall: list[bool] = []
    eval_error_dim_count: list[int] = []
    correctness_available: list[bool] = []

    for _, row in df.iterrows():
        model_a = row.get("model_a_name")
        model_b = row.get("model_b_name")

        base_a: Optional[float] = None
        base_b: Optional[float] = None
        plus_a: Optional[float] = None
        plus_b: Optional[float] = None
        has_correctness = False
        if use_correctness and objective_lookup:
            base_a, base_b, plus_a, plus_b = _lookup_sample_correctness(
                row, objective_lookup
            )
            has_correctness = base_a is not None and base_b is not None
        correctness_available.append(has_correctness)

        # Gate mode: if one model correct and the other not, auto-win
        if (
            correctness_mode == "gate"
            and has_correctness
            and base_a is not None
            and base_b is not None
        ):
            a_correct = base_a > 0
            b_correct = base_b > 0
            if a_correct and not b_correct:
                overall_winners.append(model_a)
                overall_labels.append("model_a")
                win_count_a.append(0)
                win_count_b.append(0)
                win_count_tie.append(0)
                valid_for_overall.append(True)
                eval_error_dim_count.append(0)
                continue
            if b_correct and not a_correct:
                overall_winners.append(model_b)
                overall_labels.append("model_b")
                win_count_a.append(0)
                win_count_b.append(0)
                win_count_tie.append(0)
                valid_for_overall.append(True)
                eval_error_dim_count.append(0)
                continue
            # Both correct or both incorrect: fall through to dimension comparison

        # Dimension-based comparison (standard or with correctness as dimension)
        a_wins = 0
        b_wins = 0
        ties = 0
        excluded_errors = 0

        for dim in included_dims:
            winner = row.get(f"dim_{dim}_winner_label", "tie")
            rationale = row.get(f"dim_{dim}_rationale", "")

            if exclude_evaluation_error_dimensions and _is_eval_error_rationale(
                rationale
            ):
                excluded_errors += 1
                continue

            if winner == "model_a":
                a_wins += 1
            elif winner == "model_b":
                b_wins += 1
            else:
                ties += 1

        # Add base correctness as a virtual dimension vote
        if correctness_mode == "dimension" and has_correctness:
            if base_a > base_b:
                a_wins += 1
            elif base_b > base_a:
                b_wins += 1
            else:
                ties += 1

            # Add plus correctness as another virtual dimension vote
            if include_plus_correctness and plus_a is not None and plus_b is not None:
                if plus_a > plus_b:
                    a_wins += 1
                elif plus_b > plus_a:
                    b_wins += 1
                else:
                    ties += 1

        total_dims = a_wins + b_wins + ties
        eval_error_dim_count.append(int(excluded_errors))
        valid_for_overall.append(bool(total_dims > 0))

        if total_dims <= 0:
            overall_winners.append(None)
            overall_labels.append("tie")
        elif a_wins > b_wins:
            overall_winners.append(model_a)
            overall_labels.append("model_a")
        elif b_wins > a_wins:
            overall_winners.append(model_b)
            overall_labels.append("model_b")
        else:
            overall_winners.append(None)
            overall_labels.append("tie")

        win_count_a.append(int(a_wins))
        win_count_b.append(int(b_wins))
        win_count_tie.append(int(ties))

    df["overall_winner"] = overall_winners
    df["overall_winner_label"] = overall_labels
    df["win_count_a"] = win_count_a
    df["win_count_b"] = win_count_b
    df["win_count_tie"] = win_count_tie
    df["pairwise_valid_for_overall"] = valid_for_overall
    df["pairwise_eval_error_dim_count"] = eval_error_dim_count
    df["pairwise_correctness_data_available"] = correctness_available
    return df


def recompute_pairwise_overall_winner_dimension_weighted(
    pairwise_df: pd.DataFrame,
    *,
    dimension_weights_by_user: Dict[str, Dict[str, float]],
    omit_pairwise_keys: set[str] | None = None,
    exclude_evaluation_error_dimensions: bool = True,
    correctness_mode: str = "ignore",
    objective_lookup: Optional[ObjectiveLookup] = None,
    include_plus_correctness: bool = False,
) -> pd.DataFrame:
    """
    Recompute overall pairwise winners using a dimension-weighted vote per persona.

    This is the Stage-6-only fix for computing per-sample overall winners in a way
    that reflects persona preferences. For each row, we sum the persona's weights
    over dimensions that vote for model_a/model_b/tie, then pick the max.

    Args:
        pairwise_df: Pairwise sample-level DataFrame.
        dimension_weights_by_user: Mapping user_id -> {pairwise_dim -> weight}.
        omit_pairwise_keys: Canonical pairwise dimension keys to omit (excluded from vote).
        exclude_evaluation_error_dimensions: If True, exclude dimensions whose rationale
            indicates an evaluation/parsing error from the vote.
        correctness_mode: How to incorporate pass@1 correctness.
            - ``"ignore"``: Do not use correctness (default).
            - ``"dimension"``: Treat base pass@1 as a weighted virtual dimension
              (weight = ``CORRECTNESS_DIMENSION_WEIGHT``).
            - ``"gate"``: If exactly one model is correct (pass@1 > 0), it wins
              automatically and dimension votes are skipped.
        objective_lookup: Pre-built objective lookup (required when
            ``correctness_mode != "ignore"``).
        include_plus_correctness: When True and ``correctness_mode == "dimension"``,
            add a second virtual dimension using plus-test pass@1 with weight
            equal to the persona's ``workflow_fit`` weight.

    Returns:
        A new DataFrame with recomputed columns:
        - overall_winner
        - overall_winner_label
        - pairwise_valid_for_overall (False when no usable dims remain)
        - pairwise_eval_error_dim_count (number of excluded error dimensions)
        - pairwise_weighted_score_a / _b / _tie (for transparency)
        - pairwise_correctness_data_available (per-row flag)

    Raises:
        ValueError: If required columns or persona weights are missing.
    """
    if pairwise_df is None or pairwise_df.empty:
        return pairwise_df

    _validate_correctness_mode(correctness_mode, objective_lookup)

    if not isinstance(dimension_weights_by_user, dict) or not dimension_weights_by_user:
        raise ValueError(
            "dimension_weights_by_user must be a non-empty mapping when using "
            "dimension-weighted overall winner recomputation."
        )

    omit_pairwise_keys = omit_pairwise_keys or set()
    included_dims = [d for d in PAIRWISE_DIMENSIONS if d not in omit_pairwise_keys]
    if not included_dims:
        raise ValueError(
            "No pairwise dimensions remain after omission; cannot recompute winners."
        )

    df = pairwise_df.copy()

    required_cols = {"model_a_name", "model_b_name", "user_id"}
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            "pairwise_df missing required columns for recomputing overall winner: "
            + ", ".join(sorted(missing_required))
        )

    winner_cols = [f"dim_{d}_winner_label" for d in included_dims]
    rationale_cols = [f"dim_{d}_rationale" for d in included_dims]
    missing_dim_cols = [
        c for c in (winner_cols + rationale_cols) if c not in df.columns
    ]
    if missing_dim_cols:
        raise ValueError(
            "pairwise_df missing per-dimension columns required for recomputing "
            "overall winner: " + ", ".join(sorted(missing_dim_cols))
        )

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

    use_correctness = correctness_mode != "ignore"

    overall_labels: list[str] = []
    overall_winners: list[object] = []
    valid_for_overall: list[bool] = []
    eval_error_dim_count: list[int] = []
    score_a_list: list[float] = []
    score_b_list: list[float] = []
    score_tie_list: list[float] = []
    correctness_available: list[bool] = []

    for _, row in df.iterrows():
        user_id = row.get("user_id")
        if user_id is None or (hasattr(pd, "isna") and pd.isna(user_id)):
            raise ValueError(
                "pairwise_df contains a null user_id; cannot apply weights."
            )
        user_key = str(user_id)
        if user_key not in dimension_weights_by_user:
            raise ValueError(
                f"Missing dimension weights for user_id='{user_key}'. "
                "Provide weights for every persona present in pairwise results."
            )

        dim_weights = dimension_weights_by_user[user_key]
        missing_weights = [d for d in included_dims if d not in dim_weights]
        if missing_weights:
            raise ValueError(
                f"Dimension weights for user_id='{user_key}' are missing keys: "
                + ", ".join(sorted(missing_weights))
            )

        model_a = row.get("model_a_name")
        model_b = row.get("model_b_name")

        # Look up correctness data
        base_a: Optional[float] = None
        base_b: Optional[float] = None
        plus_a: Optional[float] = None
        plus_b: Optional[float] = None
        has_correctness = False
        if use_correctness and objective_lookup:
            base_a, base_b, plus_a, plus_b = _lookup_sample_correctness(
                row, objective_lookup
            )
            has_correctness = base_a is not None and base_b is not None
        correctness_available.append(has_correctness)

        # Gate mode: if one model correct and the other not, auto-win
        if (
            correctness_mode == "gate"
            and has_correctness
            and base_a is not None
            and base_b is not None
        ):
            a_correct = base_a > 0
            b_correct = base_b > 0
            if a_correct and not b_correct:
                overall_winners.append(model_a)
                overall_labels.append("model_a")
                valid_for_overall.append(True)
                eval_error_dim_count.append(0)
                score_a_list.append(
                    float(dim_weights.get("correctness", CORRECTNESS_DIMENSION_WEIGHT))
                )
                score_b_list.append(0.0)
                score_tie_list.append(0.0)
                continue
            if b_correct and not a_correct:
                overall_winners.append(model_b)
                overall_labels.append("model_b")
                valid_for_overall.append(True)
                eval_error_dim_count.append(0)
                score_a_list.append(0.0)
                score_b_list.append(
                    float(dim_weights.get("correctness", CORRECTNESS_DIMENSION_WEIGHT))
                )
                score_tie_list.append(0.0)
                continue

        score_a = 0.0
        score_b = 0.0
        score_tie = 0.0
        excluded_errors = 0

        for dim in included_dims:
            rationale = row.get(f"dim_{dim}_rationale", "")
            if exclude_evaluation_error_dimensions and _is_eval_error_rationale(
                rationale
            ):
                excluded_errors += 1
                continue

            w = float(dim_weights[dim])
            winner = row.get(f"dim_{dim}_winner_label", "tie")
            if winner == "model_a":
                score_a += w
            elif winner == "model_b":
                score_b += w
            else:
                score_tie += w

        # Add base correctness as a weighted virtual dimension
        if correctness_mode == "dimension" and has_correctness:
            cw = float(dim_weights.get("correctness", CORRECTNESS_DIMENSION_WEIGHT))
            if base_a > base_b:
                score_a += cw
            elif base_b > base_a:
                score_b += cw
            else:
                score_tie += cw

            # Plus correctness uses the persona's plus_correctness weight
            if include_plus_correctness and plus_a is not None and plus_b is not None:
                plus_w = float(
                    dim_weights.get(
                        "plus_correctness",
                        dim_weights.get("workflow_fit", 1.0),
                    )
                )
                if plus_a > plus_b:
                    score_a += plus_w
                elif plus_b > plus_a:
                    score_b += plus_w
                else:
                    score_tie += plus_w

        total_weight = score_a + score_b + score_tie
        eval_error_dim_count.append(int(excluded_errors))
        valid_for_overall.append(bool(total_weight > 0))
        score_a_list.append(float(score_a))
        score_b_list.append(float(score_b))
        score_tie_list.append(float(score_tie))

        if total_weight <= 0:
            overall_winners.append(None)
            overall_labels.append("tie")
        elif score_a > score_b:
            overall_winners.append(model_a)
            overall_labels.append("model_a")
        elif score_b > score_a:
            overall_winners.append(model_b)
            overall_labels.append("model_b")
        else:
            overall_winners.append(None)
            overall_labels.append("tie")

    df["overall_winner"] = overall_winners
    df["overall_winner_label"] = overall_labels
    df["pairwise_valid_for_overall"] = valid_for_overall
    df["pairwise_eval_error_dim_count"] = eval_error_dim_count
    df["pairwise_weighted_score_a"] = score_a_list
    df["pairwise_weighted_score_b"] = score_b_list
    df["pairwise_weighted_score_tie"] = score_tie_list
    df["pairwise_correctness_data_available"] = correctness_available
    return df


def _strip_dimension_keys_from_metadata(meta: Any, omit_keys: set[str]) -> Any:
    """Remove omitted dimension keys from subjective_metadata if it is a dict."""
    if not isinstance(meta, dict):
        return meta
    cleaned: Dict[str, Any] = dict(meta)
    for k in list(cleaned.keys()):
        if str(k).strip().lower().replace("-", "_") in omit_keys:
            cleaned.pop(k, None)
    return cleaned
