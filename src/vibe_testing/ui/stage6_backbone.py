"""
Stage-6-style backbone helpers for the Streamlit pairwise explorer.

This module provides small, testable helpers used by the Streamlit app to
construct "paper-ready" (Stage 6 semantics) outcome tables from Stage-6-
normalized pairwise rows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from src.vibe_testing.ui.pairwise_explorer_io import (
    stage6_display_winner_name_from_row,
)

logger = logging.getLogger(__name__)


def build_stage6_outcomes_by_judge_token(
    stage6_df: pd.DataFrame,
    *,
    judge_token_col: str = "_judge_token",
    sample_id_col: str = "_stage6_sample_id",
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    """
    Build judge-token keyed outcomes and per-sample metadata from Stage-6 rows.

    Args:
        stage6_df: Stage-6-normalized pairwise rows (as a DataFrame).
        judge_token_col: Column containing the judge directory token used by the UI.
        sample_id_col: Column containing the stable per-sample id (Stage 6 convention).

    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
            - outcomes_by_judge_token: judge_token -> sample_id -> winner_name_or_tie
            - meta_by_sample_id: sample_id -> metadata fields (raw_task_id/task_id/variant_id/variant_label)

    Raises:
        ValueError: If required columns are missing or if conflicting outcomes are detected.
    """

    if stage6_df is None or stage6_df.empty:
        return {}, {}

    required = {judge_token_col, sample_id_col, "model_a_name", "model_b_name"}
    missing = [c for c in sorted(required) if c not in stage6_df.columns]
    if missing:
        raise ValueError(f"Stage-6 backbone is missing required columns: {missing}")

    outcomes_by_judge: Dict[str, Dict[str, str]] = {}
    meta_by_sample_id: Dict[str, Dict[str, Any]] = {}

    for _, r in stage6_df.iterrows():
        row = r.to_dict()
        judge_token = str(row.get(judge_token_col, "") or "")
        sample_id = str(row.get(sample_id_col, "") or "")
        if not judge_token:
            raise ValueError(
                f"Missing judge token column {judge_token_col!r} in Stage-6 row."
            )
        if not sample_id:
            raise ValueError(
                f"Missing sample id column {sample_id_col!r} in Stage-6 row."
            )

        winner = stage6_display_winner_name_from_row(row)
        existing = outcomes_by_judge.setdefault(judge_token, {}).get(sample_id)
        if existing is not None and str(existing) != str(winner):
            raise ValueError(
                "Conflicting outcomes detected for the same (sample_id, judge_token). "
                f"sample_id={sample_id!r} judge_token={judge_token!r} outcomes={[existing, winner]!r}"
            )
        outcomes_by_judge[judge_token][sample_id] = str(winner)

        if sample_id not in meta_by_sample_id:
            meta_by_sample_id[sample_id] = {
                "raw_task_id": str(row.get("raw_task_id", "") or ""),
                "base_task_id": str(row.get("task_id", "") or ""),
                "variant_id": str(row.get("variant_id", "") or ""),
                "variant_label": str(row.get("variant_label", "") or ""),
            }

    return outcomes_by_judge, meta_by_sample_id
