"""Shared utilities for classifying and filtering human vs LLM judges.

Human judge directories are identified by prefix conventions established
by the human-annotation export pipeline.  This module provides a single
source of truth so that Stage 6 analysis, Streamlit UIs, and exporters
all use the same classification logic.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

HUMAN_JUDGE_PREFIXES: Tuple[str, ...] = (
    "judge_model_human_annotator",
    "judge_model_human_",
)
HUMAN_JUDGE_TOKEN_PREFIXES: Tuple[str, ...] = (
    "human_annotator",
    "human_",
)


def is_human_judge_token(judge_token: str) -> bool:
    """Return True when *judge_token* corresponds to a human annotator.

    The check is prefix-based: tokens starting with ``judge_model_human_``
    or ``human_`` (and their ``_annotator`` variants) are classified as human.

    Args:
        judge_token: Judge directory or column token.

    Returns:
        bool: True for human judges, False otherwise.
    """
    token = str(judge_token or "").strip()
    if not token:
        return False
    return bool(
        any(token.startswith(p) for p in HUMAN_JUDGE_PREFIXES)
        or any(token.startswith(p) for p in HUMAN_JUDGE_TOKEN_PREFIXES)
    )


def split_judges_by_group(
    judges: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split judge tokens into ``(human_judges, llm_judges)``.

    Args:
        judges: Sequence of judge tokens.

    Returns:
        Tuple of (human_judges, llm_judges) lists preserving input order.
    """
    human_judges: List[str] = []
    llm_judges: List[str] = []
    for judge in judges:
        token = str(judge)
        if is_human_judge_token(token):
            human_judges.append(token)
        else:
            llm_judges.append(token)
    return human_judges, llm_judges


def filter_human_judges_from_df(
    df: pd.DataFrame,
    judge_column: str = "judge_model_name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a pairwise DataFrame into LLM-only and human-only subsets.

    Args:
        df: Pairwise DataFrame containing a judge identifier column.
        judge_column: Name of the column holding judge tokens.

    Returns:
        Tuple of ``(llm_only_df, human_only_df)``.

    Raises:
        ValueError: If *judge_column* is not present in *df*.
    """
    if df is None or df.empty:
        empty = df if df is not None else pd.DataFrame()
        return empty.copy(), empty.copy()

    if judge_column not in df.columns:
        raise ValueError(
            f"Cannot filter human judges: column '{judge_column}' "
            f"not found in DataFrame. Available columns: "
            f"{sorted(df.columns.tolist())}"
        )

    human_mask = df[judge_column].astype(str).map(is_human_judge_token)
    llm_df = df[~human_mask].copy()
    human_df = df[human_mask].copy()
    return llm_df, human_df
