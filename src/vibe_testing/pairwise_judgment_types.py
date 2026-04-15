"""
Shared helpers for pairwise judgment-type handling.
"""

from __future__ import annotations

from typing import Optional

PAIRWISE_JUDGMENT_TYPE_PERSONA = "persona"
PAIRWISE_JUDGMENT_TYPE_GENERAL_USER = "general_user"

SUPPORTED_PAIRWISE_JUDGMENT_TYPES = (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    PAIRWISE_JUDGMENT_TYPE_GENERAL_USER,
)

SHARED_GENERAL_USER_PROMPT_TYPES = {"original", "control"}


def normalize_pairwise_judgment_type(value: Optional[str]) -> str:
    """
    Normalize a pairwise judgment type token.

    Args:
        value (Optional[str]): Raw judgment-type token.

    Returns:
        str: Normalized judgment type.

    Raises:
        ValueError: If the provided value is unsupported.
    """
    token = str(value or PAIRWISE_JUDGMENT_TYPE_PERSONA).strip().lower()
    if token not in SUPPORTED_PAIRWISE_JUDGMENT_TYPES:
        raise ValueError(
            "Unsupported pairwise judgment type "
            f"{value!r}. Expected one of: {SUPPORTED_PAIRWISE_JUDGMENT_TYPES}"
        )
    return token


def uses_shared_general_user_artifacts(
    judgment_type: Optional[str],
    prompt_type: Optional[str],
) -> bool:
    """
    Determine whether a pairwise run should reuse shared general-user artifacts.

    Args:
        judgment_type (Optional[str]): Pairwise judgment type.
        prompt_type (Optional[str]): Active prompt type.

    Returns:
        bool: True when the run should reuse the reference persona artifact.
    """
    normalized = normalize_pairwise_judgment_type(judgment_type)
    return (
        normalized == PAIRWISE_JUDGMENT_TYPE_GENERAL_USER
        and str(prompt_type or "").strip().lower() in SHARED_GENERAL_USER_PROMPT_TYPES
    )
