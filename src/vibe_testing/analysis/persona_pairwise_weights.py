"""
Persona-driven pairwise dimension weights.

This module loads persona YAML configs (e.g. `configs/user_profiles/*.yaml`) and
derives per-persona weights over the canonical Stage-6 pairwise dimensions.

The mapping and semantics are intended to match Stage 6:
- Weights come from numeric fields under persona YAML `output_dimensions`.
- Unmapped pairwise dimensions default to a neutral weight of 1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Union

from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS
from src.vibe_testing.utils import load_config

logger = logging.getLogger(__name__)


def load_persona_pairwise_dimension_weights(
    config_paths: Iterable[Union[str, Path]],
    *,
    log: logging.Logger | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load per-persona weights over pairwise dimensions from persona YAML configs.

    Weight source:
        Numeric fields under persona YAML `output_dimensions`.

    Mapping (matches Stage 6):
        - clarity_and_comprehensibility -> clarity
        - workflow_fit -> workflow_fit
        - friction_and_frustration -> friction_loss_of_control
        - reliability -> reliability_user_trust
        - anthropomorphism -> anthropomorphism
        - persona_consistency_and_context_awareness -> persona_consistency + context_awareness

    For pairwise dimensions that are not mapped from the YAML, a neutral default
    weight of 1.0 is used.

    Args:
        config_paths: Iterable of persona YAML config paths.
        log: Optional logger to emit a summary of loaded weights.

    Returns:
        Dict[str, Dict[str, float]]: Mapping user_id -> {pairwise_dim -> weight}.

    Raises:
        ValueError: If configs are missing required fields, have non-positive weights,
            or contain duplicate user_id values.
    """

    mapping: Mapping[str, List[str]] = {
        "clarity_and_comprehensibility": ["clarity"],
        "workflow_fit": ["workflow_fit"],
        "friction_and_frustration": ["friction_loss_of_control"],
        "reliability": ["reliability_user_trust"],
        "anthropomorphism": ["anthropomorphism"],
        "persona_consistency_and_context_awareness": [
            "persona_consistency",
            "context_awareness",
        ],
        "plus_correctness": ["plus_correctness"],
        "correctness": ["correctness"],
    }

    paths = [Path(p).expanduser().resolve() for p in config_paths]
    if not paths:
        raise ValueError("No persona YAML config paths provided for weight loading.")

    out: Dict[str, Dict[str, float]] = {}
    for path in paths:
        if not path.exists() or not path.is_file():
            raise ValueError(f"Persona config path is not a file: {str(path)!r}")

        cfg = load_config(str(path))
        user_id = cfg.get("user_id")
        if not user_id or not str(user_id).strip():
            raise ValueError(f"Persona config is missing 'user_id': {str(path)!r}")
        user_key = str(user_id).strip()

        output_dimensions = cfg.get("output_dimensions", {})
        if not isinstance(output_dimensions, dict) or not output_dimensions:
            raise ValueError(
                "Persona config is missing a non-empty 'output_dimensions' dict: "
                f"{str(path)!r}"
            )

        weights: Dict[str, float] = {dim: 1.0 for dim in PAIRWISE_DIMENSIONS}
        for yaml_key, dims in mapping.items():
            value = output_dimensions.get(yaml_key)
            if isinstance(value, (int, float)):
                for dim in dims:
                    weights[str(dim)] = float(value)

        bad = {k: v for k, v in weights.items() if float(v) <= 0}
        if bad:
            raise ValueError(
                f"Non-positive dimension weights for user_id='{user_key}' "
                f"from '{str(path)}': {bad}"
            )

        if user_key in out:
            raise ValueError(
                f"Duplicate user_id '{user_key}' encountered in persona weight configs."
            )
        out[user_key] = weights

    use_logger = log or logger
    use_logger.info(
        "Loaded pairwise dimension weights for %d persona(s): %s",
        len(out),
        ", ".join(sorted(out.keys())),
    )
    return out
