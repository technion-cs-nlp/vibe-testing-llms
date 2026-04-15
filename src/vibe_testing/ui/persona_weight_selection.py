"""
Persistence for Streamlit persona-weight config selections.

Streamlit’s pairwise explorer supports recomputing overall winners using
persona-driven dimension weights derived from YAML configs under
`configs/user_profiles/*.yaml`. This module persists which YAML files should be
used (per run directory) so results are reproducible across app sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


SELECTIONS_VERSION = 1


@dataclass(frozen=True)
class PersonaWeightConfigSelection:
    """
    Selected persona YAML config paths used for dimension-weighted recomputation.

    Attributes:
        config_paths: Absolute paths to persona YAML files.
    """

    config_paths: List[str]


def selection_file_for_run_dir(run_dir: Path) -> Path:
    """
    Return the default selection file path under a run directory.

    Args:
        run_dir: Run directory path (e.g., runs/<run_name>/).

    Returns:
        Path: Selection file path for persona-weight config selection.
    """

    return run_dir / "streamlit_artifact_selections" / "persona_weight_configs.json"


def load_persona_weight_config_selection(
    selection_path: Path,
) -> PersonaWeightConfigSelection:
    """
    Load the persona-weight config selection file.

    Args:
        selection_path: Path to the selection JSON file.

    Returns:
        PersonaWeightConfigSelection: Loaded selection (empty if file does not exist).

    Raises:
        ValueError: If the file exists but has an invalid schema.
    """

    if not selection_path.exists():
        return PersonaWeightConfigSelection(config_paths=[])
    if not selection_path.is_file():
        raise ValueError(f"Selection path exists but is not a file: {selection_path}")

    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Selection file must be a JSON object, got {type(payload).__name__}: {selection_path}"
        )
    version = payload.get("version")
    if int(version or 0) != int(SELECTIONS_VERSION):
        raise ValueError(
            f"Unsupported selection file version {version!r} (expected {SELECTIONS_VERSION}). "
            f"Path: {selection_path}"
        )

    config_paths = payload.get("config_paths", [])
    if config_paths is None:
        config_paths = []
    if not isinstance(config_paths, list) or any(
        not isinstance(x, str) for x in config_paths
    ):
        raise ValueError(
            f"'config_paths' must be a list of strings. Path: {selection_path}"
        )
    # Keep order as stored; UI may want stable selection ordering.
    return PersonaWeightConfigSelection(config_paths=[str(x) for x in config_paths])


def save_persona_weight_config_selection(
    selection_path: Path,
    *,
    config_paths: List[str],
    app_version: Optional[str] = None,
) -> None:
    """
    Save the persona-weight config selection to disk.

    Args:
        selection_path: Path to selection JSON file.
        config_paths: Absolute persona YAML paths to store.
        app_version: Optional app version string.

    Returns:
        None

    Raises:
        ValueError: If config_paths is invalid.
    """

    if config_paths is None:
        raise ValueError("config_paths must be a list of strings.")
    if not isinstance(config_paths, list) or any(
        not isinstance(x, str) for x in config_paths
    ):
        raise ValueError("config_paths must be a list of strings.")

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": int(SELECTIONS_VERSION),
        "app_version": str(app_version) if app_version is not None else None,
        "config_paths": [str(x) for x in config_paths],
    }
    selection_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
