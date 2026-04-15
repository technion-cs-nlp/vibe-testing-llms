"""
Persistence for Streamlit Aggregate-mode filter selections.

This module stores the Aggregate-mode configuration scope used by the pairwise
explorer so reopening the app can restore the same filter selections for a
given run directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SELECTIONS_VERSION = 1


@dataclass(frozen=True)
class AggregateFilterSelection:
    """
    Selected Aggregate-mode filters for the Streamlit pairwise explorer.

    Attributes:
        personas: Selected persona directory tokens.
        generator_models: Selected generator-model tokens.
        filter_models: Selected filter-model tokens.
        prompt_types: Selected prompt-type labels.
        pairwise_judgment_types: Selected pairwise judgment-type labels.
        model_pairs: Selected model-pair labels.
        judges: Selected judge tokens.
    """

    personas: List[str]
    generator_models: List[str]
    filter_models: List[str]
    prompt_types: List[str]
    pairwise_judgment_types: List[str]
    model_pairs: List[str]
    judges: List[str]


def selection_file_for_run_dir(run_dir: Path) -> Path:
    """
    Return the default Aggregate-filter selection path under a run directory.

    Args:
        run_dir: Run directory path (e.g., runs/<run_name>/).

    Returns:
        Path: Selection file path for Aggregate-mode filter selections.
    """

    return run_dir / "streamlit_artifact_selections" / "aggregate_filters.json"


def load_aggregate_filter_selection(selection_path: Path) -> AggregateFilterSelection:
    """
    Load the Aggregate-mode filter selection file.

    Args:
        selection_path: Path to the selection JSON file.

    Returns:
        AggregateFilterSelection: Loaded selection, or empty selections when the
            file does not exist.

    Raises:
        ValueError: If the file exists but has an invalid schema.
    """

    if not selection_path.exists():
        return AggregateFilterSelection(
            personas=[],
            generator_models=[],
            filter_models=[],
            prompt_types=[],
            pairwise_judgment_types=[],
            model_pairs=[],
            judges=[],
        )
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

    def _read_string_list(field_name: str) -> List[str]:
        value = payload.get(field_name, [])
        if value is None:
            return []
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise ValueError(
                f"{field_name!r} must be a list of strings. Path: {selection_path}"
            )
        return [str(x) for x in value]

    selection = AggregateFilterSelection(
        personas=_read_string_list("personas"),
        generator_models=_read_string_list("generator_models"),
        filter_models=_read_string_list("filter_models"),
        prompt_types=_read_string_list("prompt_types"),
        pairwise_judgment_types=_read_string_list("pairwise_judgment_types"),
        model_pairs=_read_string_list("model_pairs"),
        judges=_read_string_list("judges"),
    )
    logger.debug("Loaded aggregate filter selection from %s", selection_path)
    return selection


def save_aggregate_filter_selection(
    selection_path: Path,
    *,
    selection: AggregateFilterSelection,
    app_version: Optional[str] = None,
) -> None:
    """
    Save Aggregate-mode filter selections to disk.

    Args:
        selection_path: Path to the selection JSON file.
        selection: Aggregate-mode selections to store.
        app_version: Optional app version string.

    Returns:
        None

    Raises:
        ValueError: If the selection object has invalid fields.
    """

    if not isinstance(selection, AggregateFilterSelection):
        raise ValueError("selection must be an AggregateFilterSelection instance.")

    for field_name, value in (
        ("personas", selection.personas),
        ("generator_models", selection.generator_models),
        ("filter_models", selection.filter_models),
        ("prompt_types", selection.prompt_types),
        ("pairwise_judgment_types", selection.pairwise_judgment_types),
        ("model_pairs", selection.model_pairs),
        ("judges", selection.judges),
    ):
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise ValueError(f"{field_name} must be a list of strings.")

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": int(SELECTIONS_VERSION),
        "app_version": str(app_version) if app_version is not None else None,
        "personas": [str(x) for x in selection.personas],
        "generator_models": [str(x) for x in selection.generator_models],
        "filter_models": [str(x) for x in selection.filter_models],
        "prompt_types": [str(x) for x in selection.prompt_types],
        "pairwise_judgment_types": [str(x) for x in selection.pairwise_judgment_types],
        "model_pairs": [str(x) for x in selection.model_pairs],
        "judges": [str(x) for x in selection.judges],
    }
    selection_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("Saved aggregate filter selection to %s", selection_path)
