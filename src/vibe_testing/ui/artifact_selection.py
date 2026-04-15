"""
Artifact selection persistence for Streamlit pairwise explorer.

This module provides a small, testable layer that lets Streamlit persist explicit
choices of which Stage-5b artifact file to use for each (config, model pair, judge)
selection *within a run directory*.

It also implements the "Stage 6 default" artifact selection heuristic used by the
Stage 6 scanner: choose the candidate file with the latest modification time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.vibe_testing.pairwise_judgment_types import PAIRWISE_JUDGMENT_TYPE_PERSONA
from src.vibe_testing.pathing import parse_artifact_name

logger = logging.getLogger(__name__)

SELECTIONS_VERSION = 2


@dataclass(frozen=True)
class PairwiseArtifactSelectionKey:
    """
    Stable identifier for a (config slice, model pair, judge) artifact selection.

    Attributes:
        persona: Persona directory token.
        generator_model: Generator model token (directory suffix after 'gen_model_').
        filter_model: Filter model token (directory suffix after 'filter_model_').
        prompt_type: Prompt type label, or 'legacy' when absent.
        pairwise_judgment_type: Pairwise judgment-type token.
        model_a: Left model token from the model-pair directory name.
        model_b: Right model token from the model-pair directory name.
        judge_model: Judge model token (directory suffix after 'judge_model_').
    """

    persona: str
    generator_model: str
    filter_model: str
    prompt_type: str
    model_a: str
    model_b: str
    judge_model: str
    pairwise_judgment_type: str = PAIRWISE_JUDGMENT_TYPE_PERSONA

    def to_string(self) -> str:
        """
        Render the key into a stable string used as a JSON dict key.

        Returns:
            str: Stable key string.
        """

        parts = [
            f"persona={self.persona}",
            f"gen={self.generator_model}",
            f"filter={self.filter_model}",
            f"prompt={self.prompt_type}",
            f"judgment_type={self.pairwise_judgment_type}",
            f"pair={self.model_a}_vs_{self.model_b}",
            f"judge={self.judge_model}",
        ]
        return "|".join(parts)


def selection_file_for_run_dir(run_dir: Path) -> Path:
    """
    Return the default selection file path under a run directory.

    Args:
        run_dir: Run directory path (e.g., runs/<run_name>/).

    Returns:
        Path: Selection file path.
    """

    return run_dir / "streamlit_artifact_selections" / "pairwise_artifacts.json"


def choose_stage6_default_artifact(candidates: Iterable[Path]) -> Path:
    """
    Choose the Stage-6-default artifact from candidates.

    Stage 6 selects the most recently modified artifact file when multiple are
    present in the same directory.

    Args:
        candidates: Candidate artifact paths.

    Returns:
        Path: Chosen artifact path.

    Raises:
        ValueError: If candidates is empty.
    """

    paths = [Path(p) for p in candidates]
    if not paths:
        raise ValueError("No candidate artifact paths provided for Stage 6 selection.")

    def _mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except FileNotFoundError:
            return 0.0

    # Deterministic tie-break: fall back to lexical path string.
    return max(paths, key=lambda p: (_mtime(p), str(p)))


def format_artifact_option_label(
    path: Path,
    *,
    stage6_default_path: Optional[Path] = None,
) -> str:
    """
    Format a compact label for an artifact option.

    The label is intentionally compact; Streamlit should show the full path in a
    separate read-only text area.

    Args:
        path: Artifact path.
        stage6_default_path: Stage-6-default path (to mark in the label).

    Returns:
        str: Human-readable label.
    """

    prefix = ""
    if stage6_default_path is not None and Path(path) == Path(stage6_default_path):
        prefix = "[Stage6 default] "

    version: Optional[int] = None
    timestamp: str = ""
    try:
        parsed = parse_artifact_name(Path(path).name)
        version = int(parsed.version)
        timestamp = str(parsed.timestamp or "")
    except Exception:
        version = None
        timestamp = ""

    base = Path(path).name
    if version is None and not timestamp:
        return f"{prefix}{base}"
    parts: List[str] = []
    if version is not None:
        parts.append(f"v{version:02d}")
    if timestamp:
        parts.append(timestamp)
    parts.append(base)
    return prefix + " | ".join(parts)


def load_pairwise_artifact_selections(selection_path: Path) -> Dict[str, dict]:
    """
    Load pairwise artifact selections from a JSON file.

    Args:
        selection_path: Path to selection file.

    Returns:
        Dict[str, dict]: Mapping from key string -> record dict.

    Raises:
        ValueError: If the JSON payload is invalid or has an unexpected schema.
    """

    if not selection_path.exists():
        return {}
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

    selections = payload.get("selections", {})
    if selections is None:
        return {}
    if not isinstance(selections, dict):
        raise ValueError(
            f"'selections' must be a JSON object, got {type(selections).__name__}: {selection_path}"
        )
    # Ensure all keys map to dict records.
    out: Dict[str, dict] = {}
    for k, v in selections.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, dict):
            raise ValueError(
                f"Selection record for key {k!r} must be an object, got {type(v).__name__}."
            )
        out[k] = dict(v)
    return out


def save_pairwise_artifact_selections(
    selection_path: Path,
    *,
    selections_by_key: Dict[str, dict],
    app_version: Optional[str] = None,
) -> None:
    """
    Save pairwise artifact selections to disk.

    Args:
        selection_path: Destination selection file path.
        selections_by_key: Mapping key -> record dict.
        app_version: Optional app version identifier.
    """

    if not isinstance(selections_by_key, dict):
        raise ValueError("selections_by_key must be a dict.")

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    payload = {
        "version": int(SELECTIONS_VERSION),
        "saved_at_utc": now,
        "app_version": str(app_version) if app_version else None,
        "selections": selections_by_key,
    }

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def get_saved_selected_path(
    selections_by_key: Dict[str, dict],
    key: PairwiseArtifactSelectionKey,
) -> Optional[str]:
    """
    Get the saved selected artifact path (string) for a key, if present.

    Args:
        selections_by_key: Mapping key string -> record dict.
        key: Selection key.

    Returns:
        Optional[str]: Saved selected path, if present.
    """

    record = selections_by_key.get(key.to_string())
    if not isinstance(record, dict):
        return None
    value = record.get("selected_path")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def upsert_selection_record(
    selections_by_key: Dict[str, dict],
    *,
    key: PairwiseArtifactSelectionKey,
    selected_path: str,
    stage6_default_path: str,
) -> Dict[str, dict]:
    """
    Insert or update a selection record for the given key.

    Args:
        selections_by_key: Existing mapping.
        key: Selection key.
        selected_path: Selected artifact path string.
        stage6_default_path: Stage-6-default artifact path string.

    Returns:
        Dict[str, dict]: Updated mapping (new dict).
    """

    out = dict(selections_by_key or {})
    out[key.to_string()] = {
        "persona": key.persona,
        "generator_model": key.generator_model,
        "filter_model": key.filter_model,
        "prompt_type": key.prompt_type,
        "pairwise_judgment_type": key.pairwise_judgment_type,
        "model_a": key.model_a,
        "model_b": key.model_b,
        "judge_model": key.judge_model,
        "selected_path": str(selected_path),
        "stage6_default_path": str(stage6_default_path),
    }
    return out
