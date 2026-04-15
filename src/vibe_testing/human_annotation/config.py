"""Configuration loading helpers for standalone human annotation workflows."""

from __future__ import annotations

import logging
from pathlib import Path

from src.vibe_testing.human_annotation.schemas import (
    HumanAnnotationConfig,
    StudyArtifactPaths,
)
from src.vibe_testing.utils import load_config

logger = logging.getLogger(__name__)


def load_human_annotation_config(config_path: str | Path) -> HumanAnnotationConfig:
    """
    Load and validate a human annotation configuration file.

    Args:
        config_path (str | Path): YAML config path.

    Returns:
        HumanAnnotationConfig: Parsed configuration object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config payload is invalid.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Human annotation config not found: {path}")
    raw = load_config(str(path))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Human annotation config must be a non-empty mapping: {path}")
    cfg = HumanAnnotationConfig.from_dict(raw, config_path=path)
    logger.info("Loaded human annotation config from %s", path)
    return cfg


def build_study_artifact_paths(config: HumanAnnotationConfig) -> StudyArtifactPaths:
    """
    Build canonical workspace paths for a study run.

    Args:
        config (HumanAnnotationConfig): Parsed configuration.

    Returns:
        StudyArtifactPaths: Canonical file and directory paths.
    """
    study_dir = (
        config.outputs.workspace_dir.expanduser().resolve() / config.outputs.study_name
    )
    return StudyArtifactPaths(
        study_dir=study_dir,
        selection_manifest_path=study_dir / "selection_manifest.json",
        selection_plan_csv_path=study_dir / "selection_plan.csv",
        filter_summary_path=study_dir / "filter_summary.json",
        qualtrics_dir=study_dir / "qualtrics_files",
        processed_json_path=study_dir / "processed_annotations.json",
        processed_csv_path=study_dir / "processed_annotations.csv",
        stats_json_path=study_dir / "study_stats.json",
        analysis_summary_json_path=study_dir / "annotation_analysis_summary.json",
    )
