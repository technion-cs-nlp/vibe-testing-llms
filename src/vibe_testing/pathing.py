"""
Utilities for building standardized run directories and artifact filenames.

Canonical layout (user-centric)
-------------------------------

Given a base runs directory (e.g., ``debug_runs_x``) and a user group
``<user_group>``, artifacts are organized primarily under::

    <base_dir>/<user_group>/
        1_profile/
        2_samples_selection/<selection_method>/
        3_vibe_dataset/gen_model_<name>/filter_model_<name>/dataset_<run_id>/
        4_objective_evaluation/evaluated_model_<name>/gen_model_<name>/filter_model_<name>/<dataset_type>_<run_id>/
        5_subjective_evaluation/...
        5c_indicator_scores/...
        6_analysis/...
        runs/<model_name>/<run_id>/logs/...

Directory segments encode the stable metadata (user group, stage, model
roles, run identifier). Filenames are kept relatively light-weight and
only encode:

* ``artifact_type``  – primary artifact category (e.g., ``vibe_dataset``).
* ``evaluation_type`` – secondary category (e.g., ``personalization``).
* ``detail`` – local discriminator (e.g., ``sample-110``).
* ``timestamp`` and ``version`` – for uniqueness and provenance.

This module centralizes these conventions so both the high-level pipeline
orchestrator and individual stage scripts can agree on where artifacts and
logs are written, without duplicating path-building logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from src.vibe_testing.model_names import canonicalize_model_name, strip_old_prefix
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
)

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[^a-z0-9\-_]+")
DETAIL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-_:]*$")
ARTIFACT_PATTERN = re.compile(
    r"^(?P<artifact>[a-z0-9_-]+)_(?P<evaluation>[a-z0-9_-]+)"
    r"_(?P<detail>[a-z0-9\-_:]+?)(?:_(?P<timestamp>\d{8}T\d{6}Z))?"
    r"_(?P<version>v\d{2})\.(?P<ext>[a-z0-9]+)$"
)


@dataclass(frozen=True)
class PairwiseModelRouting:
    """
    Canonical and routed identity for an unordered pairwise model comparison.

    Attributes:
        canonical_model_a: Canonical left-side model name (sorted, order-insensitive).
        canonical_model_b: Canonical right-side model name (sorted, order-insensitive).
        canonical_model_pair: Canonical unordered model-pair token.
        route_model_a_token: On-disk left-side model token used in pairwise directories.
        route_model_b_token: On-disk right-side model token used in pairwise directories.
        route_model_pair_segment: Directory segment `models_<a>_vs_<b>`.
    """

    canonical_model_a: str
    canonical_model_b: str
    canonical_model_pair: str
    route_model_a_token: str
    route_model_b_token: str
    route_model_pair_segment: str


@dataclass(frozen=True)
class ParsedPairwiseArtifactPath:
    """
    Parsed metadata for a Stage-5b pairwise artifact path.

    Attributes:
        persona: Persona root segment.
        model_routing: Canonical and routed model-pair information.
        judge_dir_name: Judge directory token.
        generator_model: Generator-model directory token.
        filter_model: Filter-model directory token.
        prompt_type: Optional prompt type segment.
        pairwise_judgment_type: Normalized pairwise judgment type.
    """

    persona: str
    model_routing: PairwiseModelRouting
    judge_dir_name: str
    generator_model: str
    filter_model: str
    prompt_type: Optional[str]
    pairwise_judgment_type: str


def build_run_id(
    timestamp: Optional[datetime] = None,
    run_name: str = "no_name",
) -> str:
    """
    Build a canonical run identifier using a UTC timestamp.

    Args:
        timestamp (Optional[datetime]): If provided, use this timestamp; otherwise use now (UTC).
        run_name (str): Human-readable label for the run.

    Returns:
        str: Formatted run identifier, e.g., ``demo-run_20250214_091500Z``.
    """
    normalized_name = normalize_token(run_name or "no_name")
    ts = timestamp or datetime.utcnow().replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    ts_str = ts.strftime("%Y%m%d_%H%M%SZ")
    return f"{normalized_name}_{ts_str}"


def normalize_token(value: str) -> str:
    """
    Normalize stage/user/model tokens to lowercase kebab-case.

    Args:
        value (str): Source value provided by the caller.

    Returns:
        str: Normalized token suitable for path segments.

    Raises:
        ValueError: If the value becomes empty after normalization.
    """
    normalized = TOKEN_PATTERN.sub("-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError(f"Cannot normalize empty token from value '{value}'.")
    return normalized


def canonicalize_pairwise_model_routing(
    model_a: str,
    model_b: str,
    *,
    route_model_a: Optional[str] = None,
    route_model_b: Optional[str] = None,
) -> PairwiseModelRouting:
    """
    Build canonical and routed identity for an unordered pairwise model pair.

    Args:
        model_a (str): First compared model identifier.
        model_b (str): Second compared model identifier.
        route_model_a (Optional[str]): Optional preferred on-disk token for model A.
        route_model_b (Optional[str]): Optional preferred on-disk token for model B.

    Returns:
        PairwiseModelRouting: Canonical and routed pair metadata.
    """
    canonical_pair = sorted(
        [canonicalize_model_name(model_a), canonicalize_model_name(model_b)],
        key=lambda value: (normalize_token(value or "unspecified"), value or ""),
    )
    route_pair = sorted(
        [
            normalize_token(
                strip_old_prefix(route_model_a or model_a) or "unspecified"
            ),
            normalize_token(
                strip_old_prefix(route_model_b or model_b) or "unspecified"
            ),
        ]
    )
    return PairwiseModelRouting(
        canonical_model_a=canonical_pair[0],
        canonical_model_b=canonical_pair[1],
        canonical_model_pair=f"{canonical_pair[0]}_vs_{canonical_pair[1]}",
        route_model_a_token=route_pair[0],
        route_model_b_token=route_pair[1],
        route_model_pair_segment=f"models_{route_pair[0]}_vs_{route_pair[1]}",
    )


def parse_pairwise_artifact_path(
    base_dir: Path | str, artifact_path: Path | str
) -> ParsedPairwiseArtifactPath:
    """
    Parse canonical metadata from a Stage-5b pairwise artifact path.

    Args:
        base_dir (Path | str): Root results directory.
        artifact_path (Path | str): Pairwise artifact file path.

    Returns:
        ParsedPairwiseArtifactPath: Parsed path metadata.

    Raises:
        ValueError: If the artifact path does not match the expected layout.
    """
    base_path = Path(base_dir).expanduser().resolve()
    path = Path(artifact_path).expanduser().resolve()
    relative_parts = path.relative_to(base_path).parts
    if len(relative_parts) < 7:
        raise ValueError(f"Unexpected pairwise artifact path: {path}")
    if relative_parts[1] != "5b_pairwise_evaluation":
        raise ValueError(f"Artifact is not under 5b_pairwise_evaluation: {path}")

    persona = relative_parts[0]
    model_pair_dir = relative_parts[2]
    judge_dir = relative_parts[3]
    gen_dir = relative_parts[4]
    filter_dir = relative_parts[5]
    tail_parts = list(relative_parts[6:-1])

    if not model_pair_dir.startswith("models_") or "_vs_" not in model_pair_dir:
        raise ValueError(f"Invalid model pair directory in path: {path}")
    if not (
        judge_dir.startswith("judge_model_") or judge_dir.startswith("OLD_judge_model_")
    ):
        raise ValueError(f"Invalid judge directory in path: {path}")
    if not gen_dir.startswith("gen_model_"):
        raise ValueError(f"Invalid generator directory in path: {path}")
    if not filter_dir.startswith("filter_model_"):
        raise ValueError(f"Invalid filter directory in path: {path}")

    pair_name = model_pair_dir.split("models_", 1)[1]
    model_a_dir, model_b_dir = pair_name.split("_vs_", 1)
    prompt_type = None
    judgment_type = PAIRWISE_JUDGMENT_TYPE_PERSONA
    for part in tail_parts:
        if part.startswith("judgment_type_"):
            judgment_type = part.split("judgment_type_", 1)[1]
        elif part.startswith("prompt_type_"):
            prompt_type = part.split("prompt_type_", 1)[1]

    model_routing = canonicalize_pairwise_model_routing(
        model_a_dir,
        model_b_dir,
        route_model_a=model_a_dir,
        route_model_b=model_b_dir,
    )
    return ParsedPairwiseArtifactPath(
        persona=persona,
        model_routing=model_routing,
        judge_dir_name=(
            judge_dir.split("judge_model_", 1)[1]
            if "judge_model_" in judge_dir
            else judge_dir
        ),
        generator_model=gen_dir.replace("gen_model_", "", 1),
        filter_model=filter_dir.replace("filter_model_", "", 1),
        prompt_type=prompt_type,
        pairwise_judgment_type=normalize_pairwise_judgment_type(judgment_type),
    )


def infer_prompt_type(identifier: str) -> str:
    """
    Infers the prompt type (original, personalized, or control) from a string.

    Args:
        identifier (str): A filename, sample ID, or directory name.

    Returns:
        str: One of 'original', 'personalized', or 'control'.
    """
    if not identifier:
        return "original"

    val = identifier.lower()

    # Control prompts heuristics
    if (
        "control_var" in val
        or "::control" in val
        or "-control-metrics" in val
        or "::variation::control_" in val
        or "variation_control_" in val
    ):
        return "control"

    # Personalized prompts heuristics
    # - Contains variation marker (::variation::)
    # - Contains variation number (-var1, -var2, etc.)
    # - Contains personalized metrics marker
    if (
        "::variation::" in val
        or "-var" in val
        or "-variation-" in val
        or "-personalized-metrics" in val
    ):
        return "personalized"

    # Default to original
    return "original"


def validate_detail_token(detail: str) -> str:
    """
    Validate the detail token used within filenames.

    Args:
        detail (str): Combined metadata segment (persona, scenario, etc.).

    Returns:
        str: The original detail if valid.

    Raises:
        ValueError: If disallowed characters are present.
    """
    if not DETAIL_PATTERN.match(detail):
        raise ValueError(
            "Detail token must be lowercase and may include '-', '_', or ':'."
        )
    return detail


def _user_root(base_dir: Path | str, user_profile_name: str) -> Path:
    """
    Resolve the base directory for a specific user profile name.

    Args:
        base_dir (Path | str): Root "Runs" directory provided by the caller.
        user_profile_name (str): Display/cohort/user profile name.

    Returns:
        Path: Directory path for the user profile root.
    """
    root = Path(base_dir)
    return root / normalize_token(user_profile_name)


def _model_segment(prefix: str, model_name: str) -> str:
    """
    Build a consistent directory segment for a model identifier.

    Args:
        prefix (str): Descriptor prefix (e.g., 'gen_model').
        model_name (str): Raw model identifier.

    Returns:
        str: Combined directory-safe segment.
    """
    normalized_name = normalize_token(model_name or "unspecified")
    return f"{prefix}_{normalized_name}"


def profile_stage_dir(base_dir: Path | str, user_profile_name: str) -> Path:
    """
    Canonical directory for Stage 1 profile artifacts.
    """
    return _user_root(base_dir, user_profile_name) / "1_profile"


def selection_stage_dir(
    base_dir: Path | str, user_profile_name: str, selection_method: str
) -> Path:
    """
    Canonical directory for Stage 2 selection outputs.
    """
    sanitized_method = normalize_token(selection_method or "default_selection")
    return (
        _user_root(base_dir, user_profile_name)
        / "2_samples_selection"
        / sanitized_method
    )


def vibe_dataset_stage_dir(
    base_dir: Path | str,
    user_profile_name: str,
    generator_model: str,
    filter_model: str,
) -> Path:
    """
    Canonical directory for Stage 3 vibe dataset artifacts.
    """
    return (
        _user_root(base_dir, user_profile_name)
        / "3_vibe_dataset"
        / _model_segment("gen_model", generator_model)
        / _model_segment("filter_model", filter_model or "none")
    )


def objective_stage_dir(
    base_dir: Path | str,
    user_profile_name: str,
    evaluated_model: str,
    generator_model: str,
    filter_model: str,
    prompt_type: Optional[str] = None,
) -> Path:
    """
    Canonical directory for Stage 4 objective evaluation outputs.
    """
    base = (
        _user_root(base_dir, user_profile_name)
        / "4_objective_evaluation"
        / _model_segment("evaluated_model", evaluated_model)
        / _model_segment("gen_model", generator_model)
        / _model_segment("filter_model", filter_model or "none")
    )
    if prompt_type:
        base = base / f"prompt_type_{normalize_token(prompt_type)}"
    return base


def subjective_stage_dir(
    base_dir: Path | str,
    user_profile_name: str,
    evaluated_model: str,
    judge_model: str,
    generator_model: str,
    filter_model: str,
    prompt_type: Optional[str] = None,
) -> Path:
    """
    Canonical directory for Stage 5 subjective evaluation outputs.
    """
    base = (
        _user_root(base_dir, user_profile_name)
        / "5_subjective_evaluation"
        / _model_segment("evaluated_model", evaluated_model)
        / _model_segment("sub_judge_model", judge_model)
        / _model_segment("gen_model", generator_model)
        / _model_segment("filter_model", filter_model or "none")
    )
    if prompt_type:
        base = base / f"prompt_type_{normalize_token(prompt_type)}"
    return base


def indicator_stage_dir(
    base_dir: Path | str,
    user_profile_name: str,
    evaluated_model: str,
    generator_model: str,
    filter_model: str,
    prompt_type: Optional[str] = None,
) -> Path:
    """
    Canonical directory for Stage 5c non-LLM-judge indicator outputs.

    Stage 5c intentionally does NOT encode a judge model in the path because
    indicator computation is deterministic and does not call an LLM judge.
    """
    base = (
        _user_root(base_dir, user_profile_name)
        / "5c_indicator_scores"
        / _model_segment("evaluated_model", evaluated_model)
        / _model_segment("gen_model", generator_model)
        / _model_segment("filter_model", filter_model or "none")
    )
    if prompt_type:
        base = base / f"prompt_type_{normalize_token(prompt_type)}"
    return base


def pairwise_stage_dir(
    base_dir: Path | str,
    user_profile_name: str,
    model_a: str,
    model_b: str,
    judge_model: str,
    generator_model: str,
    filter_model: str,
    judgment_type: Optional[str] = None,
    prompt_type: Optional[str] = None,
) -> Path:
    """
    Canonical directory for Stage 5B pairwise comparison evaluation outputs.

    Args:
        base_dir: Base runs directory.
        user_profile_name: User persona identifier.
        model_a: First model in comparison.
        model_b: Second model in comparison.
        judge_model: Judge model name.
        generator_model: Dataset generator model name.
        filter_model: Filter model name.
        judgment_type: Optional pairwise judgment type. When omitted or equal to
            ``persona``, the legacy directory layout is preserved.

    Returns:
        Path to pairwise evaluation directory.
    """
    routing = canonicalize_pairwise_model_routing(model_a, model_b)

    base = (
        _user_root(base_dir, user_profile_name)
        / "5b_pairwise_evaluation"
        / routing.route_model_pair_segment
        / _model_segment("judge_model", judge_model)
        / _model_segment("gen_model", generator_model)
        / _model_segment("filter_model", filter_model or "none")
    )
    normalized_judgment_type = normalize_pairwise_judgment_type(judgment_type)
    if normalized_judgment_type != PAIRWISE_JUDGMENT_TYPE_PERSONA:
        base = base / f"judgment_type_{normalized_judgment_type}"
    if prompt_type:
        base = base / f"prompt_type_{normalize_token(prompt_type)}"
    return base


def analysis_stage_dir(
    base_dir: Path | str,
    user_profile_name: str,
    evaluated_model: str,
    judge_model: str,
    generator_model: str,
    filter_model: str,
    prompt_type: Optional[str] = None,
) -> Path:
    """
    Canonical directory for Stage 6 analysis outputs.
    """
    base = (
        _user_root(base_dir, user_profile_name)
        / "6_analysis"
        / _model_segment("evaluated_model", evaluated_model)
        / _model_segment("sub_judge_model", judge_model)
        / _model_segment("gen_model", generator_model)
        / _model_segment("filter_model", filter_model or "none")
    )
    if prompt_type:
        base = base / f"prompt_type_{normalize_token(prompt_type)}"
    return base


def get_run_root(
    base_dir: Path | str, stage: str, user_group: str, model: str, run_id: str
) -> Path:
    """
    Build the per-run root directory under the user-centric tree.

    The ``stage`` parameter is accepted for backward compatibility and
    logging, but is no longer encoded in the directory structure. All
    stages that share ``(user_group, model, run_id)`` will therefore share
    the same run root.

    Layout::

        <base_dir>/<user_group>/runs/<model>/<run_id>

    Args:
        base_dir (Path | str): Base runs directory.
        stage (str): Pipeline stage identifier (unused for path segments).
        user_group (str): User persona or cohort.
        model (str): Model name.
        run_id (str): Canonical run identifier.

    Returns:
        Path: Fully qualified run directory path.
    """
    user_root = _user_root(base_dir, user_group)
    return user_root / "runs" / normalize_token(model or "unspecified") / run_id


def ensure_run_tree(root: Path, minimal: bool = True) -> Dict[str, Path]:
    """
    Ensure the canonical run subdirectories exist.

    Args:
        root (Path): Run root directory path.
        minimal (bool): If True, don't create directories (lazy creation). If False, create full tree.

    Returns:
        Dict[str, Path]: Mapping of logical names to paths.
    """
    logs_dir = root / "logs"

    structure = {
        "logs": logs_dir,
        "inputs_raw": root / "inputs" / "raw",
        "inputs_processed": root / "inputs" / "processed",
        "results_objective": root / "results" / "objective",
        "results_subjective": root / "results" / "subjective",
        "artifacts": root / "artifacts",
    }

    if not minimal:
        # Create full structure for backward compatibility
        root.mkdir(parents=True, exist_ok=True)
        for path in structure.values():
            path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured run tree at %s (minimal=%s)", root, minimal)
    else:
        # In minimal mode, don't create any directories yet
        # They will be created lazily when actually needed:
        # - logs/ will be created by setup_logger() when writing log files
        # - root/ will be created by stage 4 when creating temp files
        # - Other directories are not used in minimal mode
        logger.debug(
            "Minimal mode: directories will be created lazily when needed at %s", root
        )

    return structure


@dataclass(frozen=True)
class ArtifactName:
    """Structured representation of an artifact filename."""

    artifact_type: str
    evaluation_type: str
    detail: str
    timestamp: Optional[str]
    version: int
    extension: str

    @property
    def version_label(self) -> str:
        """Return the zero-padded version label."""
        return f"v{self.version:02d}"


def format_artifact_name(
    artifact_type: str,
    evaluation_type: str,
    detail: str,
    version: int,
    ext: str,
    timestamp: Optional[datetime] = None,
    run_name: str = "no_name",
) -> str:
    """
    Build a standardized artifact filename.

    The filename intentionally omits the run name because that metadata is
    already captured in the directory structure (via the run root).

    Args:
        artifact_type (str): Primary artifact category (e.g., function_eval).
        evaluation_type (str): Secondary category (objective/subjective/etc.).
        detail (str): Metadata token (scenario/persona pair).
        version (int): Zero-based iteration counter.
        ext (str): File extension without dot.
        run_name (str): (Deprecated) Human-friendly run label; ignored.

    Returns:
        str: Canonical filename.
    """
    for token_name, token_value in (
        ("artifact_type", artifact_type),
        ("evaluation_type", evaluation_type),
    ):
        normalize_token(token_value)
    validate_detail_token(detail)
    sanitized_ext = normalize_token(ext).replace("-", "")
    version_label = f"v{version:02d}"
    ts_label = None
    if timestamp:
        ts_obj = timestamp
        if isinstance(ts_obj, datetime):
            ts_obj = ts_obj.astimezone(timezone.utc)
        ts_label = ts_obj.strftime("%Y%m%dT%H%M%SZ")
    ts_segment = f"_{ts_label}" if ts_label else ""
    filename = f"{artifact_type}_{evaluation_type}_{detail}{ts_segment}_{version_label}.{sanitized_ext}"
    logger.debug("Formatted artifact filename: %s", filename)
    return filename


def parse_artifact_name(name: str) -> ArtifactName:
    """
    Parse a canonical artifact filename.

    Args:
        name (str): Filename to parse.

    Returns:
        ArtifactName: Parsed components.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    match = ARTIFACT_PATTERN.match(name)
    if not match:
        raise ValueError(f"Filename '{name}' does not match artifact pattern.")
    version = int(match.group("version")[1:])
    return ArtifactName(
        artifact_type=match.group("artifact"),
        evaluation_type=match.group("evaluation"),
        detail=match.group("detail"),
        timestamp=match.group("timestamp"),
        version=version,
        extension=match.group("ext"),
    )


@dataclass
class RunContext:
    """Container for canonical run directories."""

    stage: str
    user_group: str
    model_name: str
    run_id: str
    run_name: str
    root: Path
    inputs_raw: Path
    inputs_processed: Path
    results_objective: Path
    results_subjective: Path
    artifacts: Path
    logs: Path

    def artifact_path(
        self,
        base: Path,
        artifact_type: str,
        evaluation_type: str,
        detail: str,
        version: int,
        ext: str,
    ) -> Path:
        """
        Compute a Path for an artifact inside the provided base directory.
        """
        filename = format_artifact_name(
            artifact_type=artifact_type,
            evaluation_type=evaluation_type,
            detail=detail,
            version=version,
            ext=ext,
            run_name=self.run_name,
        )
        return base / filename


def build_run_context(
    stage: str,
    user_group: str,
    model_name: str,
    run_id: Optional[str] = None,
    base_dir: Path | str = "runs",
    ensure_tree_flag: bool = True,
    root_override: Optional[Path | str] = None,
    run_name: str = "no_name",
    minimal_tree: bool = True,
) -> RunContext:
    """
    Create a RunContext, optionally ensuring directories exist.

    Args:
        stage (str): Pipeline stage identifier.
        user_group (str): User cohort identifier.
        model_name (str): Evaluated model name.
        run_id (Optional[str]): Canonical run id (auto-generated if None).
        base_dir (Path | str): Base directory for all runs.
        root_override (Optional[Path | str]): Optional explicit root path.
        ensure_tree_flag (bool): Whether to create directories.
        run_name (str): Descriptive label prepended to run_id timestamps.
        minimal_tree (bool): If True, only create logs and root. If False, create full tree.

    Returns:
        RunContext: Populated context object.
    """
    resolved_run_id = run_id or build_run_id(run_name=run_name)
    if root_override:
        root = Path(root_override)
    else:
        root = get_run_root(base_dir, stage, user_group, model_name, resolved_run_id)
    if ensure_tree_flag:
        structure = ensure_run_tree(root, minimal=minimal_tree)
    else:
        structure = {
            "inputs_raw": root / "inputs" / "raw",
            "inputs_processed": root / "inputs" / "processed",
            "results_objective": root / "results" / "objective",
            "results_subjective": root / "results" / "subjective",
            "artifacts": root / "artifacts",
            "logs": root / "logs",
        }
    logger.info(
        "Run context stage=%s user_group=%s model=%s run_id=%s root=%s",
        stage,
        user_group,
        model_name,
        resolved_run_id,
        root,
    )
    return RunContext(
        stage=normalize_token(stage),
        user_group=normalize_token(user_group),
        model_name=normalize_token(model_name),
        run_id=resolved_run_id,
        run_name=normalize_token(run_name or "no_name"),
        root=root,
        inputs_raw=structure["inputs_raw"],
        inputs_processed=structure["inputs_processed"],
        results_objective=structure["results_objective"],
        results_subjective=structure["results_subjective"],
        artifacts=structure["artifacts"],
        logs=structure["logs"],
    )
