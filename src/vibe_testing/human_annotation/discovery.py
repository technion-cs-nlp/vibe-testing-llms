"""Discovery utilities for Stage-5b pairwise artifacts used in human studies."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
)
from src.vibe_testing.human_annotation.schemas import (
    HumanAnnotationConfig,
    PairwiseCandidateRecord,
)
from src.vibe_testing.pathing import (
    ParsedPairwiseArtifactPath,
    canonicalize_pairwise_model_routing,
    infer_prompt_type,
    parse_pairwise_artifact_path,
)
from src.vibe_testing.ui.pairwise_explorer_io import load_pairwise_json

logger = logging.getLogger(__name__)

VARIATION_TOKEN = "::variation::"


def _resolve_variant_fields(identifier: str) -> Tuple[str, str]:
    """
    Derive task and variant identifiers from a raw task identifier.

    Args:
        identifier (str): Raw task identifier from a pairwise artifact.

    Returns:
        Tuple[str, str]: Base task id and variant id.
    """
    if VARIATION_TOKEN not in identifier:
        return identifier, identifier
    base_identifier, variant_segment = identifier.split(VARIATION_TOKEN, 1)
    return base_identifier, variant_segment or base_identifier


def _iter_pairwise_artifacts(base_results_dir: Path) -> Iterable[Path]:
    """
    Yield all pairwise-comparison artifact files beneath a results directory.

    Args:
        base_results_dir (Path): Root results directory.

    Yields:
        Path: Candidate pairwise artifact paths.

    Raises:
        FileNotFoundError: If the base directory is missing.
        ValueError: If the base path is not a directory.
    """
    if not base_results_dir.exists():
        raise FileNotFoundError(
            f"Base results directory does not exist: {base_results_dir}"
        )
    if not base_results_dir.is_dir():
        raise ValueError(f"Base results path must be a directory: {base_results_dir}")
    for path in sorted(base_results_dir.rglob("*pairwise-comparison*.json")):
        if path.is_file():
            yield path


def _canonical_model_pair_token(model_a: str, model_b: str) -> str:
    """
    Build a canonical unordered model-pair token.

    Args:
        model_a (str): First model identifier.
        model_b (str): Second model identifier.

    Returns:
        str: Canonical unordered pair token.
    """
    return canonicalize_pairwise_model_routing(model_a, model_b).canonical_model_pair


def _matches_allowed_model_pair(
    parsed_path: ParsedPairwiseArtifactPath, allowed_pairs: Optional[List[str]]
) -> bool:
    """
    Check whether a parsed artifact path matches the configured model-pair allowlist.

    Args:
        parsed_path (ParsedPairwiseArtifactPath): Parsed artifact-path metadata.
        allowed_pairs (Optional[List[str]]): Optional configured allowlist.

    Returns:
        bool: True when the parsed pair should be kept.
    """
    if not allowed_pairs:
        return True
    parsed_canonical_pair = parsed_path.model_routing.canonical_model_pair
    for allowed_pair in allowed_pairs:
        if "_vs_" not in allowed_pair:
            continue
        allowed_model_a, allowed_model_b = allowed_pair.split("_vs_", 1)
        if (
            _canonical_model_pair_token(allowed_model_a, allowed_model_b)
            == parsed_canonical_pair
        ):
            return True
    return False


def _source_rejection_field(
    parsed_path: ParsedPairwiseArtifactPath, config: HumanAnnotationConfig
) -> Optional[str]:
    """
    Check whether parsed path metadata passes source-level allowlists.

    Args:
        parsed_path (ParsedPairwiseArtifactPath): Parsed path metadata.
        config (HumanAnnotationConfig): Global study configuration.

    Returns:
        Optional[str]: Rejected source field name, or None when the artifact passes.
    """
    source = config.source
    checks = [
        ("personas", parsed_path.persona),
        ("prompt_types", parsed_path.prompt_type or "legacy"),
        ("judges", parsed_path.judge_dir_name),
        (
            "pairwise_judgment_types",
            normalize_pairwise_judgment_type(
                parsed_path.pairwise_judgment_type or PAIRWISE_JUDGMENT_TYPE_PERSONA
            ),
        ),
    ]
    for field_name, value in checks:
        allowed = getattr(source, field_name)
        if allowed and value not in allowed:
            return field_name
    if not _matches_allowed_model_pair(parsed_path, source.model_pairs):
        return "model_pairs"
    return None


def _build_source_key(
    *,
    persona: str,
    prompt_type: str,
    pairwise_judgment_type: str,
    model_a_name: str,
    model_b_name: str,
    task_id: str,
    variant_id: str,
) -> str:
    """
    Build a stable identity key for cross-judge grouping.

    Args:
        persona (str): Persona identifier.
        prompt_type (str): Prompt type.
        pairwise_judgment_type (str): Judgment type.
        model_a_name (str): Compared model A.
        model_b_name (str): Compared model B.
        task_id (str): Base task id.
        variant_id (str): Variant id.

    Returns:
        str: Stable grouping key.
    """
    canonical_pair = canonicalize_pairwise_model_routing(
        model_a_name, model_b_name
    ).canonical_model_pair.split("_vs_")
    return "|".join(
        [
            persona,
            prompt_type,
            pairwise_judgment_type,
            canonical_pair[0],
            canonical_pair[1],
            task_id,
            variant_id,
        ]
    )


def discover_pairwise_candidates(
    config: HumanAnnotationConfig,
) -> List[PairwiseCandidateRecord]:
    """
    Discover and normalize candidate rows from existing Stage-5b artifacts.

    Args:
        config (HumanAnnotationConfig): Study configuration.

    Returns:
        List[PairwiseCandidateRecord]: Normalized candidate records.

    Raises:
        ValueError: If a discovered artifact is malformed.
    """
    candidates: List[PairwiseCandidateRecord] = []
    scanned_files = 0
    source_skip_counts: Counter[str] = Counter()
    for artifact_path in _iter_pairwise_artifacts(config.source.base_results_dir):
        scanned_files += 1
        parsed_path = parse_pairwise_artifact_path(
            config.source.base_results_dir, artifact_path
        )
        rejection_field = _source_rejection_field(parsed_path, config)
        if rejection_field is not None:
            source_skip_counts[rejection_field] += 1
            continue
        payload = load_pairwise_json(artifact_path)
        for artifact_index, entry in enumerate(payload):
            raw_task_id = str(entry["task_id"])
            task_id, variant_id = _resolve_variant_fields(raw_task_id)
            prompt_type = parsed_path.prompt_type or infer_prompt_type(raw_task_id)
            pairwise_judgment_type = normalize_pairwise_judgment_type(
                str(
                    (entry.get("_model_metadata") or {}).get("pairwise_judgment_type")
                    or entry.get("pairwise_judgment_type")
                    or parsed_path.pairwise_judgment_type
                    or PAIRWISE_JUDGMENT_TYPE_PERSONA
                )
            )
            candidate = PairwiseCandidateRecord(
                source_key=_build_source_key(
                    persona=parsed_path.persona,
                    prompt_type=prompt_type,
                    pairwise_judgment_type=pairwise_judgment_type,
                    model_a_name=str(entry["model_a_name"]),
                    model_b_name=str(entry["model_b_name"]),
                    task_id=task_id,
                    variant_id=variant_id,
                ),
                artifact_path=str(artifact_path),
                artifact_index=artifact_index,
                persona=parsed_path.persona,
                prompt_type=prompt_type,
                pairwise_judgment_type=pairwise_judgment_type,
                judge_dir_name=parsed_path.judge_dir_name,
                judge_model_name=str(
                    entry.get("judge_model_name") or parsed_path.judge_dir_name
                ),
                generator_model=parsed_path.generator_model,
                filter_model=parsed_path.filter_model,
                model_a_name=str(entry["model_a_name"]),
                model_b_name=str(entry["model_b_name"]),
                model_pair=_canonical_model_pair_token(
                    str(entry["model_a_name"]), str(entry["model_b_name"])
                ),
                task_id=task_id,
                variant_id=variant_id,
                raw_task_id=raw_task_id,
                input_text=str(entry.get("input_text") or ""),
                model_a_output=str(entry.get("model_a_output") or ""),
                model_b_output=str(entry.get("model_b_output") or ""),
                overall_winner=entry.get("overall_winner"),
                dimension_results=dict(entry.get("dimension_results") or {}),
                win_counts=dict(entry.get("win_counts") or {}),
                metadata=dict(entry.get("metadata") or {}),
            )
            candidates.append(candidate)
    source_item_count = len({candidate.source_key for candidate in candidates})
    logger.info(
        "Discovery summary: scanned_files=%d kept_rows=%d kept_items=%d skipped_by_allowlist=%s",
        scanned_files,
        len(candidates),
        source_item_count,
        dict(source_skip_counts),
    )
    logger.info(
        "Discovery composition: personas=%s prompt_types=%s model_pairs=%s judges=%s",
        dict(Counter(candidate.persona for candidate in candidates)),
        dict(Counter(candidate.prompt_type for candidate in candidates)),
        dict(Counter(candidate.model_pair for candidate in candidates)),
        dict(Counter(candidate.judge_dir_name for candidate in candidates)),
    )
    logger.info(
        "Discovered %d pairwise candidate records from %s",
        len(candidates),
        config.source.base_results_dir,
    )
    return candidates
