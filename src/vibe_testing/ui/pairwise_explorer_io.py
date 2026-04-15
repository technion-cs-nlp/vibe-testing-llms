"""
Pairwise exploration I/O helpers.

This module discovers and loads Stage 5b pairwise-comparison artifacts from the
canonical directory structure defined in `src/vibe_testing/pathing.py`.

The functions here are intentionally strict (fail-fast) to avoid quiet failures
when artifacts are missing or malformed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.vibe_testing.pairwise_artifact_diagnostics import (
    PairwiseArtifactLoadContext,
    PairwiseArtifactLoadError,
    wrap_pairwise_artifact_load_error,
)
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_GENERAL_USER,
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
)
from src.vibe_testing.pathing import (
    canonicalize_pairwise_model_routing,
    normalize_token,
    parse_artifact_name,
)
from src.vibe_testing.utils import load_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairwiseConfigKey:
    """
    A discovered configuration slice for pairwise exploration.

    Attributes:
        persona: Persona directory name under the results base directory.
        generator_model: `gen_model_<name>` segment value.
        filter_model: `filter_model_<name>` segment value.
        prompt_type: Prompt type string when results are under `prompt_type_<x>/`.
            When None, indicates the legacy/root layout (no prompt_type subdir).
    """

    persona: str
    generator_model: str
    filter_model: str
    prompt_type: Optional[str]
    pairwise_judgment_type: str = PAIRWISE_JUDGMENT_TYPE_PERSONA


@dataclass(frozen=True)
class ModelPairKey:
    """
    A model pair discovered under `5b_pairwise_evaluation/`.

    Attributes:
        model_a: Left-hand model token parsed from `models_<a>_vs_<b>`.
        model_b: Right-hand model token parsed from `models_<a>_vs_<b>`.
    """

    model_a: str
    model_b: str

    @property
    def label(self) -> str:
        """Return a stable display label for the pair."""
        return f"{self.model_a}_vs_{self.model_b}"


def format_prompt_type(prompt_type: Optional[str]) -> str:
    """
    Format a prompt type for UI display.

    Args:
        prompt_type: Optional prompt type.

    Returns:
        Display-friendly prompt type string.
    """

    return prompt_type or "legacy"


def parse_model_pair_dirname(dirname: str) -> Optional[ModelPairKey]:
    """
    Parse a `models_<a>_vs_<b>` directory name.

    Args:
        dirname: Directory name.

    Returns:
        Parsed ModelPairKey, or None when the name does not match.
    """

    if not dirname.startswith("models_"):
        return None
    pair_part = dirname.replace("models_", "", 1)
    if "_vs_" not in pair_part:
        return None
    model_a, model_b = pair_part.split("_vs_", 1)
    if not model_a or not model_b:
        return None
    return ModelPairKey(model_a=model_a, model_b=model_b)


def _iter_persona_dirs(base_dir: Path) -> Iterable[Path]:
    """
    Yield candidate persona directories under the base results directory.

    Args:
        base_dir: Root results directory containing persona subdirectories.

    Yields:
        Persona directories (paths) that exist.
    """

    if not base_dir.exists():
        raise FileNotFoundError(f"Base results directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise ValueError(f"Base results path must be a directory: {base_dir}")

    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir():
            yield entry


def _iter_pairwise_persona_dirs(base_dir: Path) -> Iterable[Path]:
    """
    Yield persona directories that actually contain pairwise evaluation roots.

    Args:
        base_dir: Root results directory containing persona subdirectories.

    Yields:
        Persona directories with an existing ``5b_pairwise_evaluation`` directory.
    """
    for entry in _iter_persona_dirs(base_dir):
        pairwise_root = entry / "5b_pairwise_evaluation"
        if pairwise_root.exists() and pairwise_root.is_dir():
            yield entry


def discover_pairwise_configs(base_dir: Path) -> List[PairwiseConfigKey]:
    """
    Discover all available pairwise configuration slices on disk.

    A configuration is defined as:
        (persona, generator_model, filter_model, prompt_type)

    Args:
        base_dir: Root results directory (e.g. `runs/` or `Runs/`).

    Returns:
        Sorted list of discovered configuration keys.
    """

    discovered: set[PairwiseConfigKey] = set()

    for persona_dir in _iter_pairwise_persona_dirs(base_dir):
        pairwise_root = persona_dir / "5b_pairwise_evaluation"
        persona_name = persona_dir.name
        for model_pair_dir in sorted(pairwise_root.iterdir()):
            if not model_pair_dir.is_dir():
                continue
            if parse_model_pair_dirname(model_pair_dir.name) is None:
                continue

            for judge_dir in sorted(model_pair_dir.iterdir()):
                if not judge_dir.is_dir() or not judge_dir.name.startswith(
                    "judge_model_"
                ):
                    continue

                for gen_dir in sorted(judge_dir.iterdir()):
                    if not gen_dir.is_dir() or not gen_dir.name.startswith(
                        "gen_model_"
                    ):
                        continue
                    generator_model = gen_dir.name.replace("gen_model_", "", 1)

                    for filter_dir in sorted(gen_dir.iterdir()):
                        if not filter_dir.is_dir() or not filter_dir.name.startswith(
                            "filter_model_"
                        ):
                            continue
                        filter_model = filter_dir.name.replace("filter_model_", "", 1)
                        judgment_dirs = [
                            d
                            for d in sorted(filter_dir.iterdir())
                            if d.is_dir() and d.name.startswith("judgment_type_")
                        ]
                        explicit_persona_dir = filter_dir / (
                            f"judgment_type_{PAIRWISE_JUDGMENT_TYPE_PERSONA}"
                        )
                        use_root_persona_fallback = not explicit_persona_dir.exists()
                        root_prompt_dirs = [
                            d.name
                            for d in sorted(filter_dir.iterdir())
                            if d.is_dir() and d.name.startswith("prompt_type_")
                        ]
                        if explicit_persona_dir.exists() and root_prompt_dirs:
                            logger.debug(
                                "Ignoring root-level persona fallback during config discovery because explicit persona branch exists: filter_dir=%s explicit_persona_dir=%s root_prompt_dirs=%s",
                                filter_dir,
                                explicit_persona_dir,
                                root_prompt_dirs,
                            )
                        logger.debug(
                            "Discovering pairwise configs for filter_dir=%s judgment_dirs=%s root_prompt_dirs=%s root_persona_fallback=%s",
                            filter_dir,
                            [d.name for d in judgment_dirs],
                            root_prompt_dirs,
                            use_root_persona_fallback,
                        )

                        for judgment_dir in judgment_dirs:
                            judgment_type = judgment_dir.name.replace(
                                "judgment_type_", "", 1
                            )
                            prompt_dirs = [
                                d
                                for d in sorted(judgment_dir.iterdir())
                                if d.is_dir() and d.name.startswith("prompt_type_")
                            ]
                            if prompt_dirs:
                                for prompt_dir in prompt_dirs:
                                    prompt_type = prompt_dir.name.replace(
                                        "prompt_type_", "", 1
                                    )
                                    if _has_pairwise_artifact(prompt_dir):
                                        discovered.add(
                                            PairwiseConfigKey(
                                                persona=persona_name,
                                                generator_model=generator_model,
                                                filter_model=filter_model,
                                                prompt_type=prompt_type,
                                                pairwise_judgment_type=judgment_type,
                                            )
                                        )
                                        if (
                                            judgment_type
                                            == PAIRWISE_JUDGMENT_TYPE_GENERAL_USER
                                            and prompt_type in {"original", "control"}
                                        ):
                                            for persona_alias in _iter_pairwise_persona_dirs(
                                                base_dir
                                            ):
                                                discovered.add(
                                                    PairwiseConfigKey(
                                                        persona=persona_alias.name,
                                                        generator_model=generator_model,
                                                        filter_model=filter_model,
                                                        prompt_type=prompt_type,
                                                        pairwise_judgment_type=judgment_type,
                                                    )
                                                )
                            elif _has_pairwise_artifact(judgment_dir):
                                discovered.add(
                                    PairwiseConfigKey(
                                        persona=persona_name,
                                        generator_model=generator_model,
                                        filter_model=filter_model,
                                        prompt_type=None,
                                        pairwise_judgment_type=judgment_type,
                                    )
                                )

                        if use_root_persona_fallback:
                            prompt_dirs = [
                                d
                                for d in sorted(filter_dir.iterdir())
                                if d.is_dir() and d.name.startswith("prompt_type_")
                            ]
                            if prompt_dirs:
                                for prompt_dir in prompt_dirs:
                                    prompt_type = prompt_dir.name.replace(
                                        "prompt_type_", "", 1
                                    )
                                    if _has_pairwise_artifact(prompt_dir):
                                        discovered.add(
                                            PairwiseConfigKey(
                                                persona=persona_name,
                                                generator_model=generator_model,
                                                filter_model=filter_model,
                                                prompt_type=prompt_type,
                                                pairwise_judgment_type=PAIRWISE_JUDGMENT_TYPE_PERSONA,
                                            )
                                        )
                            elif _has_pairwise_artifact(filter_dir):
                                discovered.add(
                                    PairwiseConfigKey(
                                        persona=persona_name,
                                        generator_model=generator_model,
                                        filter_model=filter_model,
                                        prompt_type=None,
                                        pairwise_judgment_type=PAIRWISE_JUDGMENT_TYPE_PERSONA,
                                    )
                                )

    def _sort_key(cfg: PairwiseConfigKey) -> Tuple[str, str, str, str, str]:
        return (
            normalize_token(cfg.persona),
            normalize_token(cfg.generator_model),
            normalize_token(cfg.filter_model),
            normalize_token(format_prompt_type(cfg.prompt_type)),
            normalize_token(cfg.pairwise_judgment_type),
        )

    return sorted(discovered, key=_sort_key)


def discover_model_pairs(
    base_dir: Path, config: PairwiseConfigKey
) -> List[ModelPairKey]:
    """
    Discover model pairs available for a given configuration.

    Args:
        base_dir: Root results directory.
        config: Configuration slice to filter by.

    Returns:
        Sorted list of model pairs that have at least one artifact for at least
        one judge.
    """

    pairwise_root = _resolve_pairwise_root_for_config(base_dir, config)
    if not pairwise_root.exists():
        raise FileNotFoundError(
            f"Pairwise root not found for persona '{config.persona}': {pairwise_root}"
        )

    pairs_by_canonical_key: Dict[str, ModelPairKey] = {}
    for model_pair_dir in sorted(pairwise_root.iterdir()):
        if not model_pair_dir.is_dir():
            continue
        parsed = parse_model_pair_dirname(model_pair_dir.name)
        if parsed is None:
            continue
        if _pair_has_any_artifacts_for_config(model_pair_dir, config):
            canonical_key = canonicalize_pairwise_model_routing(
                parsed.model_a,
                parsed.model_b,
                route_model_a=parsed.model_a,
                route_model_b=parsed.model_b,
            ).canonical_model_pair
            pairs_by_canonical_key.setdefault(canonical_key, parsed)

    return sorted(
        pairs_by_canonical_key.values(),
        key=lambda p: (normalize_token(p.model_a), normalize_token(p.model_b)),
    )


def discover_judges_for_pair(
    base_dir: Path,
    config: PairwiseConfigKey,
    pair: ModelPairKey,
) -> List[str]:
    """
    Discover judge model names available for a given (config, pair).

    Args:
        base_dir: Root results directory.
        config: Configuration slice.
        pair: Selected model pair.

    Returns:
        Sorted list of judge model tokens (directory suffixes after `judge_model_`).
    """

    model_pair_dir = _resolve_model_pair_dir_for_config(base_dir, config, pair)
    judges: set[str] = set()
    for judge_dir in sorted(model_pair_dir.iterdir()):
        if not judge_dir.is_dir() or not judge_dir.name.startswith("judge_model_"):
            continue
        judge = judge_dir.name.replace("judge_model_", "", 1)
        artifact_dir = _resolve_artifact_dir_for_config(judge_dir, config)
        if artifact_dir is None:
            continue
        if _has_pairwise_artifact(artifact_dir):
            judges.add(judge)
    return sorted(judges, key=normalize_token)


def list_pairwise_artifacts_for_pair_and_judge(
    base_dir: Path,
    config: PairwiseConfigKey,
    pair: ModelPairKey,
    judge_model: str,
) -> List[Path]:
    """
    List all pairwise-comparison artifacts for a specific judge.

    Args:
        base_dir: Root results directory.
        config: Configuration slice.
        pair: Selected model pair.
        judge_model: Judge model token (without `judge_model_` prefix).

    Returns:
        List of artifact paths. May be empty if none exist.
    """

    model_pair_dir = _resolve_model_pair_dir_for_config(base_dir, config, pair)
    judge_dir = model_pair_dir / f"judge_model_{judge_model}"
    if not judge_dir.exists():
        logger.debug(
            "No pairwise judge directory found: base_dir=%s persona=%s pair=%s judge=%s path=%s",
            base_dir,
            config.persona,
            pair.label,
            judge_model,
            judge_dir,
        )
        return []
    artifact_dir = _resolve_artifact_dir_for_config(judge_dir, config)
    if artifact_dir is None or not artifact_dir.exists():
        logger.debug(
            "No pairwise artifact directory found: persona=%s pair=%s judge=%s config_prompt=%s judgment_type=%s path=%s",
            config.persona,
            pair.label,
            judge_model,
            config.prompt_type,
            config.pairwise_judgment_type,
            artifact_dir,
        )
        return []
    artifacts = sorted(_iter_pairwise_artifacts(artifact_dir))
    if not artifacts:
        logger.debug(
            "Artifact directory exists but no pairwise JSON files were found: persona=%s pair=%s judge=%s path=%s",
            config.persona,
            pair.label,
            judge_model,
            artifact_dir,
        )
    else:
        logger.debug(
            "Discovered %d pairwise artifact candidate(s): persona=%s pair=%s judge=%s path=%s",
            len(artifacts),
            config.persona,
            pair.label,
            judge_model,
            artifact_dir,
        )
    return artifacts


def select_latest_pairwise_artifact(paths: Sequence[Path]) -> Path:
    """
    Select the best candidate pairwise artifact from a list.

    Preference order:
    1) Highest parsed artifact version (`vNN`) when the filename matches the
       canonical artifact naming convention.
    2) Most recent parsed timestamp (if present).
    3) Most recently modified file (mtime) as a fallback.

    Args:
        paths: Candidate artifact paths.

    Returns:
        Selected artifact path.

    Raises:
        ValueError: If paths is empty.
    """

    if not paths:
        raise ValueError("No candidate pairwise artifact paths provided.")

    scored: List[Tuple[Tuple[int, str, float], Path]] = []
    used_mtime_fallback = False
    for path in paths:
        version = -1
        ts = ""
        try:
            parsed = parse_artifact_name(path.name)
            version = int(parsed.version)
            ts = parsed.timestamp or ""
        except Exception:
            # Non-canonical filename: rely on mtime only.
            used_mtime_fallback = True
        try:
            mtime = float(path.stat().st_mtime)
        except FileNotFoundError:
            mtime = 0.0
        scored.append(((version, ts, mtime), path))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[0][1]
    if len(paths) > 1:
        logger.debug(
            "Selected latest pairwise artifact from %d candidate(s): selected=%s candidates=%s",
            len(paths),
            selected,
            [str(path) for path in paths],
        )
    if used_mtime_fallback:
        logger.debug(
            "Pairwise artifact selection used mtime fallback due to non-canonical filename(s): selected=%s",
            selected,
        )
    return selected


def load_pairwise_json(path: Path) -> List[Dict[str, Any]]:
    """
    Load a Stage 5b pairwise artifact JSON.

    Args:
        path: Path to a `pairwise-comparison_*.json` artifact.

    Returns:
        List of pairwise comparison records.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the payload shape is invalid.
    """

    if not path.exists():
        raise PairwiseArtifactLoadError(
            context=PairwiseArtifactLoadContext(
                artifact_path=str(path),
                failure_stage="path_validation",
            ),
            message="Pairwise artifact not found.",
        )
    if not path.is_file():
        raise PairwiseArtifactLoadError(
            context=PairwiseArtifactLoadContext(
                artifact_path=str(path),
                failure_stage="path_validation",
            ),
            message="Expected a file path, got directory.",
        )

    try:
        payload = load_json(str(path))
    except Exception as exc:
        raise wrap_pairwise_artifact_load_error(
            exc,
            context=PairwiseArtifactLoadContext(
                artifact_path=str(path),
                failure_stage="load_json",
            ),
            message="Failed to read pairwise artifact JSON.",
        ) from exc
    if isinstance(payload, dict):
        payload = payload.get("results", [payload])

    if not isinstance(payload, list):
        raise PairwiseArtifactLoadError(
            context=PairwiseArtifactLoadContext(
                artifact_path=str(path),
                failure_stage="payload_shape",
                payload_type=type(payload).__name__,
            ),
            message="Pairwise artifact must be a JSON list (or dict with 'results').",
        )

    records: List[Dict[str, Any]] = []
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise PairwiseArtifactLoadError(
                context=PairwiseArtifactLoadContext(
                    artifact_path=str(path),
                    failure_stage="record_shape",
                    record_index=idx,
                    record_count=len(payload),
                    payload_type=type(payload).__name__,
                ),
                message="Pairwise record is not a JSON object.",
                cause_type=type(entry).__name__,
                cause_message=repr(entry),
            )
        try:
            _validate_pairwise_record(entry, path=path, index=idx)
        except Exception as exc:
            raise wrap_pairwise_artifact_load_error(
                exc,
                context=PairwiseArtifactLoadContext(
                    artifact_path=str(path),
                    failure_stage="schema_validation",
                    record_index=idx,
                    record_count=len(payload),
                    payload_type=type(payload).__name__,
                    task_id=str(entry.get("task_id") or "") or None,
                    model_a_name=str(entry.get("model_a_name") or "") or None,
                    model_b_name=str(entry.get("model_b_name") or "") or None,
                    judge_model_name=str(entry.get("judge_model_name") or "") or None,
                ),
                message="Pairwise artifact record failed schema validation.",
            ) from exc
        records.append(entry)

    if not records:
        raise PairwiseArtifactLoadError(
            context=PairwiseArtifactLoadContext(
                artifact_path=str(path),
                failure_stage="empty_payload",
                record_count=0,
                payload_type=type(payload).__name__,
            ),
            message="Pairwise artifact contains no records.",
        )

    return records


def index_records_by_task_id(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Index pairwise records by `task_id`.

    Args:
        records: List of pairwise records.

    Returns:
        Dict mapping task_id -> record.

    Raises:
        ValueError: If task_id is missing or duplicated.
    """

    indexed: Dict[str, Dict[str, Any]] = {}
    for entry in records:
        task_id = entry.get("task_id")
        if not task_id:
            raise ValueError("Pairwise record is missing 'task_id'.")
        task_id_str = str(task_id)
        if task_id_str in indexed:
            raise ValueError(f"Duplicate task_id in pairwise artifact: {task_id_str!r}")
        indexed[task_id_str] = dict(entry)
    return indexed


def win_counts_by_model_name(record: Dict[str, Any]) -> Dict[str, int]:
    """
    Convert a Stage 5b record's win_counts into a model_name -> wins mapping.

    Stage 5b stores win counts as keys relative to the record's A/B positions:
    - win_counts['model_a'] counts wins for record['model_a_name']
    - win_counts['model_b'] counts wins for record['model_b_name']

    This helper projects those counts onto the *actual* model names so UIs and
    analysis code can display counts consistently even when different artifacts
    list the same unordered model pair in different A/B orders.

    Args:
        record: Stage 5b pairwise record dict.

    Returns:
        Dict mapping model name -> integer win count. Missing/invalid win_counts
        entries are treated as 0.

    Raises:
        ValueError: If model_a_name/model_b_name are missing.
    """
    model_a = record.get("model_a_name")
    model_b = record.get("model_b_name")
    if not model_a or not model_b:
        raise ValueError(
            "Pairwise record is missing model_a_name/model_b_name required to map win_counts."
        )
    win_counts = record.get("win_counts") or {}
    if not isinstance(win_counts, dict):
        win_counts = {}

    def _as_int(x: object) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    return {
        str(model_a): _as_int(win_counts.get("model_a", 0)),
        str(model_b): _as_int(win_counts.get("model_b", 0)),
    }


def build_task_id_union(
    index_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    """
    Build a stable, sorted union of task_ids across judges.

    Args:
        index_by_judge: Mapping judge -> (task_id -> record).

    Returns:
        Sorted list of task_ids.
    """

    task_ids: set[str] = set()
    for indexed in index_by_judge.values():
        task_ids.update(indexed.keys())
    return sorted(task_ids)


def stage6_sample_id_from_row(row: Dict[str, Any]) -> str:
    """
    Build a stable per-sample identifier from a Stage-6-normalized pairwise row.

    Stage 6 resolves variant info from the original `task_id` and preserves the
    original identifier in `raw_task_id`. For UI purposes, we prefer `raw_task_id`
    when available; otherwise fall back to `(task_id, variant_id)` pairing.

    Args:
        row: A dict-like row from `AnalysisInputLoader.load_pairwise_results(...)`.

    Returns:
        str: Stable per-sample id suitable for indexing and filtering.
    """

    raw = row.get("raw_task_id")
    if raw is not None and str(raw).strip():
        return str(raw)
    task_id = str(row.get("task_id", "") or "")
    variant_id = str(row.get("variant_id", "") or "")
    if task_id and variant_id:
        return f"{task_id}::{variant_id}"
    if task_id:
        return task_id
    return "unknown_sample"


def stage6_display_winner_name_from_row(row: Dict[str, Any]) -> str:
    """
    Convert Stage-6 `overall_winner_label` into a display winner name.

    Stage 6 stores winners as labels relative to the (canonicalized) A/B ordering:
      - 'model_a' means model_a_name wins
      - 'model_b' means model_b_name wins
      - 'tie' means tie

    Args:
        row: A dict-like row from `AnalysisInputLoader.load_pairwise_results(...)`.

    Returns:
        str: Winner model name or 'tie'.
    """

    label = str(row.get("overall_winner_label", "tie") or "tie").strip()
    if label == "model_a":
        return str(row.get("model_a_name", "model_a"))
    if label == "model_b":
        return str(row.get("model_b_name", "model_b"))
    return "tie"


def _validate_pairwise_record(entry: Dict[str, Any], path: Path, index: int) -> None:
    """
    Validate the minimum expected schema for a Stage 5b pairwise record.

    Args:
        entry: Pairwise record payload.
        path: Artifact file path (for error context).
        index: Record index (for error context).

    Raises:
        ValueError: If required fields are missing or malformed.
    """

    required_top_level = [
        "task_id",
        "input_text",
        "model_a_name",
        "model_b_name",
        "model_a_output",
        "model_b_output",
        "judge_model_name",
        "dimension_results",
    ]
    missing = [k for k in required_top_level if k not in entry]
    if missing:
        raise ValueError(
            f"Missing required fields in pairwise record index={index} file={path}: {missing}"
        )

    dim = entry.get("dimension_results")
    if not isinstance(dim, dict):
        raise ValueError(
            f"dimension_results must be an object in record index={index} file={path}."
        )

    # Ensure each dimension payload is dict-like when present.
    for dim_name, dim_payload in dim.items():
        if not isinstance(dim_payload, dict):
            raise ValueError(
                f"dimension_results[{dim_name!r}] must be an object in record index={index} file={path}."
            )


def _iter_pairwise_artifacts(directory: Path) -> Iterable[Path]:
    """
    Yield `pairwise-comparison` artifact files under a directory.

    Args:
        directory: Directory to search (non-recursive).

    Yields:
        Paths to JSON files matching the artifact hint.
    """

    if not directory.exists() or not directory.is_dir():
        return
    for path in sorted(directory.glob("*pairwise-comparison*.json")):
        if path.is_file():
            yield path


def _has_pairwise_artifact(directory: Path) -> bool:
    """
    Check whether a directory contains at least one pairwise-comparison artifact.

    Args:
        directory: Directory to check.

    Returns:
        True if at least one artifact exists.
    """

    return any(_iter_pairwise_artifacts(directory))


def _resolve_config_base_dir(
    filter_dir: Path,
    config: PairwiseConfigKey,
    *,
    require_exists: bool,
) -> Optional[Path]:
    """
    Resolve the config-specific base directory under a filter-model directory.

    Persona-scoped artifacts prefer an explicit `judgment_type_persona` branch when
    present, and otherwise fall back to the legacy root layout directly under the
    filter directory.

    Args:
        filter_dir: Directory `.../filter_model_<f>`.
        config: Configuration slice.
        require_exists: Whether the returned directory must already exist.

    Returns:
        Resolved base directory, or None when `require_exists` is true and no valid
        directory exists.
    """

    if config.pairwise_judgment_type == PAIRWISE_JUDGMENT_TYPE_PERSONA:
        explicit_persona_dir = filter_dir / (
            f"judgment_type_{PAIRWISE_JUDGMENT_TYPE_PERSONA}"
        )
        base_dir = explicit_persona_dir if explicit_persona_dir.exists() else filter_dir
        if (
            explicit_persona_dir.exists()
            and filter_dir != explicit_persona_dir
            and any(
                entry.is_dir() and entry.name.startswith("prompt_type_")
                for entry in filter_dir.iterdir()
            )
        ):
            logger.debug(
                "Preferring explicit persona branch over root fallback: filter_dir=%s explicit_persona_dir=%s",
                filter_dir,
                explicit_persona_dir,
            )
    else:
        base_dir = filter_dir / f"judgment_type_{config.pairwise_judgment_type}"

    if require_exists and not base_dir.exists():
        return None
    return base_dir


def _resolve_model_pair_dir(base_dir: Path, persona: str, pair: ModelPairKey) -> Path:
    """
    Resolve the model-pair directory for a given persona.

    Args:
        base_dir: Root results directory.
        persona: Persona token.
        pair: Model pair key.

    Returns:
        Path to the model pair directory.

    Raises:
        FileNotFoundError: If the directory cannot be resolved.
    """

    persona_dir = base_dir / persona
    if not persona_dir.exists():
        persona_dir = base_dir / normalize_token(persona)
    root = persona_dir / "5b_pairwise_evaluation"
    return _find_matching_model_pair_dir(root, pair)


def _resolve_model_pair_dir_for_config(
    base_dir: Path,
    config: PairwiseConfigKey,
    pair: ModelPairKey,
) -> Path:
    """
    Resolve a model-pair directory using config-aware shared-artifact fallback.

    Args:
        base_dir: Root results directory.
        config: Pairwise configuration slice.
        pair: Model pair.

    Returns:
        Path: Existing model-pair directory.
    """
    root = _resolve_pairwise_root_for_config(base_dir, config)
    return _find_matching_model_pair_dir(root, pair)


def _resolve_pairwise_root_for_config(
    base_dir: Path, config: PairwiseConfigKey
) -> Path:
    """
    Resolve the pairwise root directory for a configuration, including shared fallback.

    Args:
        base_dir: Root results directory.
        config: Pairwise configuration slice.

    Returns:
        Path: Resolved pairwise root.
    """
    def _root_has_any_matching_artifacts(root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        for model_pair_dir in sorted(root.iterdir()):
            if not model_pair_dir.is_dir():
                continue
            if parse_model_pair_dirname(model_pair_dir.name) is None:
                continue
            for judge_dir in sorted(model_pair_dir.iterdir()):
                if not judge_dir.is_dir() or not judge_dir.name.startswith("judge_model_"):
                    continue
                gen_dir = judge_dir / f"gen_model_{config.generator_model}"
                filter_dir = gen_dir / f"filter_model_{config.filter_model}"
                if not filter_dir.exists():
                    continue
                candidate_dir = _resolve_config_base_dir(
                    filter_dir, config, require_exists=False
                )
                if not candidate_dir.exists():
                    continue
                if config.prompt_type:
                    candidate_dir = candidate_dir / f"prompt_type_{config.prompt_type}"
                if candidate_dir.exists() and _has_pairwise_artifact(candidate_dir):
                    return True
        return False

    persona_dir = base_dir / config.persona
    if not persona_dir.exists():
        persona_dir = base_dir / normalize_token(config.persona)
    pairwise_root = persona_dir / "5b_pairwise_evaluation"
    if pairwise_root.exists() and (
        config.pairwise_judgment_type != PAIRWISE_JUDGMENT_TYPE_GENERAL_USER
        or config.prompt_type not in {"original", "control"}
        or _root_has_any_matching_artifacts(pairwise_root)
    ):
        return pairwise_root

    if (
        config.pairwise_judgment_type == PAIRWISE_JUDGMENT_TYPE_GENERAL_USER
        and config.prompt_type in {"original", "control"}
    ):
        for candidate in _iter_pairwise_persona_dirs(base_dir):
            fallback_root = candidate / "5b_pairwise_evaluation"
            if _root_has_any_matching_artifacts(fallback_root):
                return fallback_root
    return pairwise_root


def _find_matching_model_pair_dir(root: Path, pair: ModelPairKey) -> Path:
    """
    Find the on-disk model-pair directory matching a requested unordered pair.

    Args:
        root: Pairwise root directory containing `models_*_vs_*` children.
        pair: Requested model pair.

    Returns:
        Path: Matching model-pair directory.

    Raises:
        FileNotFoundError: If no matching directory exists.
    """
    direct_dir = root / f"models_{pair.model_a}_vs_{pair.model_b}"
    if direct_dir.exists():
        return direct_dir

    requested_routing = canonicalize_pairwise_model_routing(
        pair.model_a,
        pair.model_b,
        route_model_a=pair.model_a,
        route_model_b=pair.model_b,
    )
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        parsed_pair = parse_model_pair_dirname(candidate.name)
        if parsed_pair is None:
            continue
        candidate_routing = canonicalize_pairwise_model_routing(
            parsed_pair.model_a,
            parsed_pair.model_b,
            route_model_a=parsed_pair.model_a,
            route_model_b=parsed_pair.model_b,
        )
        if (
            candidate_routing.canonical_model_pair
            == requested_routing.canonical_model_pair
        ):
            return candidate
    raise FileNotFoundError(
        f"Model pair directory not found under {root} for pair {pair.label}"
    )


def _resolve_artifact_dir_for_config(
    judge_dir: Path, config: PairwiseConfigKey
) -> Optional[Path]:
    """
    Resolve the expected artifact directory for a config slice under a judge dir.

    Args:
        judge_dir: Directory `.../judge_model_<j>`.
        config: Configuration slice.

    Returns:
        Artifact directory if it exists, otherwise None.
    """

    gen_dir = judge_dir / f"gen_model_{config.generator_model}"
    filter_dir = gen_dir / f"filter_model_{config.filter_model}"
    if not filter_dir.exists():
        return None
    base_dir = _resolve_config_base_dir(filter_dir, config, require_exists=True)
    if base_dir is None:
        return None
    if config.prompt_type:
        prompt_dir = base_dir / f"prompt_type_{config.prompt_type}"
        return prompt_dir if prompt_dir.exists() else None
    return base_dir


def _pair_has_any_artifacts_for_config(
    model_pair_dir: Path, config: PairwiseConfigKey
) -> bool:
    """
    Check if any judge under the model pair directory has artifacts for config.

    Args:
        model_pair_dir: Directory `.../models_<a>_vs_<b>`.
        config: Configuration slice.

    Returns:
        True if any judge has at least one artifact for this config.
    """

    for judge_dir in sorted(model_pair_dir.iterdir()):
        if not judge_dir.is_dir() or not judge_dir.name.startswith("judge_model_"):
            continue
        artifact_dir = _resolve_artifact_dir_for_config(judge_dir, config)
        if artifact_dir is None:
            continue
        if _has_pairwise_artifact(artifact_dir):
            return True
    return False
