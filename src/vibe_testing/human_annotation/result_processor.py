"""Result processing for standalone human-annotation studies."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.vibe_testing.human_annotation.qualtrics_generator import (
    CONFIDENCE_PROMPT,
    MATRIX_PROMPT,
    OVERALL_PROMPT,
    RATIONALE_PROMPT,
    _question_ids,
)
from src.vibe_testing.human_annotation.schemas import (
    AnnotatorAssignment,
    HumanAnnotationConfig,
    ProcessedAnnotationRecord,
)
from src.vibe_testing.utils import load_json

logger = logging.getLogger(__name__)

_LEGACY_OVERALL_PROMPT = "Which response is better overall for this persona?"
_LEGACY_MATRIX_PROMPT = (
    "Which response is better for this persona for the following dimensions?"
)
_OVERALL_PROMPT_ALIASES = (OVERALL_PROMPT, _LEGACY_OVERALL_PROMPT)
_MATRIX_PROMPT_ALIASES = (MATRIX_PROMPT, _LEGACY_MATRIX_PROMPT)
_CONFIDENCE_PROMPT_ALIASES = (CONFIDENCE_PROMPT,)
_RATIONALE_PROMPT_ALIASES = (RATIONALE_PROMPT,)


@dataclass(frozen=True)
class _QualtricsColumn:
    """Normalized metadata for one Qualtrics CSV column."""

    index: int
    raw_header: str
    label_header: str
    import_id: str
    normalized_name: str


@dataclass(frozen=True)
class _NormalizedQualtricsExport:
    """Normalized Qualtrics export with response rows and column metadata."""

    path: Path
    response_df: pd.DataFrame
    columns: List[_QualtricsColumn]
    response_row_count: int


@dataclass(frozen=True)
class _SelectedSubmission:
    """Best available submission row for one annotator."""

    annotator_id: str
    export_path: Path
    row_index: int
    response_row: pd.Series
    columns: List[_QualtricsColumn]
    response_id: str


def _safe_str(value: Any) -> str:
    """
    Convert a CSV value to a normalized string.

    Args:
        value (Any): Arbitrary CSV cell value.

    Returns:
        str: Stripped string, or empty string for missing values.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_header_name(name: str) -> str:
    """
    Normalize a header token into an identifier-safe column name.

    Args:
        name (str): Raw header token.

    Returns:
        str: Normalized identifier.
    """
    text = "".join(ch if ch.isalnum() else "_" for ch in _safe_str(name))
    text = "_".join(part for part in text.split("_") if part)
    return (text or "column").lower()


def _parse_import_id(cell: str) -> str:
    """
    Parse an ImportId JSON cell from a Qualtrics export metadata row.

    Args:
        cell (str): Raw cell text.

    Returns:
        str: Parsed ImportId value, or empty string if unavailable.
    """
    text = _safe_str(cell)
    if not text.startswith("{"):
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return _safe_str(payload.get("ImportId"))


def _looks_like_import_id_row(row: pd.Series) -> bool:
    """
    Detect whether a CSV row is a Qualtrics ImportId metadata row.

    Args:
        row (pandas.Series): Raw row values.

    Returns:
        bool: True when the row looks like ImportId JSON metadata.
    """
    values = [_safe_str(value) for value in row]
    parsed = [value for value in values if _parse_import_id(value)]
    return bool(parsed) and len(parsed) >= max(1, len(values) // 3)


def _looks_like_label_row(row: pd.Series) -> bool:
    """
    Detect whether a CSV row contains human-readable Qualtrics labels.

    Args:
        row (pandas.Series): Raw row values.

    Returns:
        bool: True when the row looks like a label header row.
    """
    values = [_safe_str(value) for value in row]
    signals = 0
    signals += int(
        any(
            "Which response is better" in value
            or "Which response better serves" in value
            for value in values
        )
    )
    signals += int(any(value == "Start Date" for value in values))
    signals += int(any(value == "Response ID" for value in values))
    signals += int(any(value == "annotator_id" for value in values))
    return signals >= 2


def _build_normalized_columns(
    raw_headers: Sequence[str],
    label_headers: Sequence[str],
    import_ids: Sequence[str],
) -> List[_QualtricsColumn]:
    """
    Build normalized column metadata for a Qualtrics export.

    Args:
        raw_headers (Sequence[str]): First-row machine headers.
        label_headers (Sequence[str]): Optional human-readable labels.
        import_ids (Sequence[str]): Optional ImportId tokens.

    Returns:
        List[_QualtricsColumn]: Column metadata.
    """
    seen_counts: Dict[str, int] = {}
    columns: List[_QualtricsColumn] = []
    for index, raw_header in enumerate(raw_headers):
        label_header = label_headers[index] if index < len(label_headers) else ""
        import_id = import_ids[index] if index < len(import_ids) else ""
        base_name = import_id or raw_header or label_header or f"column_{index}"
        normalized = _normalize_header_name(base_name)
        count = seen_counts.get(normalized, 0)
        seen_counts[normalized] = count + 1
        if count > 0:
            normalized = f"{normalized}__{count + 1}"
        columns.append(
            _QualtricsColumn(
                index=index,
                raw_header=_safe_str(raw_header),
                label_header=_safe_str(label_header),
                import_id=_safe_str(import_id),
                normalized_name=normalized,
            )
        )
    return columns


def _normalize_qualtrics_export(path: str | Path) -> _NormalizedQualtricsExport:
    """
    Normalize one Qualtrics CSV export into response rows plus metadata.

    Args:
        path (str | Path): CSV export path.

    Returns:
        _NormalizedQualtricsExport: Normalized export payload.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Response CSV not found: {resolved}")
    raw = pd.read_csv(resolved, dtype=str, header=None).fillna("")
    if raw.empty:
        logger.info("Skipped empty response CSV %s", resolved)
        return _NormalizedQualtricsExport(
            path=resolved, response_df=pd.DataFrame(), columns=[], response_row_count=0
        )

    raw_headers = [_safe_str(value) for value in raw.iloc[0]]
    cursor = 1
    label_headers = list(raw_headers)
    if len(raw) > cursor and _looks_like_label_row(raw.iloc[cursor]):
        label_headers = [_safe_str(value) for value in raw.iloc[cursor]]
        cursor += 1
    import_ids = [""] * len(raw_headers)
    if len(raw) > cursor and _looks_like_import_id_row(raw.iloc[cursor]):
        import_ids = [_parse_import_id(value) for value in raw.iloc[cursor]]
        cursor += 1

    columns = _build_normalized_columns(raw_headers, label_headers, import_ids)
    response_df = raw.iloc[cursor:].reset_index(drop=True).copy()
    response_df.columns = [column.normalized_name for column in columns]

    logger.info(
        "Normalized Qualtrics CSV %s: raw_rows=%d response_rows=%d label_row=%s import_id_row=%s",
        resolved,
        len(raw),
        len(response_df),
        label_headers != raw_headers,
        any(column.import_id for column in columns),
    )
    return _NormalizedQualtricsExport(
        path=resolved,
        response_df=response_df,
        columns=columns,
        response_row_count=len(response_df),
    )


def _load_manifest_assignments(manifest_path: str | Path) -> List[AnnotatorAssignment]:
    """
    Load annotator assignments from a selection manifest.

    Args:
        manifest_path (str | Path): Manifest path created during Qualtrics generation.

    Returns:
        List[AnnotatorAssignment]: Restored annotator assignments.
    """
    payload = load_json(str(Path(manifest_path).expanduser().resolve()))
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Selection manifest must contain an 'assignments' list.")
    restored: List[AnnotatorAssignment] = []
    for assignment in assignments:
        restored.append(
            AnnotatorAssignment(
                annotator_id=str(assignment["annotator_id"]),
                item=assignment["item"],
                assignment_rank=int(assignment["assignment_rank"]),
                presentation_swapped=bool(assignment["presentation_swapped"]),
                assignment_role=str(assignment["assignment_role"]),
            )
        )
    logger.info(
        "Loaded %d assignment rows from manifest %s",
        len(restored),
        Path(manifest_path).expanduser().resolve(),
    )
    return restored


def _choice_to_canonical(display_choice: str, presentation_swapped: bool) -> str:
    """
    Convert participant-facing labels back into canonical A/B/tie form.

    Args:
        display_choice (str): Participant-facing choice label.
        presentation_swapped (bool): Whether displayed positions were swapped.

    Returns:
        str: Canonical `A`, `B`, or `tie` label.
    """
    normalized = display_choice.strip()
    if normalized in {"Tie / Equal", "Tie", "Tie / Not sure"}:
        return "tie"
    if normalized == "Response A":
        return "B" if presentation_swapped else "A"
    if normalized == "Response B":
        return "A" if presentation_swapped else "B"
    raise ValueError(f"Unsupported response choice value: {display_choice!r}")


def _matrix_cell_column(matrix_id: str, row_index: int) -> str:
    """
    Derive the Qualtrics export column name for a matrix cell.

    Args:
        matrix_id (str): Matrix question ID.
        row_index (int): 1-based row index.

    Returns:
        str: Column name in exported CSV.
    """
    return f"{matrix_id}_{row_index}"


def _item_mapping_value(item: Any, field_name: str) -> Any:
    """
    Read a field from either a dataclass-backed or dict-backed manifest item.

    Args:
        item (Any): Stored manifest item payload.
        field_name (str): Field name to read.

    Returns:
        Any: Stored field value.
    """
    if isinstance(item, dict):
        return item[field_name]
    return getattr(item, field_name)


def _column_candidates(
    columns: Sequence[_QualtricsColumn], predicate
) -> List[_QualtricsColumn]:
    """
    Filter columns using a caller-provided predicate.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        predicate: Callable returning True for matching columns.

    Returns:
        List[_QualtricsColumn]: Matching columns in survey order.
    """
    return [column for column in columns if predicate(column)]


def _find_column_by_id(
    columns: Sequence[_QualtricsColumn], expected_id: str
) -> Optional[_QualtricsColumn]:
    """
    Find a column by raw header, ImportId, or normalized name.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        expected_id (str): Stable question identifier.

    Returns:
        Optional[_QualtricsColumn]: Matching column, if present.
    """
    normalized_expected = _normalize_header_name(expected_id)
    matches = _column_candidates(
        columns,
        lambda column: expected_id
        in {column.raw_header, column.import_id, column.normalized_name}
        or normalized_expected == column.normalized_name,
    )
    return matches[0] if matches else None


def _column_label_matches_prompts(
    column: _QualtricsColumn, prompts: Sequence[str]
) -> bool:
    """
    Check whether a resolved Qualtrics column label is compatible with prompts.

    Args:
        column (_QualtricsColumn): Candidate column metadata.
        prompts (Sequence[str]): Accepted prompt labels.

    Returns:
        bool: True when the column label is compatible.
    """
    if not prompts:
        return True
    normalized_label = _safe_str(column.label_header).lower()
    if not normalized_label:
        return True
    # When no dedicated label row exists, the label header mirrors the raw header.
    if normalized_label == _safe_str(column.raw_header).lower():
        return True
    return normalized_label in {_safe_str(prompt).lower() for prompt in prompts}


def _normalize_matrix_prompt_label(text: str) -> str:
    """
    Normalize a matrix row label so multiline description text is ignored.

    Args:
        text (str): Raw Qualtrics label text.

    Returns:
        str: Normalized first-line prompt text.
    """
    first_line = _safe_str(text).splitlines()[0] if _safe_str(text) else ""
    return " ".join(first_line.split()).lower()


def _discover_question_blocks(
    columns: Sequence[_QualtricsColumn], prompt: str
) -> List[_QualtricsColumn]:
    """
    Collect repeated survey columns that share the same participant-facing prompt.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        prompt (str): Human-readable question label.

    Returns:
        List[_QualtricsColumn]: Matching columns in left-to-right order.
    """
    normalized_prompt = _safe_str(prompt).lower()
    return _column_candidates(
        columns, lambda column: column.label_header.lower() == normalized_prompt
    )


def _discover_matrix_question_blocks(
    columns: Sequence[_QualtricsColumn], prompt: str
) -> List[_QualtricsColumn]:
    """
    Collect repeated matrix columns using only the first visible label line.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        prompt (str): Expected matrix-row label prefix line.

    Returns:
        List[_QualtricsColumn]: Matching columns in survey order.
    """
    normalized_prompt = _normalize_matrix_prompt_label(prompt)
    return _column_candidates(
        columns,
        lambda column: _normalize_matrix_prompt_label(column.label_header)
        == normalized_prompt,
    )


def _resolve_assignment_block(
    columns: Sequence[_QualtricsColumn], prompt: str, assignment_rank: int
) -> Optional[_QualtricsColumn]:
    """
    Resolve the nth repeated column for a repeated survey question.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        prompt (str): Human-readable question label.
        assignment_rank (int): 1-based assignment rank for an annotator.

    Returns:
        Optional[_QualtricsColumn]: Matching column for the assignment block.
    """
    matches = _discover_question_blocks(columns, prompt)
    index = assignment_rank - 1
    if index < 0 or index >= len(matches):
        return None
    return matches[index]


def _resolve_matrix_assignment_block(
    columns: Sequence[_QualtricsColumn], prompt: str, assignment_rank: int
) -> Optional[_QualtricsColumn]:
    """
    Resolve the nth repeated column for a repeated matrix-row question.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        prompt (str): Human-readable first-line matrix row label.
        assignment_rank (int): 1-based assignment rank for an annotator.

    Returns:
        Optional[_QualtricsColumn]: Matching column for the assignment block.
    """
    matches = _discover_matrix_question_blocks(columns, prompt)
    index = assignment_rank - 1
    if index < 0 or index >= len(matches):
        return None
    return matches[index]


def _resolve_assignment_block_from_prompts(
    columns: Sequence[_QualtricsColumn],
    prompts: Sequence[str],
    assignment_rank: int,
) -> Optional[_QualtricsColumn]:
    """
    Resolve the nth repeated column for any accepted prompt alias.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        prompts (Sequence[str]): Accepted human-readable question labels.
        assignment_rank (int): 1-based assignment rank for an annotator.

    Returns:
        Optional[_QualtricsColumn]: Matching column for the assignment block.
    """
    for prompt in prompts:
        resolved = _resolve_assignment_block(columns, prompt, assignment_rank)
        if resolved is not None:
            return resolved
    return None


def _resolve_matrix_assignment_block_from_prompts(
    columns: Sequence[_QualtricsColumn],
    prompts: Sequence[str],
    assignment_rank: int,
) -> Optional[_QualtricsColumn]:
    """
    Resolve the nth repeated matrix-row column for any accepted prompt alias.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        prompts (Sequence[str]): Accepted first-line matrix row labels.
        assignment_rank (int): 1-based assignment rank for an annotator.

    Returns:
        Optional[_QualtricsColumn]: Matching matrix column for the assignment block.
    """
    for prompt in prompts:
        resolved = _resolve_matrix_assignment_block(columns, prompt, assignment_rank)
        if resolved is not None:
            return resolved
    return None


def _resolve_single_question_column(
    columns: Sequence[_QualtricsColumn],
    expected_id: str,
    prompt_aliases: Sequence[str],
    assignment_rank: int,
) -> Optional[_QualtricsColumn]:
    """
    Resolve a non-matrix question column by id, then validate against prompt aliases.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        expected_id (str): Stable question identifier.
        prompt_aliases (Sequence[str]): Accepted prompt labels.
        assignment_rank (int): 1-based assignment rank for fallback prompt resolution.

    Returns:
        Optional[_QualtricsColumn]: Matching column, if present.
    """
    direct = _find_column_by_id(columns, expected_id)
    if direct is not None and _column_label_matches_prompts(direct, prompt_aliases):
        return direct
    return _resolve_assignment_block_from_prompts(
        columns, prompt_aliases, assignment_rank
    )


def _extract_matrix_choice_from_block(
    columns: Sequence[_QualtricsColumn],
    assignment_rank: int,
    dimension_title: str,
    matrix_id: str,
    row_index: int,
) -> Optional[_QualtricsColumn]:
    """
    Resolve one matrix-cell column for an assignment block.

    Args:
        columns (Sequence[_QualtricsColumn]): Available columns.
        assignment_rank (int): 1-based assignment rank.
        dimension_title (str): Human-readable dimension title.
        matrix_id (str): Stable matrix identifier for future exports.
        row_index (int): 1-based dimension row index.

    Returns:
        Optional[_QualtricsColumn]: Matching matrix column, if present.
    """
    direct_id = _find_column_by_id(columns, _matrix_cell_column(matrix_id, row_index))
    if direct_id is not None:
        return direct_id
    prompt_aliases = [
        f"{prompt} - {dimension_title}" for prompt in _MATRIX_PROMPT_ALIASES
    ]
    return _resolve_matrix_assignment_block_from_prompts(
        columns, prompt_aliases, assignment_rank
    )


def _parse_bool(value: Any) -> bool:
    """
    Parse a loose boolean string from Qualtrics exports.

    Args:
        value (Any): Raw cell value.

    Returns:
        bool: Parsed boolean.
    """
    text = _safe_str(value).lower()
    return text in {"1", "true", "yes"}


def _parse_int(value: Any) -> int:
    """
    Parse an integer-like CSV field safely.

    Args:
        value (Any): Raw cell value.

    Returns:
        int: Parsed integer, defaulting to 0 on failure.
    """
    text = _safe_str(value)
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _row_classification_reason(
    response_row: pd.Series, expected_study_name: Optional[str]
) -> tuple[bool, str]:
    """
    Classify whether a Qualtrics row is a valid submission candidate.

    Args:
        response_row (pandas.Series): Response row.
        expected_study_name (Optional[str]): Expected study identifier.

    Returns:
        tuple[bool, str]: `(is_valid_candidate, reason)`.
    """
    annotator_id = _safe_str(response_row.get("annotator_id"))
    if not annotator_id:
        return False, "missing_annotator_id"
    if expected_study_name:
        study_name = _safe_str(response_row.get("study_name"))
        if study_name and study_name != expected_study_name:
            return False, "study_name_mismatch"
    status = _safe_str(response_row.get("status")).lower()
    distribution_channel = _safe_str(response_row.get("distributionchannel")).lower()
    if status == "survey test" or distribution_channel == "test":
        return False, "survey_test"
    return True, "valid_submission"


def _submission_sort_key(response_row: pd.Series) -> tuple[int, int, pd.Timestamp]:
    """
    Build a ranking key for selecting the best annotator submission.

    Args:
        response_row (pandas.Series): Response row.

    Returns:
        tuple[int, int, pandas.Timestamp]: Sort key (higher is better).
    """
    finished = int(_parse_bool(response_row.get("finished")))
    progress = _parse_int(response_row.get("progress"))
    recorded = pd.to_datetime(
        _safe_str(response_row.get("recordeddate")), dayfirst=True, errors="coerce"
    )
    if pd.isna(recorded):
        recorded = pd.Timestamp.min
    return finished, progress, recorded


def _select_best_submission_for_annotator(
    exports: Sequence[_NormalizedQualtricsExport],
    annotator_id: str,
    expected_study_name: Optional[str],
) -> Optional[_SelectedSubmission]:
    """
    Select the best valid submission row for one annotator.

    Args:
        exports (Sequence[_NormalizedQualtricsExport]): Normalized exports.
        annotator_id (str): Annotator identifier.
        expected_study_name (Optional[str]): Expected study identifier.

    Returns:
        Optional[_SelectedSubmission]: Selected submission, or None if unavailable.
    """
    candidates: List[_SelectedSubmission] = []
    dropped: Dict[str, int] = {}
    for export in exports:
        if "annotator_id" not in export.response_df.columns:
            continue
        for row_index, row in export.response_df.iterrows():
            if _safe_str(row.get("annotator_id")) != annotator_id:
                continue
            is_valid, reason = _row_classification_reason(row, expected_study_name)
            if not is_valid:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            candidates.append(
                _SelectedSubmission(
                    annotator_id=annotator_id,
                    export_path=export.path,
                    row_index=row_index,
                    response_row=row,
                    columns=export.columns,
                    response_id=_safe_str(row.get("responseid")),
                )
            )

    if not candidates:
        logger.warning(
            "No valid Qualtrics submission found for annotator=%s dropped=%s",
            annotator_id,
            dropped,
        )
        return None

    ranked = sorted(
        candidates,
        key=lambda item: _submission_sort_key(item.response_row),
        reverse=True,
    )
    best_key = _submission_sort_key(ranked[0].response_row)
    tied_best = [
        item for item in ranked if _submission_sort_key(item.response_row) == best_key
    ]
    if len(tied_best) > 1:
        response_ids = [
            item.response_id or f"row_{item.row_index}" for item in tied_best
        ]
        raise ValueError(
            "Multiple equally valid Qualtrics response rows found for "
            f"annotator_id={annotator_id!r}: response_ids={response_ids}"
        )

    selected = ranked[0]
    logger.info(
        "Selected Qualtrics submission for annotator=%s response_id=%s source=%s dropped=%s",
        annotator_id,
        selected.response_id or f"row_{selected.row_index}",
        selected.export_path,
        dropped,
    )
    return selected


def process_qualtrics_results(
    config: HumanAnnotationConfig,
    manifest_path: str | Path,
    response_paths: Sequence[str | Path],
    annotator_name: str | None = None,
) -> List[ProcessedAnnotationRecord]:
    """
    Process Qualtrics exports into canonical annotation records.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        manifest_path (str | Path): Selection manifest path.
        response_paths (Sequence[str | Path]): Response CSV export paths.
        annotator_name (str | None): Optional runtime annotator name that replaces the
            manifest slot id in emitted processed records after matching succeeds.

    Returns:
        List[ProcessedAnnotationRecord]: Normalized annotation records.
    """
    assignments = _load_manifest_assignments(manifest_path)
    exports = [
        export
        for export in (_normalize_qualtrics_export(path) for path in response_paths)
        if export.response_row_count > 0
    ]
    if not exports:
        raise ValueError("No response rows were loaded from the provided CSV paths.")

    unique_annotators = sorted({assignment.annotator_id for assignment in assignments})
    selected_submissions = {
        annotator_id: _select_best_submission_for_annotator(
            exports, annotator_id, config.outputs.study_name
        )
        for annotator_id in unique_annotators
    }

    processed: List[ProcessedAnnotationRecord] = []
    skipped_assignments = 0
    runtime_annotator_name = _safe_str(annotator_name) or None
    matched_manifest_annotator_ids: set[str] = set()

    for assignment in assignments:
        submission = selected_submissions.get(assignment.annotator_id)
        if submission is None:
            skipped_assignments += 1
            continue

        response_row = submission.response_row
        columns = submission.columns
        question_ids = _question_ids(assignment, "overall")

        overall_column = _resolve_single_question_column(
            columns,
            question_ids["overall"],
            _OVERALL_PROMPT_ALIASES,
            assignment.assignment_rank,
        )
        overall_raw = _safe_str(
            response_row.get(overall_column.normalized_name if overall_column else "")
        )
        if not overall_raw:
            logger.warning(
                "Skipping incomplete assignment: annotator=%s item_id=%s item_rank=%d missing=overall",
                assignment.annotator_id,
                _item_mapping_value(assignment.item, "item_id"),
                assignment.assignment_rank,
            )
            skipped_assignments += 1
            continue

        dimension_choices: Dict[str, str] = {}
        matrix_id = question_ids.get("dimensions_matrix")
        if not matrix_id:
            raise ValueError("Missing dimensions_matrix question id in question_ids.")
        missing_dimension = False
        for row_index, dimension in enumerate(config.qualtrics.dimensions, start=1):
            dimension_title = dimension.replace("_", " ").title()
            column = _extract_matrix_choice_from_block(
                columns=columns,
                assignment_rank=assignment.assignment_rank,
                dimension_title=dimension_title,
                matrix_id=matrix_id,
                row_index=row_index,
            )
            dim_raw = _safe_str(
                response_row.get(column.normalized_name if column else "")
            )
            if not dim_raw:
                logger.warning(
                    "Skipping incomplete assignment: annotator=%s item_id=%s item_rank=%d missing_dimension=%s",
                    assignment.annotator_id,
                    _item_mapping_value(assignment.item, "item_id"),
                    assignment.assignment_rank,
                    dimension,
                )
                missing_dimension = True
                break
            dimension_choices[dimension] = _choice_to_canonical(
                dim_raw, assignment.presentation_swapped
            )
        if missing_dimension:
            skipped_assignments += 1
            continue

        overall_confidence = None
        if config.qualtrics.include_confidence_question:
            confidence_column = _resolve_single_question_column(
                columns,
                question_ids["confidence"],
                _CONFIDENCE_PROMPT_ALIASES,
                assignment.assignment_rank,
            )
            overall_confidence = (
                _safe_str(
                    response_row.get(
                        confidence_column.normalized_name if confidence_column else ""
                    )
                )
                or None
            )

        overall_rationale = ""
        if config.qualtrics.include_rationale_question:
            rationale_column = _resolve_single_question_column(
                columns,
                question_ids["rationale"],
                _RATIONALE_PROMPT_ALIASES,
                assignment.assignment_rank,
            )
            overall_rationale = _safe_str(
                response_row.get(
                    rationale_column.normalized_name if rationale_column else ""
                )
            )

        candidate = _item_mapping_value(assignment.item, "candidate")
        if not isinstance(candidate, dict):
            raise ValueError("Manifest item.candidate must be a dictionary payload.")
        matched_manifest_annotator_ids.add(assignment.annotator_id)
        processed_annotator_id = runtime_annotator_name or assignment.annotator_id
        processed.append(
            ProcessedAnnotationRecord(
                annotator_id=processed_annotator_id,
                item_id=str(_item_mapping_value(assignment.item, "item_id")),
                task_id=str(candidate["task_id"]),
                variant_id=str(candidate["variant_id"]),
                raw_task_id=str(candidate["raw_task_id"]),
                persona=str(candidate["persona"]),
                prompt_type=str(candidate["prompt_type"]),
                pairwise_judgment_type=str(candidate["pairwise_judgment_type"]),
                model_a_name=str(candidate["model_a_name"]),
                model_b_name=str(candidate["model_b_name"]),
                model_a_output=str(candidate["model_a_output"]),
                model_b_output=str(candidate["model_b_output"]),
                input_text=str(candidate["input_text"]),
                overall_choice=_choice_to_canonical(
                    overall_raw, assignment.presentation_swapped
                ),
                overall_confidence=overall_confidence,
                overall_rationale=overall_rationale,
                dimension_choices=dimension_choices,
                metadata={
                    "presentation_swapped": assignment.presentation_swapped,
                    "assignment_rank": assignment.assignment_rank,
                    "assignment_role": assignment.assignment_role,
                    "manifest_annotator_id": assignment.annotator_id,
                    "selection_target": _item_mapping_value(
                        assignment.item, "selection_target"
                    ),
                    "artifact_path": str(candidate["artifact_path"]),
                    "artifact_index": int(candidate["artifact_index"]),
                    "generator_model": str(candidate["generator_model"]),
                    "filter_model": str(candidate["filter_model"]),
                    "qualtrics_response_id": submission.response_id,
                    "qualtrics_export_path": str(submission.export_path),
                },
            )
        )

    if runtime_annotator_name and len(matched_manifest_annotator_ids) > 1:
        raise ValueError(
            "annotator_name override matched multiple manifest annotator slots: "
            f"{sorted(matched_manifest_annotator_ids)}"
        )

    logger.info(
        "Processed %d human-annotation response records across %d annotators skipped_assignments=%d",
        len(processed),
        len({record.annotator_id for record in processed}),
        skipped_assignments,
    )
    return processed


def processed_records_to_frame(
    records: Iterable[ProcessedAnnotationRecord],
) -> pd.DataFrame:
    """
    Convert processed records into a flat pandas DataFrame.

    Args:
        records (Iterable[ProcessedAnnotationRecord]): Processed records.

    Returns:
        pandas.DataFrame: Flat dataframe representation.
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        row = asdict(record)
        for dimension, choice in record.dimension_choices.items():
            row[f"dim_{dimension}_choice"] = choice
        rows.append(row)
    return pd.DataFrame(rows)
