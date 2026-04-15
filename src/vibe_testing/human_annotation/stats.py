"""Study-level statistics for standalone human annotation workflows."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.vibe_testing.human_annotation.schemas import ProcessedAnnotationRecord
from src.vibe_testing.ui.pairwise_explorer_stats import compute_judge_pair_agreement

logger = logging.getLogger(__name__)


def compute_study_stats(
    records: Iterable[ProcessedAnnotationRecord],
) -> Dict[str, Any]:
    """
    Compute compact study statistics from processed annotations.

    Args:
        records (Iterable[ProcessedAnnotationRecord]): Processed annotation records.

    Returns:
        Dict[str, Any]: Summary statistics payload.
    """
    record_list = list(records)
    if not record_list:
        raise ValueError("Cannot compute study statistics from zero processed records.")

    overall_counts = Counter(record.overall_choice for record in record_list)
    by_annotator = Counter(record.annotator_id for record in record_list)
    by_persona = Counter(record.persona for record in record_list)
    by_prompt_type = Counter(record.prompt_type for record in record_list)
    grouped_by_item: Dict[str, List[ProcessedAnnotationRecord]] = defaultdict(list)
    for record in record_list:
        grouped_by_item[record.item_id].append(record)

    agreement_counts = Counter()
    for item_records in grouped_by_item.values():
        choices = {record.overall_choice for record in item_records}
        if len(item_records) < 2:
            continue
        agreement_counts["items_with_overlap"] += 1
        if len(choices) == 1:
            agreement_counts["items_with_full_overall_agreement"] += 1
        else:
            agreement_counts["items_with_overall_disagreement"] += 1

    stats = {
        "n_records": len(record_list),
        "n_annotators": len(by_annotator),
        "n_unique_items": len(grouped_by_item),
        "overall_choice_counts": dict(overall_counts),
        "records_by_annotator": dict(by_annotator),
        "records_by_persona": dict(by_persona),
        "records_by_prompt_type": dict(by_prompt_type),
        "agreement": dict(agreement_counts),
    }
    logger.info(
        "Computed study stats: records=%d annotators=%d unique_items=%d overall_choices=%s",
        len(record_list),
        len(by_annotator),
        len(grouped_by_item),
        dict(overall_counts),
    )
    return stats


def _choice_counts_by_dimension(
    records: List[ProcessedAnnotationRecord],
) -> Dict[str, Dict[str, int]]:
    """
    Count human choices by dimension across processed records.

    Args:
        records (List[ProcessedAnnotationRecord]): Processed annotation records.

    Returns:
        Dict[str, Dict[str, int]]: Dimension -> choice histogram.
    """
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for dimension, choice in record.dimension_choices.items():
            counts[dimension][choice] += 1
    return {dimension: dict(counter) for dimension, counter in sorted(counts.items())}


def _mean_or_none(frame: pd.DataFrame, column_name: str) -> float | None:
    """
    Compute a numeric column mean while returning None for empty/NaN-only columns.

    Args:
        frame (pd.DataFrame): Input data frame.
        column_name (str): Column to average.

    Returns:
        float | None: Mean value or None when undefined.
    """
    if frame.empty or column_name not in frame.columns:
        return None
    series = frame[column_name]
    non_null = series.dropna()
    if non_null.empty:
        return None
    return float(non_null.mean())


def _agreement_table_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert an agreement data frame into JSON-friendly records.

    Args:
        frame (pd.DataFrame): Agreement metrics table.

    Returns:
        List[Dict[str, Any]]: JSON-friendly row records.
    """
    if frame.empty:
        return []
    sanitized = frame.where(pd.notna(frame), None)
    return sanitized.to_dict(orient="records")


def _compute_choice_agreement_summary(
    outcomes_by_annotator: Dict[str, Dict[str, str]],
    *,
    item_ids: List[str],
) -> Dict[str, Any]:
    """
    Compute pairwise agreement metrics for a categorical choice space.

    Args:
        outcomes_by_annotator (Dict[str, Dict[str, str]]): Annotator -> item -> choice.
        item_ids (List[str]): Overlapping item ids to analyze.

    Returns:
        Dict[str, Any]: Pairwise agreement table plus aggregate means.
    """
    if len(outcomes_by_annotator) < 2 or not item_ids:
        return {
            "n_overlap_items": len(item_ids),
            "pairwise_rows": [],
            "mean_percent_agreement": None,
            "mean_cohens_kappa": None,
            "mean_percent_agreement_excl_ties": None,
            "mean_cohens_kappa_excl_ties": None,
        }
    agreement_frame = compute_judge_pair_agreement(
        outcomes_by_judge=outcomes_by_annotator,
        task_ids=item_ids,
        categories=("A", "B", "tie"),
    )
    return {
        "n_overlap_items": len(item_ids),
        "pairwise_rows": _agreement_table_records(agreement_frame),
        "mean_percent_agreement": _mean_or_none(agreement_frame, "percent_agreement"),
        "mean_cohens_kappa": _mean_or_none(agreement_frame, "cohens_kappa"),
        "mean_percent_agreement_excl_ties": _mean_or_none(
            agreement_frame, "percent_agreement_excl_ties"
        ),
        "mean_cohens_kappa_excl_ties": _mean_or_none(
            agreement_frame, "cohens_kappa_excl_ties"
        ),
    }


def compute_annotation_analysis_summary(
    records: Iterable[ProcessedAnnotationRecord],
) -> Dict[str, Any]:
    """
    Compute a richer descriptive and agreement summary for processed annotations.

    Args:
        records (Iterable[ProcessedAnnotationRecord]): Processed annotation records.

    Returns:
        Dict[str, Any]: JSON-friendly analysis summary.
    """
    record_list = list(records)
    if not record_list:
        raise ValueError(
            "Cannot compute annotation analysis summary from zero records."
        )

    by_annotator = Counter(record.annotator_id for record in record_list)
    by_persona = Counter(record.persona for record in record_list)
    by_prompt_type = Counter(record.prompt_type for record in record_list)
    overall_choice_counts = Counter(record.overall_choice for record in record_list)
    grouped_by_item: Dict[str, List[ProcessedAnnotationRecord]] = defaultdict(list)
    for record in record_list:
        grouped_by_item[record.item_id].append(record)

    overlap_item_ids = sorted(
        item_id
        for item_id, item_records in grouped_by_item.items()
        if len(item_records) > 1
    )
    overlap_counts = {
        item_id: len(grouped_by_item[item_id]) for item_id in overlap_item_ids
    }

    overall_outcomes_by_annotator: Dict[str, Dict[str, str]] = defaultdict(dict)
    dimension_outcomes_by_dimension: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for item_id in overlap_item_ids:
        for record in grouped_by_item[item_id]:
            overall_outcomes_by_annotator[record.annotator_id][
                item_id
            ] = record.overall_choice
            for dimension, choice in record.dimension_choices.items():
                dimension_outcomes_by_dimension[dimension][record.annotator_id][
                    item_id
                ] = choice

    agreement_summary = {
        "items_with_overlap": len(overlap_item_ids),
        "overlap_item_annotation_counts": overlap_counts,
        "overall": _compute_choice_agreement_summary(
            dict(overall_outcomes_by_annotator),
            item_ids=overlap_item_ids,
        ),
        "dimensions": {
            dimension: _compute_choice_agreement_summary(
                {
                    annotator_id: item_map
                    for annotator_id, item_map in outcomes_by_annotator.items()
                },
                item_ids=sorted(
                    {
                        item_id
                        for item_map in outcomes_by_annotator.values()
                        for item_id in item_map.keys()
                    }
                ),
            )
            for dimension, outcomes_by_annotator in sorted(
                dimension_outcomes_by_dimension.items()
            )
        },
    }

    summary = {
        "counts": {
            "n_records": len(record_list),
            "n_unique_items": len(grouped_by_item),
            "n_annotators": len(by_annotator),
        },
        "distributions": {
            "records_by_annotator": dict(by_annotator),
            "records_by_persona": dict(by_persona),
            "records_by_prompt_type": dict(by_prompt_type),
            "overall_choice_counts": dict(overall_choice_counts),
            "dimension_choice_counts": _choice_counts_by_dimension(record_list),
        },
        "agreement": agreement_summary,
    }
    logger.info(
        "Computed annotation analysis summary: records=%d overlap_items=%d annotators=%d",
        len(record_list),
        len(overlap_item_ids),
        len(by_annotator),
    )
    return summary
