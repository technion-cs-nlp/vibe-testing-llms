"""Export helpers for standalone human annotation workflows."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from src.vibe_testing.human_annotation.config import build_study_artifact_paths
from src.vibe_testing.human_annotation.filters import FilterResult
from src.vibe_testing.human_annotation.sample_type_utils import sample_type_label
from src.vibe_testing.human_annotation.schemas import (
    AnnotatorAssignment,
    HumanAnnotationConfig,
    PairwiseCandidateRecord,
    ProcessedAnnotationRecord,
    SampledAnnotationItem,
)
from src.vibe_testing.model_names import canonicalize_model_name
from src.vibe_testing.pathing import (
    PairwiseModelRouting,
    canonicalize_pairwise_model_routing,
    normalize_token,
    pairwise_stage_dir,
    parse_pairwise_artifact_path,
)
from src.vibe_testing.utils import save_json

logger = logging.getLogger(__name__)
_DEFAULT_HUMAN_OVERALL_POLICY = "dimension_majority"


def _swap_choice_label(choice: str) -> str:
    """
    Swap a canonical pairwise choice label.

    Args:
        choice (str): Canonical `A`, `B`, or `tie` label.

    Returns:
        str: Swapped choice label.
    """
    if choice == "A":
        return "B"
    if choice == "B":
        return "A"
    if choice == "tie":
        return "tie"
    raise ValueError(f"Unsupported canonical annotation choice: {choice}")


def _record_orientation_matches_route(
    record: ProcessedAnnotationRecord, routing: PairwiseModelRouting
) -> bool:
    """
    Determine whether a processed record already matches the routed A/B ordering.

    Args:
        record (ProcessedAnnotationRecord): Processed annotation record.
        routing (PairwiseModelRouting): Target routed/canonical model-pair metadata.

    Returns:
        bool: True when the record already matches the routed left/right ordering.
    """
    record_model_a_token = normalize_token(record.model_a_name)
    record_model_b_token = normalize_token(record.model_b_name)
    if (
        record_model_a_token == routing.route_model_a_token
        and record_model_b_token == routing.route_model_b_token
    ):
        return True
    return (
        canonicalize_model_name(record.model_a_name) == routing.canonical_model_a
        and canonicalize_model_name(record.model_b_name) == routing.canonical_model_b
    )


def _export_route_from_record(
    config: HumanAnnotationConfig, record: ProcessedAnnotationRecord
) -> Dict[str, Any]:
    """
    Resolve the routed export context for a processed human annotation record.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        record (ProcessedAnnotationRecord): Processed annotation record.

    Returns:
        Dict[str, Any]: Resolved route metadata and model routing.
    """
    artifact_path = str(record.metadata.get("artifact_path") or "").strip()
    if artifact_path:
        try:
            parsed = parse_pairwise_artifact_path(
                config.outputs.export_base_dir, artifact_path
            )
            return {
                "persona": parsed.persona,
                "generator_model": parsed.generator_model,
                "filter_model": parsed.filter_model,
                "prompt_type": parsed.prompt_type or record.prompt_type,
                "pairwise_judgment_type": parsed.pairwise_judgment_type,
                "model_routing": parsed.model_routing,
            }
        except ValueError:
            logger.warning(
                "Falling back to record-level routing for artifact path %s",
                artifact_path,
            )
    return {
        "persona": record.persona,
        "generator_model": str(record.metadata.get("generator_model") or "unknown"),
        "filter_model": str(record.metadata.get("filter_model") or "none"),
        "prompt_type": record.prompt_type,
        "pairwise_judgment_type": record.pairwise_judgment_type,
        "model_routing": canonicalize_pairwise_model_routing(
            record.model_a_name,
            record.model_b_name,
            route_model_a=record.model_a_name,
            route_model_b=record.model_b_name,
        ),
    }


def _canonicalize_record_for_export(
    config: HumanAnnotationConfig, record: ProcessedAnnotationRecord
) -> tuple[ProcessedAnnotationRecord, Dict[str, Any]]:
    """
    Rewrite a processed annotation record into canonical routed A/B order.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        record (ProcessedAnnotationRecord): Processed annotation record.

    Returns:
        tuple[ProcessedAnnotationRecord, Dict[str, Any]]: Rewritten record and route.
    """
    route = _export_route_from_record(config, record)
    routing = route["model_routing"]
    if _record_orientation_matches_route(record, routing):
        rewritten = replace(
            record,
            persona=route["persona"],
            prompt_type=route["prompt_type"],
            pairwise_judgment_type=route["pairwise_judgment_type"],
        )
    else:
        rewritten = replace(
            record,
            persona=route["persona"],
            prompt_type=route["prompt_type"],
            pairwise_judgment_type=route["pairwise_judgment_type"],
            model_a_name=record.model_b_name,
            model_b_name=record.model_a_name,
            model_a_output=record.model_b_output,
            model_b_output=record.model_a_output,
            overall_choice=_swap_choice_label(record.overall_choice),
            dimension_choices={
                dimension: _swap_choice_label(choice)
                for dimension, choice in record.dimension_choices.items()
            },
            metadata={
                **record.metadata,
                "raw_model_a_name": record.model_a_name,
                "raw_model_b_name": record.model_b_name,
            },
        )
    return rewritten, route


def _count_rows_by_field(
    candidates: Sequence[PairwiseCandidateRecord], field_name: str
) -> Dict[str, int]:
    """
    Count candidate rows by a single field.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Candidate rows.
        field_name (str): Candidate field name.

    Returns:
        Dict[str, int]: Field-value counts.
    """
    return dict(
        Counter(str(getattr(candidate, field_name)) for candidate in candidates)
    )


def _count_items_by_field(
    candidates: Sequence[PairwiseCandidateRecord], field_name: str
) -> Dict[str, int]:
    """
    Count unique source items by a representative field value.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Candidate rows.
        field_name (str): Candidate field name.

    Returns:
        Dict[str, int]: Unique source-item counts by field.
    """
    by_source_key: Dict[str, PairwiseCandidateRecord] = {}
    for candidate in candidates:
        by_source_key.setdefault(candidate.source_key, candidate)
    return _count_rows_by_field(list(by_source_key.values()), field_name)


def _selected_items_by_field(
    sampled_items: Sequence[SampledAnnotationItem], field_name: str
) -> Dict[str, int]:
    """
    Count sampled items by candidate field.

    Args:
        sampled_items (Sequence[SampledAnnotationItem]): Sampled study items.
        field_name (str): Candidate field name.

    Returns:
        Dict[str, int]: Sampled-item counts by field.
    """
    return dict(
        Counter(str(getattr(item.candidate, field_name)) for item in sampled_items)
    )


def _selected_agreement_stats(
    filter_result: FilterResult, sampled_items: Sequence[SampledAnnotationItem]
) -> Dict[str, Any]:
    """
    Summarize agreement metadata for the actually sampled source items.

    Args:
        filter_result (FilterResult): Filter result with per-group audit data.
        sampled_items (Sequence[SampledAnnotationItem]): Sampled study items.

    Returns:
        Dict[str, Any]: Selected-set agreement statistics.
    """
    sampled_source_keys = {item.candidate.source_key for item in sampled_items}
    selected_outcomes = [
        outcome
        for outcome in filter_result.group_outcomes
        if outcome.source_key in sampled_source_keys
    ]
    winning_bucket_histogram = Counter(
        str(outcome.consensus_bucket_size)
        for outcome in selected_outcomes
        if outcome.consensus_bucket_size > 0
    )
    return {
        "n_selected_groups": len(selected_outcomes),
        "selected_group_outcomes": dict(
            Counter(outcome.outcome for outcome in selected_outcomes)
        ),
        "selected_consensus_bucket_sizes": dict(winning_bucket_histogram),
        "selected_signature_examples": [
            {
                "source_key": outcome.source_key,
                "selected_signature": outcome.selected_signature,
                "signature_counts": outcome.signature_counts,
            }
            for outcome in selected_outcomes[:5]
        ],
    }


def save_filter_summary(
    config: HumanAnnotationConfig,
    *,
    discovered_candidates: Sequence[PairwiseCandidateRecord],
    filter_result: FilterResult,
    prior_selection_plan_audit: Dict[str, Any] | None,
    sample_type_inclusion_audit: Dict[str, Any] | None,
    repeat_inclusion_audit: Dict[str, Any] | None,
    sampled_items: Sequence[SampledAnnotationItem],
    assignments: Sequence[AnnotatorAssignment] | None = None,
) -> Path:
    """
    Save a filter-summary payload for auditing the study input set.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        discovered_candidates (Sequence[PairwiseCandidateRecord]): Candidate rows
            before filtering.
        filter_result (FilterResult): Filtering result payload.
        prior_selection_plan_audit (Dict[str, Any] | None): Optional audit payload for
            exclusions applied from prior selection plans before sampling.
        sample_type_inclusion_audit (Dict[str, Any] | None): Optional audit payload for
            exact sample-type inclusion before sampling.
        repeat_inclusion_audit (Dict[str, Any] | None): Optional audit payload for
            repeated-item inclusion before sampling.
        sampled_items (Sequence[SampledAnnotationItem]): Sampled study items.
        assignments (Sequence[AnnotatorAssignment] | None): Optional assignment records.

    Returns:
        Path: Written summary path.
    """
    paths = build_study_artifact_paths(config)
    kept_candidates = list(filter_result.kept)
    rejected_candidates = list(filter_result.rejected)
    discovered_source_keys = {
        candidate.source_key for candidate in discovered_candidates
    }
    kept_source_keys = {candidate.source_key for candidate in kept_candidates}
    rejected_source_keys = {candidate.source_key for candidate in rejected_candidates}
    fully_rejected_source_keys = discovered_source_keys - kept_source_keys
    mixed_outcome_source_keys = kept_source_keys & rejected_source_keys
    agreement_stats = dict(filter_result.agreement_stats)
    prior_plan_audit = dict(prior_selection_plan_audit or {})
    sample_type_audit = dict(sample_type_inclusion_audit or {})
    repeat_audit = dict(repeat_inclusion_audit or {})
    marginal_balance_audit = {}
    weighted_include_audit = {}
    for item in sampled_items:
        audit = item.selection_metadata.get("marginal_balance_audit")
        if isinstance(audit, dict):
            marginal_balance_audit = dict(audit)
            break
    for item in sampled_items:
        audit = item.selection_metadata.get("included_type_audit")
        if isinstance(audit, dict):
            weighted_include_audit = dict(audit)
            break
    assignment_records = list(assignments or [])
    sampleable_after_prior_plan_exclusion = int(
        prior_plan_audit.get("kept_source_items", len(kept_source_keys))
    )
    per_annotator_type_counts: Dict[str, Dict[str, int]] = {}
    per_annotator_role_counts: Dict[str, Dict[str, int]] = {}
    if assignment_records and config.annotators.balance_by:
        for assignment in assignment_records:
            per_annotator_type_counts.setdefault(assignment.annotator_id, {})
            label = sample_type_label(
                assignment.item.candidate, config.annotators.balance_by
            )
            per_annotator_type_counts[assignment.annotator_id][label] = (
                per_annotator_type_counts[assignment.annotator_id].get(label, 0) + 1
            )
    if assignment_records:
        for assignment in assignment_records:
            per_annotator_role_counts.setdefault(assignment.annotator_id, {})
            item_role = str(
                assignment.item.selection_metadata.get("item_role")
                or assignment.assignment_role
            )
            per_annotator_role_counts[assignment.annotator_id][item_role] = (
                per_annotator_role_counts[assignment.annotator_id].get(item_role, 0) + 1
            )
    sampled_item_role_counts = dict(
        Counter(
            str(item.selection_metadata.get("item_role") or "regular")
            for item in sampled_items
        )
    )
    assignment_role_counts = dict(
        Counter(
            str(
                assignment.item.selection_metadata.get("item_role")
                or assignment.assignment_role
            )
            for assignment in assignment_records
        )
    )
    regular_item_assignment_counts = Counter(
        assignment.item.item_id
        for assignment in assignment_records
        if str(assignment.item.selection_metadata.get("item_role") or "") == "regular"
    )
    winning_consensus_histogram = {
        key.replace("winning_consensus_size_", ""): value
        for key, value in agreement_stats.items()
        if key.startswith("winning_consensus_size_")
    }
    examples = [
        {
            "source_key": outcome.source_key,
            "persona": outcome.persona,
            "prompt_type": outcome.prompt_type,
            "model_pair": outcome.model_pair,
            "pairwise_judgment_type": outcome.pairwise_judgment_type,
            "outcome": outcome.outcome,
            "agreement_pool_size": outcome.agreement_pool_size,
            "required_consensus_size": outcome.required_consensus_size,
            "consensus_bucket_size": outcome.consensus_bucket_size,
            "signature_counts": outcome.signature_counts,
        }
        for outcome in filter_result.group_outcomes
        if not outcome.outcome.startswith("kept_")
    ][:10]
    payload = {
        "counts": {
            "n_discovered_rows": len(discovered_candidates),
            "n_discovered_source_items": len(discovered_source_keys),
            "n_rejected_rows": len(rejected_candidates),
            "n_rejected_source_items": len(fully_rejected_source_keys),
            "n_source_items_with_any_rejected_rows": len(rejected_source_keys),
            "n_kept_rows": len(kept_candidates),
            "n_kept_source_items": len(kept_source_keys),
            "n_mixed_outcome_source_items": len(mixed_outcome_source_keys),
            "n_sampleable_after_dedup": sampleable_after_prior_plan_exclusion,
            "n_sampled_items": len(sampled_items),
        },
        "filters": config.filters.to_public_dict(),
        "sample_type_inclusion": {
            "enabled": bool(sample_type_audit.get("enabled", False)),
            "requested_types": list(sample_type_audit.get("requested_types", [])),
            "requested_type_counts": dict(
                sample_type_audit.get("requested_type_counts", {})
            ),
            "requested_total_weight": int(
                sample_type_audit.get("requested_total_weight", 0)
            ),
            "matched_rows": int(sample_type_audit.get("matched_rows", 0)),
            "matched_source_items": int(
                sample_type_audit.get("matched_source_items", 0)
            ),
            "available_source_items_by_type": dict(
                sample_type_audit.get("available_source_items_by_type", {})
            ),
            "missing_types": list(sample_type_audit.get("missing_types", [])),
            "matched_type_labels": list(
                sample_type_audit.get("matched_type_labels", [])
            ),
            "weighted_quota_enabled": bool(
                weighted_include_audit.get("enabled", False)
            ),
            "quota_unit": weighted_include_audit.get("quota_unit"),
            "requested_quota_by_type": dict(
                weighted_include_audit.get("requested_quota_by_type", {})
            ),
            "sampled_items_by_included_type": dict(
                Counter(
                    str(item.selection_metadata.get("included_type_label") or "")
                    for item in sampled_items
                    if item.selection_metadata.get("included_type_label")
                )
            ),
            "override_equal_allocation": bool(
                weighted_include_audit.get("override_equal_allocation", False)
            ),
            "override_fields": list(weighted_include_audit.get("override_fields", [])),
        },
        "repeat_inclusion": {
            "enabled": bool(repeat_audit.get("enabled", False)),
            "repeat_mode": repeat_audit.get("repeat_mode"),
            "requested_source_keys": list(
                repeat_audit.get("requested_source_keys", [])
            ),
            "matched_source_keys": list(repeat_audit.get("matched_source_keys", [])),
            "missing_source_keys": list(repeat_audit.get("missing_source_keys", [])),
            "pinned_source_items": int(repeat_audit.get("pinned_source_items", 0)),
            "included_from_selection_plan_paths": list(
                repeat_audit.get("included_from_selection_plan_paths", [])
            ),
            "included_from_manifest_paths": list(
                repeat_audit.get("included_from_manifest_paths", [])
            ),
        },
        "prior_selection_plan_exclusion": {
            "enabled": bool(prior_plan_audit.get("enabled", False)),
            "plan_paths": list(prior_plan_audit.get("plan_paths", [])),
            "excluded_rows": int(prior_plan_audit.get("excluded_rows", 0)),
            "excluded_source_items": int(
                prior_plan_audit.get("excluded_source_items", 0)
            ),
            "kept_rows": int(prior_plan_audit.get("kept_rows", len(kept_candidates))),
            "kept_source_items": sampleable_after_prior_plan_exclusion,
            "excluded_rows_by_persona": dict(
                prior_plan_audit.get("excluded_rows_by_persona", {})
            ),
            "excluded_rows_by_prompt_type": dict(
                prior_plan_audit.get("excluded_rows_by_prompt_type", {})
            ),
            "excluded_rows_by_model_pair": dict(
                prior_plan_audit.get("excluded_rows_by_model_pair", {})
            ),
        },
        "marginal_balance": {
            "enabled": bool(marginal_balance_audit.get("enabled", False)),
            "mode": marginal_balance_audit.get("mode"),
            "fields": list(marginal_balance_audit.get("fields", [])),
            "requested_total_samples": marginal_balance_audit.get(
                "requested_total_samples"
            ),
            "eligible_pool_size": marginal_balance_audit.get("eligible_pool_size"),
            "eligible_counts_by_field": dict(
                marginal_balance_audit.get("eligible_counts_by_field", {})
            ),
            "field_values": dict(marginal_balance_audit.get("field_values", {})),
            "required_strata": list(marginal_balance_audit.get("required_strata", [])),
            "stratum_support": dict(marginal_balance_audit.get("stratum_support", {})),
            "per_stratum_quota": marginal_balance_audit.get("per_stratum_quota"),
            "realized_counts_by_field": dict(
                marginal_balance_audit.get("realized_counts_by_field", {})
            ),
            "realized_stratum_counts": dict(
                marginal_balance_audit.get("realized_stratum_counts", {})
            ),
        },
        "rejection_counts": dict(filter_result.rejection_counts),
        "agreement": {
            "enabled": config.filters.require_judge_consensus,
            "consensus_scope": config.filters.consensus_scope,
            "min_judges_in_pool": config.filters.min_judges_in_pool,
            "min_consensus_judges": config.filters.min_consensus_judges,
            "exclude_tied_overall_from_consensus": (
                config.filters.exclude_tied_overall_from_consensus
            ),
            "groups_total": agreement_stats.get("groups_total", 0),
            "groups_with_sufficient_pool": agreement_stats.get(
                "groups_with_sufficient_pool", 0
            ),
            "groups_passing_consensus": agreement_stats.get(
                "groups_passing_consensus", 0
            ),
            "groups_failing_pool_threshold": agreement_stats.get(
                "groups_failing_pool_threshold", 0
            ),
            "groups_failing_consensus": agreement_stats.get(
                "groups_failing_consensus", 0
            ),
            "rows_excluded_tied_before_consensus": agreement_stats.get(
                "rows_excluded_tied_before_consensus", 0
            ),
            "winning_consensus_size_histogram": winning_consensus_histogram,
        },
        "distributions": {
            "rows_by_persona": {
                "kept": _count_rows_by_field(kept_candidates, "persona"),
                "rejected": _count_rows_by_field(rejected_candidates, "persona"),
            },
            "rows_by_prompt_type": {
                "kept": _count_rows_by_field(kept_candidates, "prompt_type"),
                "rejected": _count_rows_by_field(rejected_candidates, "prompt_type"),
            },
            "rows_by_model_pair": {
                "kept": _count_rows_by_field(kept_candidates, "model_pair"),
                "rejected": _count_rows_by_field(rejected_candidates, "model_pair"),
            },
            "rows_by_pairwise_judgment_type": {
                "kept": _count_rows_by_field(kept_candidates, "pairwise_judgment_type"),
                "rejected": _count_rows_by_field(
                    rejected_candidates, "pairwise_judgment_type"
                ),
            },
            "rows_by_judge_dir_name": {
                "kept": _count_rows_by_field(kept_candidates, "judge_dir_name"),
                "rejected": _count_rows_by_field(rejected_candidates, "judge_dir_name"),
            },
            "rows_by_judge_model_name": {
                "kept": _count_rows_by_field(kept_candidates, "judge_model_name"),
                "rejected": _count_rows_by_field(
                    rejected_candidates, "judge_model_name"
                ),
            },
            "source_items_by_persona": {
                "kept": _count_items_by_field(kept_candidates, "persona"),
                "with_any_rejected_rows": _count_items_by_field(
                    rejected_candidates, "persona"
                ),
                "fully_rejected": _count_rows_by_field(
                    [
                        candidate
                        for candidate in discovered_candidates
                        if candidate.source_key in fully_rejected_source_keys
                    ],
                    "persona",
                ),
            },
        },
        "selection_stats": {
            "sampled_items_by_persona": _selected_items_by_field(
                sampled_items, "persona"
            ),
            "sampled_items_by_prompt_type": _selected_items_by_field(
                sampled_items, "prompt_type"
            ),
            "sampled_items_by_model_pair": _selected_items_by_field(
                sampled_items, "model_pair"
            ),
            "sampled_items_by_pairwise_judgment_type": _selected_items_by_field(
                sampled_items, "pairwise_judgment_type"
            ),
            "selected_overall_winner_counts": dict(
                Counter(str(item.candidate.overall_winner) for item in sampled_items)
            ),
            "selected_overall_tie_rate": (
                sum(
                    1
                    for item in sampled_items
                    if str(item.candidate.overall_winner) in {"None", "tie", ""}
                )
                / len(sampled_items)
                if sampled_items
                else 0.0
            ),
            "selected_agreement": _selected_agreement_stats(
                filter_result, sampled_items
            ),
            "sampled_items_by_role": sampled_item_role_counts,
            "assignment_rows_by_role": assignment_role_counts,
            "regular_item_assignment_count_histogram": dict(
                Counter(regular_item_assignment_counts.values())
            ),
        },
        "annotator_balance": {
            "enabled": bool(config.annotators.balance_by),
            "fields": list(config.annotators.balance_by),
            "scope": config.annotators.balance_scope,
            "mode": config.annotators.balance_mode,
            "annotators_per_regular_item": config.annotators.annotators_per_regular_item,
            "calibration_sample_types": [
                selector.to_dict()
                for selector in config.annotators.calibration_sample_types
            ],
            "per_annotator_role_counts": per_annotator_role_counts,
            "per_annotator_type_counts": per_annotator_type_counts,
        },
        "examples": examples,
    }
    save_json(payload, str(paths.filter_summary_path))
    logger.info(
        "Saved filter summary to %s with kept_rows=%d sampled_items=%d rejected_rows=%d",
        paths.filter_summary_path,
        len(kept_candidates),
        len(sampled_items),
        len(rejected_candidates),
    )
    return paths.filter_summary_path


def save_processed_outputs(
    config: HumanAnnotationConfig,
    records: Sequence[ProcessedAnnotationRecord],
    stats_payload: Dict[str, Any],
    *,
    analysis_payload: Dict[str, Any] | None = None,
) -> Dict[str, Path]:
    """
    Save processed records and study stats into the workspace.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        records (Sequence[ProcessedAnnotationRecord]): Processed records.
        stats_payload (Dict[str, Any]): Study-level statistics payload.
        analysis_payload (Dict[str, Any] | None): Optional richer annotation-analysis
            summary payload.

    Returns:
        Dict[str, Path]: Written artifact paths.
    """
    paths = build_study_artifact_paths(config)
    rows = [asdict(record) for record in records]
    save_json(rows, str(paths.processed_json_path))
    pd.DataFrame(rows).to_csv(paths.processed_csv_path, index=False)
    save_json(stats_payload, str(paths.stats_json_path))
    if analysis_payload is not None:
        save_json(analysis_payload, str(paths.analysis_summary_json_path))
    logger.info(
        "Saved processed outputs: json=%s csv=%s stats=%s analysis=%s records=%d",
        paths.processed_json_path,
        paths.processed_csv_path,
        paths.stats_json_path,
        paths.analysis_summary_json_path if analysis_payload is not None else None,
        len(rows),
    )
    written_paths: Dict[str, Path] = {
        "processed_json": paths.processed_json_path,
        "processed_csv": paths.processed_csv_path,
        "stats_json": paths.stats_json_path,
    }
    if analysis_payload is not None:
        written_paths["analysis_summary_json"] = paths.analysis_summary_json_path
    return written_paths


def _choice_to_winner_model(
    choice: str, model_a_name: str, model_b_name: str
) -> tuple[str, str | None]:
    """
    Convert a canonical annotation choice into Stage-5b winner fields.

    Args:
        choice (str): Canonical `A`, `B`, or `tie` label.
        model_a_name (str): Compared model A.
        model_b_name (str): Compared model B.

    Returns:
        tuple[str, str | None]: Stage-5b `winner` value and resolved winner model.
    """
    if choice == "A":
        return "A", model_a_name
    if choice == "B":
        return "B", model_b_name
    if choice == "tie":
        return "tie", None
    raise ValueError(f"Unsupported canonical annotation choice: {choice}")


def _determine_human_overall_winner(
    record: ProcessedAnnotationRecord, policy: str
) -> tuple[str, str | None]:
    """
    Determine the exported overall winner for a human annotation record.

    Args:
        record (ProcessedAnnotationRecord): Processed annotation record.
        policy (str): Overall-winner export policy.

    Returns:
        tuple[str, str | None]: Stage-5b-compatible overall winner fields.
    """
    if policy == "human_overall":
        return _choice_to_winner_model(
            record.overall_choice, record.model_a_name, record.model_b_name
        )
    if policy != "dimension_majority":
        raise ValueError(f"Unsupported human overall-winner policy: {policy}")

    win_count_a = sum(
        1 for choice in record.dimension_choices.values() if choice == "A"
    )
    win_count_b = sum(
        1 for choice in record.dimension_choices.values() if choice == "B"
    )
    if win_count_a > win_count_b:
        return "A", record.model_a_name
    if win_count_b > win_count_a:
        return "B", record.model_b_name
    return "tie", None


def _dimension_payload(
    record: ProcessedAnnotationRecord, dimension: str
) -> Dict[str, Any]:
    """
    Build a Stage-5b dimension payload from a processed annotation.

    Args:
        record (ProcessedAnnotationRecord): Processed annotation record.
        dimension (str): Pairwise dimension identifier.

    Returns:
        Dict[str, Any]: Stage-5b-compatible dimension payload.
    """
    winner, winner_model = _choice_to_winner_model(
        record.dimension_choices[dimension],
        record.model_a_name,
        record.model_b_name,
    )
    return {
        "dimension": dimension,
        "winner": winner,
        "winner_model": winner_model,
        "confidence": None,
        "rationale": "",
        "raw_response": "",
        "model_a_name": record.model_a_name,
        "model_b_name": record.model_b_name,
        "position_bias_detected": None,
    }


def _record_to_stage5b_payload(
    config: HumanAnnotationConfig, record: ProcessedAnnotationRecord
) -> Dict[str, Any]:
    """
    Convert a processed annotation into a Stage-5b-compatible record.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        record (ProcessedAnnotationRecord): Processed annotation record.

    Returns:
        Dict[str, Any]: Stage-5b-compatible record payload.
    """
    overall_policy = _DEFAULT_HUMAN_OVERALL_POLICY
    overall_winner_token, overall_winner_model = _determine_human_overall_winner(
        record, overall_policy
    )
    dimension_results = {
        dimension: _dimension_payload(record, dimension)
        for dimension in sorted(record.dimension_choices.keys())
    }
    win_count_a = sum(
        1 for choice in record.dimension_choices.values() if choice == "A"
    )
    win_count_b = sum(
        1 for choice in record.dimension_choices.values() if choice == "B"
    )
    win_count_tie = sum(
        1 for choice in record.dimension_choices.values() if choice == "tie"
    )
    judge_name = f"{config.outputs.judge_name_prefix}_{record.annotator_id}"
    return {
        "user_id": record.persona,
        "task_id": record.raw_task_id,
        "model_a_name": record.model_a_name,
        "model_b_name": record.model_b_name,
        "model_a_output": record.model_a_output,
        "model_b_output": record.model_b_output,
        "dimension_results": dimension_results,
        "overall_winner": overall_winner_model,
        "win_counts": {
            "model_a": win_count_a,
            "model_b": win_count_b,
            "tie": win_count_tie,
        },
        "judge_model_name": judge_name,
        "input_text": record.input_text,
        "pairwise_judgment_type": record.pairwise_judgment_type,
        "metadata": {
            "human_annotation_item_id": record.item_id,
            "annotator_id": record.annotator_id,
            "overall_choice": overall_winner_token,
            "human_overall_choice": record.overall_choice,
            "human_overall_confidence": record.overall_confidence,
            "human_overall_rationale": record.overall_rationale,
            "annotated_dimensions": sorted(record.dimension_choices.keys()),
            "bias_measured": False,
            "human_export_overall_policy": overall_policy,
            **record.metadata,
        },
        "_model_metadata": {
            "judge_model_name": judge_name,
            "raw_judge_model_name": judge_name,
            "pairwise_judgment_type": record.pairwise_judgment_type,
            "source_persona": record.persona,
            "prompt_type": record.prompt_type,
            "position_swap_enabled": False,
            "bias_measured": False,
            "annotated_dimensions": sorted(record.dimension_choices.keys()),
            "generator_model_name": record.metadata.get("generator_model"),
            "filter_model_name": record.metadata.get("filter_model"),
            "human_annotation_study_name": config.outputs.study_name,
        },
    }


def export_human_pairwise_artifacts(
    config: HumanAnnotationConfig,
    records: Iterable[ProcessedAnnotationRecord],
) -> List[Path]:
    """
    Export processed human judgments into canonical Stage-5b artifact paths.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        records (Iterable[ProcessedAnnotationRecord]): Processed annotation records.

    Returns:
        List[Path]: Written artifact paths.
    """
    record_list = sorted(
        list(records),
        key=lambda record: (
            record.persona,
            record.prompt_type,
            record.annotator_id,
            record.task_id,
            record.variant_id,
            str(record.metadata.get("artifact_path") or ""),
            record.item_id,
        ),
    )
    grouped: Dict[
        tuple[str, str, str, str, str, str, str],
        Dict[str, Any],
    ] = {}
    for record in record_list:
        judge_name = f"{config.outputs.judge_name_prefix}_{record.annotator_id}"
        rewritten_record, route = _canonicalize_record_for_export(config, record)
        routing = route["model_routing"]
        key = (
            route["persona"],
            routing.canonical_model_pair,
            judge_name,
            route["generator_model"],
            route["filter_model"],
            route["prompt_type"],
            route["pairwise_judgment_type"],
        )
        group = grouped.setdefault(
            key,
            {
                "route": route,
                "payload": [],
            },
        )
        group["payload"].append(_record_to_stage5b_payload(config, rewritten_record))

    written: List[Path] = []
    for (
        persona,
        _canonical_model_pair,
        judge_name,
        generator_model,
        filter_model,
        prompt_type,
        pairwise_judgment_type,
    ), group in grouped.items():
        route = group["route"]
        routing = route["model_routing"]
        payload = group["payload"]
        out_dir = pairwise_stage_dir(
            base_dir=config.outputs.export_base_dir,
            user_profile_name=persona,
            model_a=routing.route_model_a_token,
            model_b=routing.route_model_b_token,
            judge_model=judge_name,
            generator_model=generator_model,
            filter_model=filter_model,
            judgment_type=pairwise_judgment_type,
            prompt_type=prompt_type,
        )
        detail = normalize_token(
            f"{config.outputs.study_name}_{judge_name}_{prompt_type}_v00"
        )
        output_path = out_dir / f"pairwise-comparison_human_annotation_{detail}.json"
        save_json(payload, str(output_path))
        written.append(output_path)

    logger.info(
        "Exported %d human Stage-5b artifacts from %d processed records",
        len(written),
        len(record_list),
    )
    logger.info(
        "Export composition: personas=%s human_judges=%s",
        dict(Counter(record.persona for record in record_list)),
        dict(
            Counter(
                f"{config.outputs.judge_name_prefix}_{record.annotator_id}"
                for record in record_list
            )
        ),
    )
    return written
