"""Sampling utilities for standalone human-annotation studies."""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.vibe_testing.human_annotation.schemas import (
    AllocationConfig,
    AnnotatorConfig,
    FilterConfig,
    PairwiseCandidateRecord,
    SampledAnnotationItem,
    SampleTypeSpec,
    SamplingTarget,
)
from src.vibe_testing.human_annotation.sample_type_utils import (
    candidate_field_value,
    field_map_label,
    field_map_signature,
    matches_field_map,
    normalize_field_map,
    sample_type_key,
    sample_type_label,
    sample_type_map,
)
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
)
from src.vibe_testing.pathing import canonicalize_pairwise_model_routing
from src.vibe_testing.utils import load_json

logger = logging.getLogger(__name__)
_VARIATION_TOKEN = "::variation::"


def _stable_item_token(candidate: PairwiseCandidateRecord) -> str:
    """
    Build a stable sampling token for a candidate.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.

    Returns:
        str: Stable token used for ordering and hashing.
    """
    return "|".join(
        [
            candidate.source_key,
            candidate.judge_dir_name,
            candidate.raw_task_id,
            candidate.artifact_path,
            str(candidate.artifact_index),
        ]
    )


def _split_selection_plan_variant_fields(
    row: Dict[str, str],
) -> tuple[str, str]:
    """
    Resolve task and variant identifiers from a selection-plan row.

    Args:
        row (Dict[str, str]): Selection-plan CSV row.

    Returns:
        tuple[str, str]: Base task id and variant id.
    """
    raw_task_id = str(row.get("raw_task_id") or "").strip()
    if raw_task_id and _VARIATION_TOKEN in raw_task_id:
        task_id, variant_id = raw_task_id.split(_VARIATION_TOKEN, 1)
        return task_id, variant_id or task_id
    task_id = str(row.get("task_id") or "").strip()
    variant_id = str(row.get("variant_id") or "").strip()
    if task_id:
        return task_id, variant_id or task_id
    return raw_task_id, variant_id or raw_task_id


def _source_key_from_selection_plan_row(row: Dict[str, str]) -> str | None:
    """
    Recover a candidate source key from a prior `selection_plan.csv` row.

    Args:
        row (Dict[str, str]): Parsed CSV row.

    Returns:
        str | None: Canonical source key, or None when the row is unusable.
    """
    source_key = str(row.get("source_key") or "").strip()
    if source_key:
        return source_key
    persona = str(row.get("persona") or "").strip()
    prompt_type = str(row.get("prompt_type") or "").strip()
    if not persona or not prompt_type:
        return None
    task_id, variant_id = _split_selection_plan_variant_fields(row)
    if not task_id or not variant_id:
        return None
    model_a_name = str(row.get("model_a_name") or "").strip()
    model_b_name = str(row.get("model_b_name") or "").strip()
    if not model_a_name or not model_b_name:
        return None
    routing = canonicalize_pairwise_model_routing(model_a_name, model_b_name)
    pairwise_judgment_type = normalize_pairwise_judgment_type(
        str(row.get("pairwise_judgment_type") or PAIRWISE_JUDGMENT_TYPE_PERSONA)
    )
    return "|".join(
        [
            persona,
            prompt_type,
            pairwise_judgment_type,
            routing.canonical_model_a,
            routing.canonical_model_b,
            task_id,
            variant_id,
        ]
    )


def _load_prior_selection_plan_source_keys(paths: Sequence[Path]) -> set[str]:
    """
    Load source keys from previously generated selection plans.

    Args:
        paths (Sequence[Path]): Prior `selection_plan.csv` paths.

    Returns:
        set[str]: Source keys present in prior plans.
    """
    source_keys: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Prior selection plan path not found: {resolved}")
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_source_key = _source_key_from_selection_plan_row(row)
                if row_source_key:
                    source_keys.add(row_source_key)
    return source_keys


def _load_manifest_source_keys(paths: Sequence[Path]) -> set[str]:
    """
    Load sampled source keys from prior study manifests.

    Args:
        paths (Sequence[Path]): Prior manifest paths.

    Returns:
        set[str]: Source keys present in the manifests.
    """
    source_keys: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Prior manifest path not found: {resolved}")
        payload = load_json(str(resolved))
        sampled_items = payload.get("sampled_items") if isinstance(payload, dict) else None
        if not isinstance(sampled_items, list):
            raise ValueError(
                f"Prior manifest must contain a 'sampled_items' list: {resolved}"
            )
        for item in sampled_items:
            candidate_payload = item.get("candidate") if isinstance(item, dict) else None
            source_key = (
                candidate_payload.get("source_key")
                if isinstance(candidate_payload, dict)
                else None
            )
            if source_key:
                source_keys.add(str(source_key))
    return source_keys


def include_sample_type_candidates(
    candidates: Sequence[PairwiseCandidateRecord],
    filter_config: FilterConfig,
) -> tuple[List[PairwiseCandidateRecord], Dict[str, Any]]:
    """
    Restrict candidates to an explicit set of exact sample types when requested.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible candidate rows.
        filter_config (FilterConfig): Study filter configuration.

    Returns:
        tuple[List[PairwiseCandidateRecord], Dict[str, Any]]: Filtered candidates and
            audit metadata.

    Raises:
        ValueError: If requested sample types are missing from the eligible pool.
    """
    selectors = list(filter_config.include_sample_types)
    if not selectors:
        return list(candidates), {
            "enabled": False,
            "requested_types": [],
            "requested_type_counts": {},
            "requested_total_weight": 0,
            "matched_source_items": 0,
            "available_source_items_by_type": {},
            "missing_types": [],
        }
    requested_type_counts: Dict[str, int] = Counter(
        selector.label for selector in selectors
    )
    available_source_items_by_type: Dict[str, int] = {}
    for selector in selectors:
        available_source_items_by_type[selector.label] = len(
            {
                candidate.source_key
                for candidate in candidates
                if matches_field_map(candidate, selector.where)
            }
        )
    filtered = [
        candidate
        for candidate in candidates
        if any(matches_field_map(candidate, selector.where) for selector in selectors)
    ]
    matched_labels = {
        "::".join(
            f"{field_name}={candidate_field_value(candidate, field_name)}"
            for field_name in selector.where.keys()
        )
        for selector in selectors
        for candidate in filtered
        if matches_field_map(candidate, selector.where)
    }
    requested_labels = [
        "::".join(
            f"{field_name}={allowed_values[0]}"
            if len(allowed_values) == 1
            else f"{field_name}={','.join(allowed_values)}"
            for field_name, allowed_values in selector.where.items()
        )
        for selector in selectors
    ]
    missing_types = [
        requested_labels[index]
        for index, selector in enumerate(selectors)
        if not any(matches_field_map(candidate, selector.where) for candidate in candidates)
    ]
    if missing_types:
        raise ValueError(
            "Requested include_sample_types are missing from the eligible pool: "
            f"{missing_types}"
        )
    logger.info(
        "Exact sample-type inclusion: input_rows=%d kept_rows=%d requested_types=%d",
        len(candidates),
        len(filtered),
        len(selectors),
    )
    return filtered, {
        "enabled": True,
        "requested_types": [selector.to_dict() for selector in selectors],
        "requested_type_counts": dict(sorted(requested_type_counts.items())),
        "requested_total_weight": len(selectors),
        "matched_rows": len(filtered),
        "matched_source_items": len({candidate.source_key for candidate in filtered}),
        "available_source_items_by_type": dict(
            sorted(available_source_items_by_type.items())
        ),
        "missing_types": missing_types,
        "matched_type_labels": sorted(matched_labels),
    }


def resolve_repeat_candidates(
    candidates: Sequence[PairwiseCandidateRecord],
    filter_config: FilterConfig,
) -> tuple[List[PairwiseCandidateRecord], set[str], Dict[str, Any]]:
    """
    Resolve explicit repeat requests from source keys, selection plans, or manifests.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible candidate rows.
        filter_config (FilterConfig): Study filter configuration.

    Returns:
        tuple[List[PairwiseCandidateRecord], set[str], Dict[str, Any]]: Candidate rows
            after any include-only restriction, pinned source keys, and audit metadata.

    Raises:
        ValueError: If requested repeated source keys are missing from the eligible pool.
    """
    explicit_source_keys = set(filter_config.include_prior_sample_ids)
    requested_source_keys: set[str] = set()
    repeat_sources_by_source_key: Dict[str, List[str]] = defaultdict(list)
    for source_key in explicit_source_keys:
        requested_source_keys.add(source_key)
        repeat_sources_by_source_key[source_key].append("explicit")
    for path in filter_config.include_from_selection_plan_paths:
        for source_key in _load_prior_selection_plan_source_keys([path]):
            requested_source_keys.add(source_key)
            repeat_sources_by_source_key[source_key].append(str(path.expanduser()))
    for path in filter_config.include_from_manifest_paths:
        for source_key in _load_manifest_source_keys([path]):
            requested_source_keys.add(source_key)
            repeat_sources_by_source_key[source_key].append(str(path.expanduser()))
    if not requested_source_keys:
        return list(candidates), set(), {
            "enabled": False,
            "repeat_mode": filter_config.repeat_mode,
            "requested_source_keys": [],
            "matched_source_keys": [],
            "missing_source_keys": [],
            "repeat_sources_by_source_key": {},
        }
    available_source_keys = {candidate.source_key for candidate in candidates}
    missing_source_keys = sorted(requested_source_keys - available_source_keys)
    if missing_source_keys:
        raise ValueError(
            "Requested repeated source_key values are missing from the eligible pool: "
            f"{missing_source_keys}"
        )
    if filter_config.repeat_mode == "include_only":
        filtered = [
            candidate
            for candidate in candidates
            if candidate.source_key in requested_source_keys
        ]
        pinned_source_keys = set(requested_source_keys)
    else:
        filtered = list(candidates)
        pinned_source_keys = set(requested_source_keys)
    logger.info(
        "Repeat inclusion: input_rows=%d kept_rows=%d pinned_source_items=%d repeat_mode=%s",
        len(candidates),
        len(filtered),
        len(pinned_source_keys),
        filter_config.repeat_mode,
    )
    return filtered, pinned_source_keys, {
        "enabled": True,
        "repeat_mode": filter_config.repeat_mode,
        "requested_source_keys": sorted(requested_source_keys),
        "matched_source_keys": sorted(requested_source_keys),
        "missing_source_keys": [],
        "included_from_selection_plan_paths": [
            str(path.expanduser()) for path in filter_config.include_from_selection_plan_paths
        ],
        "included_from_manifest_paths": [
            str(path.expanduser()) for path in filter_config.include_from_manifest_paths
        ],
        "repeat_sources_by_source_key": {
            source_key: sorted(set(sources))
            for source_key, sources in repeat_sources_by_source_key.items()
        },
        "pinned_source_items": len(pinned_source_keys),
        "kept_rows": len(filtered),
        "kept_source_items": len({candidate.source_key for candidate in filtered}),
    }


def exclude_prior_selection_plan_candidates(
    candidates: Sequence[PairwiseCandidateRecord],
    filter_config: FilterConfig,
    *,
    protected_source_keys: Optional[set[str]] = None,
) -> tuple[List[PairwiseCandidateRecord], Dict[str, Any]]:
    """
    Exclude candidate rows whose source items appear in prior selection plans.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible candidate rows.
        filter_config (FilterConfig): Study filter configuration.

    Returns:
        tuple[List[PairwiseCandidateRecord], Dict[str, Any]]: Filtered candidates and
            JSON-friendly audit metadata.
    """
    plan_paths = list(filter_config.prior_selection_plan_paths)
    if not filter_config.exclude_prior_selection_plans:
        return list(candidates), {
            "enabled": False,
            "plan_paths": [str(path) for path in plan_paths],
            "excluded_rows": 0,
            "excluded_source_items": 0,
        }
    if not plan_paths:
        logger.warning(
            "filters.exclude_prior_selection_plans is enabled, but no prior_selection_plan_paths were provided."
        )
        return list(candidates), {
            "enabled": True,
            "plan_paths": [],
            "excluded_rows": 0,
            "excluded_source_items": 0,
        }

    excluded_source_keys = _load_prior_selection_plan_source_keys(plan_paths)
    protected = set(protected_source_keys or set())
    filtered_candidates = [
        candidate
        for candidate in candidates
        if candidate.source_key not in excluded_source_keys
        or candidate.source_key in protected
    ]
    excluded_candidates = [
        candidate
        for candidate in candidates
        if candidate.source_key in excluded_source_keys
        and candidate.source_key not in protected
    ]
    logger.info(
        "Prior selection-plan exclusion: input_rows=%d kept_rows=%d excluded_rows=%d excluded_source_items=%d plans=%d",
        len(candidates),
        len(filtered_candidates),
        len(excluded_candidates),
        len({candidate.source_key for candidate in excluded_candidates}),
        len(plan_paths),
    )
    return filtered_candidates, {
        "enabled": True,
        "plan_paths": [str(path.expanduser()) for path in plan_paths],
        "excluded_rows": len(excluded_candidates),
        "excluded_source_items": len(
            {candidate.source_key for candidate in excluded_candidates}
        ),
        "protected_source_items": len(protected & excluded_source_keys),
        "kept_rows": len(filtered_candidates),
        "kept_source_items": len(
            {candidate.source_key for candidate in filtered_candidates}
        ),
        "excluded_rows_by_persona": dict(
            Counter(candidate.persona for candidate in excluded_candidates)
        ),
        "excluded_rows_by_prompt_type": dict(
            Counter(candidate.prompt_type for candidate in excluded_candidates)
        ),
        "excluded_rows_by_model_pair": dict(
            Counter(candidate.model_pair for candidate in excluded_candidates)
        ),
    }


def _make_item_id(candidate: PairwiseCandidateRecord, selection_target: str) -> str:
    """
    Build a deterministic human-annotation item id.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        selection_target (str): Sampling target label.

    Returns:
        str: Stable item identifier.
    """
    digest = hashlib.sha1(
        f"{selection_target}|{_stable_item_token(candidate)}".encode("utf-8")
    ).hexdigest()
    return f"ha_{digest[:12]}"


def _sort_candidates(
    candidates: Iterable[PairwiseCandidateRecord],
) -> List[PairwiseCandidateRecord]:
    """
    Stably sort candidate records.

    Args:
        candidates (Iterable[PairwiseCandidateRecord]): Candidate records.

    Returns:
        List[PairwiseCandidateRecord]: Sorted candidates.
    """
    return sorted(candidates, key=_stable_item_token)


def deduplicate_candidates(
    candidates: Sequence[PairwiseCandidateRecord],
) -> List[PairwiseCandidateRecord]:
    """
    Deduplicate candidate rows by source item, preferring stable earliest records.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Candidate records.

    Returns:
        List[PairwiseCandidateRecord]: Deduplicated candidates.
    """
    deduped: Dict[str, PairwiseCandidateRecord] = {}
    # Sampling happens at the source-item level, so multiple judge rows for the same
    # item collapse to the first stable representative before allocation.
    for candidate in _sort_candidates(candidates):
        deduped.setdefault(candidate.source_key, candidate)
    return list(deduped.values())


def apply_stride(
    candidates: Sequence[PairwiseCandidateRecord], allocation: AllocationConfig
) -> List[PairwiseCandidateRecord]:
    """
    Apply a deterministic stride selection over sorted candidates.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Candidate records.
        allocation (AllocationConfig): Sampling allocation configuration.

    Returns:
        List[PairwiseCandidateRecord]: Stride-filtered candidates.
    """
    sorted_candidates = _sort_candidates(candidates)
    return sorted_candidates[allocation.stride_offset :: allocation.stride]


def _group_key(
    candidate: PairwiseCandidateRecord, fields: Sequence[str]
) -> Tuple[str, ...]:
    """
    Build a grouping key from requested candidate fields.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        fields (Sequence[str]): Field names used for grouping.

    Returns:
        Tuple[str, ...]: Comparable grouping tuple.

    Raises:
        ValueError: If a requested field is missing.
    """
    return sample_type_key(candidate, fields)


def _shuffle_in_place(
    values: List[PairwiseCandidateRecord], random_seed: int
) -> List[PairwiseCandidateRecord]:
    """
    Deterministically shuffle a candidate list.

    Args:
        values (List[PairwiseCandidateRecord]): Candidate list.
        random_seed (int): Seed used for reproducible shuffling.

    Returns:
        List[PairwiseCandidateRecord]: Shuffled list.
    """
    rng = random.Random(random_seed)
    copy = list(values)
    rng.shuffle(copy)
    return copy


def _build_sampling_pool(
    candidates: Sequence[PairwiseCandidateRecord],
    allocation: AllocationConfig,
) -> List[PairwiseCandidateRecord]:
    """
    Build the canonical eligible source-item pool used for sampling.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible candidate rows.
        allocation (AllocationConfig): Allocation configuration.

    Returns:
        List[PairwiseCandidateRecord]: Post-dedup, post-stride source-item pool.
    """
    deduped_candidates = deduplicate_candidates(candidates)
    return apply_stride(deduped_candidates, allocation)


def _eligible_count_by_field(
    candidates: Sequence[PairwiseCandidateRecord], field_name: str
) -> Dict[str, int]:
    """
    Count eligible candidates by field value.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible source-item pool.
        field_name (str): Candidate field to count.

    Returns:
        Dict[str, int]: Field-value support counts.
    """
    return dict(
        sorted(
            Counter(
                str(getattr(candidate, field_name)) for candidate in candidates
            ).items()
        )
    )


def _field_value(candidate: PairwiseCandidateRecord, field_name: str) -> str:
    """
    Read a string field value from a candidate record.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        field_name (str): Field to read.

    Returns:
        str: String field value.
    """
    return candidate_field_value(candidate, field_name)


def _ordered_field_values(
    candidates: Sequence[PairwiseCandidateRecord], fields: Sequence[str]
) -> Dict[str, List[str]]:
    """
    Resolve sorted distinct values for each requested balanced field.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible source-item pool.
        fields (Sequence[str]): Requested balanced fields.

    Returns:
        Dict[str, List[str]]: Sorted field values.
    """
    ordered: Dict[str, List[str]] = {}
    for field_name in fields:
        ordered[field_name] = sorted(
            {_field_value(candidate, field_name) for candidate in candidates}
        )
    return ordered


def _format_equal_allocation_failure(
    reason: str,
    *,
    fields: Sequence[str],
    total_samples: Optional[int],
    field_values: Dict[str, List[str]],
    eligible_counts_by_field: Dict[str, Dict[str, int]],
    stratum_support: Dict[str, int],
    stratum_quota: Optional[int],
) -> str:
    """
    Format a fail-loud equal-allocation error message.

    Args:
        reason (str): Human-readable failure reason.
        fields (Sequence[str]): Requested balanced fields.
        total_samples (Optional[int]): Requested total sample count.
        field_values (Dict[str, List[str]]): Distinct observed values per field.
        eligible_counts_by_field (Dict[str, Dict[str, int]]): Eligible support counts.
        stratum_support (Dict[str, int]): Eligible counts per joint stratum.
        stratum_quota (Optional[int]): Required quota for each joint stratum.

    Returns:
        str: Detailed error message.
    """
    return (
        "Equal allocation is infeasible. "
        f"reason={reason}; "
        f"equal_allocation_by={list(fields)}; "
        f"total_samples={total_samples}; "
        f"field_values={field_values}; "
        f"eligible_counts_by_field={eligible_counts_by_field}; "
        f"stratum_support={stratum_support}; "
        f"stratum_quota={stratum_quota}"
    )


def _build_equal_allocation_spec(
    candidates: Sequence[PairwiseCandidateRecord],
    allocation: AllocationConfig,
) -> Dict[str, Any]:
    """
    Build and validate strict equal-allocation quotas over the full joint strata.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible source-item pool.
        allocation (AllocationConfig): Allocation configuration.

    Returns:
        Dict[str, Any]: Equal-allocation spec and audit metadata.

    Raises:
        ValueError: If strict equal allocation is infeasible before selection.
    """
    fields = list(allocation.equal_allocation_by)
    if allocation.total_samples is None:
        raise ValueError(
            "allocation.total_samples is required when using equal_allocation_by."
        )
    if not candidates:
        raise ValueError(
            "Equal allocation is infeasible because the eligible source-item pool is empty."
        )

    field_values = _ordered_field_values(candidates, fields)
    eligible_counts_by_field = {
        field_name: _eligible_count_by_field(candidates, field_name)
        for field_name in fields
    }
    for field_name in fields:
        if not field_values[field_name]:
            raise ValueError(
                _format_equal_allocation_failure(
                    f"field '{field_name}' has no eligible values",
                    fields=fields,
                    total_samples=allocation.total_samples,
                    field_values=field_values,
                    eligible_counts_by_field=eligible_counts_by_field,
                    stratum_support={},
                    stratum_quota=None,
                )
            )
    ordered_strata = list(product(*(field_values[field_name] for field_name in fields)))
    if allocation.total_samples % len(ordered_strata) != 0:
        raise ValueError(
            _format_equal_allocation_failure(
                "allocation.total_samples is not divisible by the number of required joint strata",
                fields=fields,
                total_samples=allocation.total_samples,
                field_values=field_values,
                eligible_counts_by_field=eligible_counts_by_field,
                stratum_support={},
                stratum_quota=None,
            )
        )
    per_stratum_quota = allocation.total_samples // len(ordered_strata)
    strata_to_candidates: Dict[Tuple[str, ...], List[PairwiseCandidateRecord]] = {
        stratum: [] for stratum in ordered_strata
    }
    for candidate in candidates:
        strata_to_candidates[_group_key(candidate, fields)].append(candidate)
    stratum_support = {
        "::".join(
            f"{field_name}={value}" for field_name, value in zip(fields, stratum)
        ): len(strata_to_candidates[stratum])
        for stratum in ordered_strata
    }
    missing_strata = [label for label, count in stratum_support.items() if count == 0]
    if missing_strata:
        raise ValueError(
            _format_equal_allocation_failure(
                f"required strata are missing from the eligible pool: {missing_strata}",
                fields=fields,
                total_samples=allocation.total_samples,
                field_values=field_values,
                eligible_counts_by_field=eligible_counts_by_field,
                stratum_support=stratum_support,
                stratum_quota=per_stratum_quota,
            )
        )
    thin_strata = [
        label for label, count in stratum_support.items() if count < per_stratum_quota
    ]
    if thin_strata:
        raise ValueError(
            _format_equal_allocation_failure(
                f"required strata do not have enough support for quota {per_stratum_quota}: {thin_strata}",
                fields=fields,
                total_samples=allocation.total_samples,
                field_values=field_values,
                eligible_counts_by_field=eligible_counts_by_field,
                stratum_support=stratum_support,
                stratum_quota=per_stratum_quota,
            )
        )
    return {
        "fields": fields,
        "field_values": field_values,
        "eligible_counts_by_field": eligible_counts_by_field,
        "total_samples": allocation.total_samples,
        "ordered_strata": ordered_strata,
        "strata_to_candidates": strata_to_candidates,
        "stratum_support": stratum_support,
        "per_stratum_quota": per_stratum_quota,
    }


def _attach_marginal_balance_metadata(
    selected: Sequence[SampledAnnotationItem],
    *,
    balance_audit: Dict[str, Any],
) -> List[SampledAnnotationItem]:
    """
    Attach equal-allocation audit metadata to sampled items.

    Args:
        selected (Sequence[SampledAnnotationItem]): Selected items.
        balance_audit (Dict[str, Any]): Equal-allocation audit payload.

    Returns:
        List[SampledAnnotationItem]: Enriched sampled items.
    """
    enriched: List[SampledAnnotationItem] = []
    fields = list(balance_audit.get("fields", []))
    for item in selected:
        enriched.append(
            replace(
                item,
                selection_metadata={
                    **item.selection_metadata,
                    "marginal_balance_mode": str(
                        balance_audit.get("mode") or "equal_strata"
                    ),
                    "marginal_balance_fields": ",".join(fields),
                    "marginal_balance_audit": balance_audit,
                },
            )
        )
    return enriched


def _select_n(
    values: Sequence[PairwiseCandidateRecord],
    n_samples: int,
    random_seed: int,
    selection_target: str,
) -> List[SampledAnnotationItem]:
    """
    Select a fixed number of candidates and wrap them as sampled items.

    Args:
        values (Sequence[PairwiseCandidateRecord]): Eligible candidates.
        n_samples (int): Number of items to select.
        random_seed (int): Seed used for deterministic shuffling.
        selection_target (str): Sampling target label.

    Returns:
        List[SampledAnnotationItem]: Selected items.

    Raises:
        ValueError: If there are not enough eligible candidates.
    """
    ordered = _shuffle_in_place(_sort_candidates(values), random_seed)
    if len(ordered) < n_samples:
        raise ValueError(
            f"Sampling target '{selection_target}' requires {n_samples} items, "
            f"but only {len(ordered)} are eligible."
        )
    selected: List[SampledAnnotationItem] = []
    for selection_rank, candidate in enumerate(ordered[:n_samples], start=1):
        selected.append(
            SampledAnnotationItem(
                item_id=_make_item_id(candidate, selection_target),
                candidate=candidate,
                selection_target=selection_target,
                selection_rank=selection_rank,
                selection_metadata={
                    "selection_target": selection_target,
                    "random_seed": random_seed,
                },
            )
        )
    return selected


def _select_fixed_candidates(
    values: Sequence[PairwiseCandidateRecord],
    selection_target: str,
) -> List[SampledAnnotationItem]:
    """
    Wrap a fixed candidate set as sampled items without further random sampling.

    Args:
        values (Sequence[PairwiseCandidateRecord]): Preselected candidates.
        selection_target (str): Selection-target label.

    Returns:
        List[SampledAnnotationItem]: Wrapped sampled items.
    """
    selected: List[SampledAnnotationItem] = []
    for selection_rank, candidate in enumerate(_sort_candidates(values), start=1):
        selected.append(
            SampledAnnotationItem(
                item_id=_make_item_id(candidate, selection_target),
                candidate=candidate,
                selection_target=selection_target,
                selection_rank=selection_rank,
                selection_metadata={
                    "selection_target": selection_target,
                    "selection_mode": "pinned_repeat",
                },
            )
        )
    return selected


def _build_judge_selection_metadata(
    candidates: Sequence[PairwiseCandidateRecord],
) -> Dict[str, Dict[str, Any]]:
    """
    Build per-source-item judge summaries for selection-plan auditing.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Candidate rows still eligible
            for sampling.

    Returns:
        Dict[str, Dict[str, Any]]: Selection metadata keyed by source key.
    """
    by_source_key: Dict[str, List[PairwiseCandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        by_source_key[candidate.source_key].append(candidate)

    metadata_by_source_key: Dict[str, Dict[str, Any]] = {}
    for source_key, source_candidates in by_source_key.items():
        judge_summary: Dict[str, Any] = {}
        judge_names: List[str] = []
        for candidate in sorted(source_candidates, key=_stable_item_token):
            judge_column_prefix = f"judge_{candidate.judge_dir_name}"
            judge_names.append(candidate.judge_dir_name)
            judge_summary[f"{judge_column_prefix}_model_name"] = (
                candidate.judge_model_name
            )
            judge_summary[f"{judge_column_prefix}_overall_winner"] = (
                candidate.overall_winner or "tie"
            )
            for dimension in sorted(candidate.dimension_results.keys()):
                dimension_result = candidate.dimension_results.get(dimension) or {}
                judge_summary[f"{judge_column_prefix}_{dimension}_winner"] = str(
                    dimension_result.get("winner") or "tie"
                )
                judge_summary[f"{judge_column_prefix}_{dimension}_winner_model"] = str(
                    dimension_result.get("winner_model") or ""
                )
        representative_candidate = source_candidates[0]
        metadata_by_source_key[source_key] = {
            "source_key": source_key,
            "model_pair": representative_candidate.model_pair,
            "llm_judge_names": ",".join(sorted(set(judge_names))),
            "llm_judge_count": len(set(judge_names)),
            **judge_summary,
        }
    return metadata_by_source_key


def _attach_selection_metadata(
    selected: Sequence[SampledAnnotationItem],
    candidates: Sequence[PairwiseCandidateRecord],
) -> List[SampledAnnotationItem]:
    """
    Attach per-source-item judge summaries to sampled items.

    Args:
        selected (Sequence[SampledAnnotationItem]): Sampled items.
        candidates (Sequence[PairwiseCandidateRecord]): Candidate rows eligible for
            sampling.

    Returns:
        List[SampledAnnotationItem]: Sampled items with enriched selection metadata.
    """
    metadata_by_source_key = _build_judge_selection_metadata(candidates)
    enriched: List[SampledAnnotationItem] = []
    for item in selected:
        extra_metadata = metadata_by_source_key.get(item.candidate.source_key, {})
        enriched.append(
            replace(
                item,
                selection_metadata={
                    **item.selection_metadata,
                    **extra_metadata,
                },
            )
        )
    return enriched


def _attach_repeat_metadata(
    selected: Sequence[SampledAnnotationItem],
    *,
    pinned_source_keys: set[str],
    repeat_audit: Dict[str, Any],
) -> List[SampledAnnotationItem]:
    """
    Attach repeated-run provenance to sampled items.

    Args:
        selected (Sequence[SampledAnnotationItem]): Sampled items.
        pinned_source_keys (set[str]): Explicit repeated source keys.
        repeat_audit (Dict[str, Any]): Repeat-resolution audit metadata.

    Returns:
        List[SampledAnnotationItem]: Enriched sampled items.
    """
    if not repeat_audit.get("enabled"):
        return [
            replace(
                item,
                selection_metadata={
                    **item.selection_metadata,
                    "repeat_status": "fresh",
                    "repeat_source_key": item.candidate.source_key,
                    "repeat_source_plan": "",
                },
            )
            for item in selected
        ]
    repeat_sources_by_source_key = dict(
        repeat_audit.get("repeat_sources_by_source_key", {})
    )
    repeat_mode = str(repeat_audit.get("repeat_mode") or "include_only")
    enriched: List[SampledAnnotationItem] = []
    for item in selected:
        source_key = item.candidate.source_key
        repeat_sources = repeat_sources_by_source_key.get(source_key, [])
        enriched.append(
            replace(
                item,
                selection_metadata={
                    **item.selection_metadata,
                    "repeat_mode": repeat_mode,
                    "repeat_status": (
                        "repeated" if source_key in pinned_source_keys else "fresh"
                    ),
                    "repeat_source_key": source_key,
                    "repeat_source_plan": ",".join(repeat_sources),
                },
            )
        )
    return enriched


def _attach_sample_type_metadata(
    selected: Sequence[SampledAnnotationItem],
    sample_type_fields: Sequence[str],
) -> List[SampledAnnotationItem]:
    """
    Attach sample-type keys used for assignment-time balancing and auditing.

    Args:
        selected (Sequence[SampledAnnotationItem]): Sampled items.
        sample_type_fields (Sequence[str]): Fields that define the shared sample type.

    Returns:
        List[SampledAnnotationItem]: Enriched sampled items.
    """
    if not sample_type_fields:
        return list(selected)
    enriched: List[SampledAnnotationItem] = []
    for item in selected:
        enriched.append(
            replace(
                item,
                selection_metadata={
                    **item.selection_metadata,
                    "sample_type_fields": ",".join(sample_type_fields),
                    "sample_type_key": sample_type_label(
                        item.candidate, sample_type_fields
                    ),
                },
            )
        )
    return enriched


def _attach_item_role_metadata(
    selected: Sequence[SampledAnnotationItem],
    *,
    item_role: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[SampledAnnotationItem]:
    """
    Attach a stable sampled-item role for downstream auditing.

    Args:
        selected (Sequence[SampledAnnotationItem]): Sampled items.
        item_role (str): High-level role such as `calibration` or `regular`.
        extra_metadata (Optional[Dict[str, Any]]): Extra metadata to copy onto each item.

    Returns:
        List[SampledAnnotationItem]: Enriched sampled items.
    """
    extra_metadata = dict(extra_metadata or {})
    enriched: List[SampledAnnotationItem] = []
    for item in selected:
        enriched.append(
            replace(
                item,
                selection_metadata={
                    **item.selection_metadata,
                    "item_role": item_role,
                    **extra_metadata,
                },
            )
        )
    return enriched


def _selector_count_map(
    selectors: Sequence[SampleTypeSpec],
) -> Counter[tuple[tuple[str, tuple[str, ...]], ...]]:
    """
    Count selector occurrences by canonical signature.

    Args:
        selectors (Sequence[SampleTypeSpec]): Selector list.

    Returns:
        Counter[tuple[tuple[str, tuple[str, ...]], ...]]: Counts by selector signature.
    """
    return Counter(selector.signature for selector in selectors)


def _subtract_selector_instances(
    include_sample_types: Sequence[SampleTypeSpec],
    calibration_sample_types: Sequence[SampleTypeSpec],
) -> List[SampleTypeSpec]:
    """
    Remove one occurrence of each calibration selector from the packet template.

    Args:
        include_sample_types (Sequence[SampleTypeSpec]): Full per-packet type template.
        calibration_sample_types (Sequence[SampleTypeSpec]): Fixed calibration selectors.

    Returns:
        List[SampleTypeSpec]: Remaining regular packet template selectors.

    Raises:
        ValueError: If a calibration selector is not present in the packet template.
    """
    if not calibration_sample_types:
        return list(include_sample_types)
    include_counts = _selector_count_map(include_sample_types)
    calibration_counts = _selector_count_map(calibration_sample_types)
    missing = [
        selector.label
        for selector in calibration_sample_types
        if include_counts.get(selector.signature, 0)
        < calibration_counts.get(selector.signature, 0)
    ]
    if missing:
        raise ValueError(
            "Calibration selectors must be included in filters.include_sample_types. "
            f"Missing selectors: {sorted(set(missing))}"
        )
    to_remove = Counter(calibration_counts)
    remaining: List[SampleTypeSpec] = []
    for selector in include_sample_types:
        if to_remove.get(selector.signature, 0) > 0:
            to_remove[selector.signature] -= 1
            continue
        remaining.append(selector)
    return remaining


def _select_calibration_candidates(
    candidates: Sequence[PairwiseCandidateRecord],
    calibration_sample_types: Sequence[SampleTypeSpec],
    *,
    random_seed: int,
) -> List[SampledAnnotationItem]:
    """
    Select fixed calibration items from exact sample-type selectors.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible post-filter candidates.
        calibration_sample_types (Sequence[SampleTypeSpec]): Fixed calibration selectors.
        random_seed (int): Seed used for deterministic selection.

    Returns:
        List[SampledAnnotationItem]: Selected calibration sampled items.

    Raises:
        ValueError: If any calibration selector cannot be satisfied.
    """
    if not calibration_sample_types:
        return []
    aggregated: Dict[
        tuple[tuple[str, tuple[str, ...]], ...], Dict[str, Any]
    ] = {}
    for selector in calibration_sample_types:
        entry = aggregated.setdefault(
            selector.signature,
            {
                "selector": selector,
                "count": 0,
            },
        )
        entry["count"] += 1
    selected_items: List[SampledAnnotationItem] = []
    used_source_keys: set[str] = set()
    for selector_index, entry in enumerate(
        sorted(aggregated.values(), key=lambda value: value["selector"].label)
    ):
        selector = entry["selector"]
        requested_count = int(entry["count"])
        matching = [
            candidate
            for candidate in candidates
            if matches_field_map(candidate, selector.where)
            and candidate.source_key not in used_source_keys
        ]
        picked = _select_n(
            matching,
            requested_count,
            random_seed + selector_index,
            f"calibration::{selector.label}",
        )
        selected_items.extend(picked)
        used_source_keys.update(item.candidate.source_key for item in picked)
    return [
        replace(item, selection_rank=selection_rank)
        for selection_rank, item in enumerate(selected_items, start=1)
    ]


def _build_include_sample_type_quota_spec(
    candidates: Sequence[PairwiseCandidateRecord],
    include_sample_types: Sequence[Any],
    allocation: AllocationConfig,
) -> Dict[str, Any]:
    """
    Convert repeated `include_sample_types` selectors into weighted sampling targets.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Post-dedup, post-stride pool.
        include_sample_types (Sequence[Any]): Configured selector objects.
        allocation (AllocationConfig): Allocation configuration.

    Returns:
        Dict[str, Any]: Weighted target spec and audit payload.

    Raises:
        ValueError: If the requested weighted selection is infeasible.
    """
    selectors = list(include_sample_types)
    if not selectors:
        return {
            "enabled": False,
            "targets": [],
            "requested_total_weight": 0,
            "requested_type_counts": {},
            "requested_quota_by_type": {},
            "available_source_items_by_type": {},
            "override_equal_allocation": False,
            "override_fields": [],
        }
    if allocation.total_samples is None:
        raise ValueError(
            "allocation.total_samples is required when using filters.include_sample_types."
        )

    aggregated_selectors: Dict[
        tuple[tuple[str, tuple[str, ...]], ...], Dict[str, Any]
    ] = {}
    for selector in selectors:
        signature = field_map_signature(selector.where)
        entry = aggregated_selectors.setdefault(
            signature,
            {
                "where": normalize_field_map(selector.where),
                "label": field_map_label(selector.where),
                "weight": 0,
            },
        )
        entry["weight"] += 1

    total_weight = sum(int(entry["weight"]) for entry in aggregated_selectors.values())
    if allocation.total_samples % total_weight != 0:
        raise ValueError(
            "allocation.total_samples must be divisible by the total include_sample_types "
            f"weight. total_samples={allocation.total_samples} total_weight={total_weight}"
        )
    unit_quota = allocation.total_samples // total_weight

    overlaps: Dict[str, List[str]] = defaultdict(list)
    targets: List[SamplingTarget] = []
    requested_quota_by_type: Dict[str, int] = {}
    available_source_items_by_type: Dict[str, int] = {}
    source_to_labels: Dict[str, set[str]] = defaultdict(set)
    for entry in aggregated_selectors.values():
        matching_candidates = [
            candidate
            for candidate in candidates
            if matches_field_map(candidate, entry["where"])
        ]
        label = str(entry["label"])
        available_source_items_by_type[label] = len(
            {candidate.source_key for candidate in matching_candidates}
        )
        requested_quota = int(entry["weight"]) * unit_quota
        requested_quota_by_type[label] = requested_quota
        if not matching_candidates:
            raise ValueError(
                "Requested include_sample_types are missing from the sampleable pool: "
                f"{label}"
            )
        if available_source_items_by_type[label] < requested_quota:
            raise ValueError(
                "Requested include_sample_types quota exceeds sampleable support: "
                f"{label}; requested={requested_quota}; "
                f"available={available_source_items_by_type[label]}"
            )
        for candidate in matching_candidates:
            source_to_labels[candidate.source_key].add(label)
        targets.append(
            SamplingTarget(
                name=f"included_type::{label}",
                where=dict(entry["where"]),
                n_samples=requested_quota,
            )
        )

    overlapping_sources = {
        source_key: sorted(labels)
        for source_key, labels in source_to_labels.items()
        if len(labels) > 1
    }
    if overlapping_sources:
        overlap_examples = list(overlapping_sources.items())[:5]
        raise ValueError(
            "include_sample_types selectors must resolve to non-overlapping exact sample "
            f"types. Overlap examples: {overlap_examples}"
        )

    override_fields = [
        field_name
        for field_name in allocation.equal_allocation_by
        if any(field_name in entry["where"] for entry in aggregated_selectors.values())
    ]
    return {
        "enabled": True,
        "targets": targets,
        "requested_total_weight": total_weight,
        "quota_unit": unit_quota,
        "requested_type_counts": dict(
            sorted(
                (str(entry["label"]), int(entry["weight"]))
                for entry in aggregated_selectors.values()
            )
        ),
        "requested_quota_by_type": dict(sorted(requested_quota_by_type.items())),
        "available_source_items_by_type": dict(
            sorted(available_source_items_by_type.items())
        ),
        "override_equal_allocation": bool(override_fields),
        "override_fields": override_fields,
    }


def _attach_include_sample_type_metadata(
    selected: Sequence[SampledAnnotationItem],
    include_quota_spec: Dict[str, Any],
) -> List[SampledAnnotationItem]:
    """
    Attach weighted include-type quota metadata to sampled items.

    Args:
        selected (Sequence[SampledAnnotationItem]): Sampled items.
        include_quota_spec (Dict[str, Any]): Weighted include-type audit payload.

    Returns:
        List[SampledAnnotationItem]: Enriched sampled items.
    """
    if not include_quota_spec.get("enabled"):
        return list(selected)
    requested_quota_by_type = dict(include_quota_spec.get("requested_quota_by_type", {}))
    requested_type_counts = dict(include_quota_spec.get("requested_type_counts", {}))
    enriched: List[SampledAnnotationItem] = []
    for item in selected:
        target_label = str(item.selection_target)
        if target_label.startswith("included_type::"):
            target_label = target_label.split("included_type::", 1)[1]
        enriched.append(
            replace(
                item,
                selection_metadata={
                    **item.selection_metadata,
                    "included_type_quota_enabled": True,
                    "included_type_label": target_label,
                    "included_type_selector_weight": requested_type_counts.get(
                        target_label, 0
                    ),
                    "included_type_requested_quota": requested_quota_by_type.get(
                        target_label, 0
                    ),
                    "included_type_override_equal_allocation": bool(
                        include_quota_spec.get("override_equal_allocation", False)
                    ),
                    "included_type_override_fields": ",".join(
                        include_quota_spec.get("override_fields", [])
                    ),
                    "included_type_audit": include_quota_spec,
                },
            )
        )
    return enriched


def _matches_target(candidate: PairwiseCandidateRecord, target: SamplingTarget) -> bool:
    """
    Check whether a candidate matches an explicit sampling target.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        target (SamplingTarget): Explicit sampling target.

    Returns:
        bool: True when all `where` clauses match.
    """
    return matches_field_map(candidate, target.where)


def _sample_explicit_targets(
    candidates: Sequence[PairwiseCandidateRecord],
    allocation: AllocationConfig,
    *,
    preselected_candidates: Sequence[PairwiseCandidateRecord] = (),
) -> List[SampledAnnotationItem]:
    """
    Sample candidates according to explicit target definitions.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible candidates.
        allocation (AllocationConfig): Allocation configuration.

    Returns:
        List[SampledAnnotationItem]: Selected items.
    """
    selected: List[SampledAnnotationItem] = []
    pinned_selected = _select_fixed_candidates(preselected_candidates, "repeat_pinned")
    pinned_by_token = {_stable_item_token(item.candidate) for item in pinned_selected}
    matched_pinned_tokens: set[str] = set()
    used_source_tokens: set[str] = set()
    for target_index, target in enumerate(allocation.targets):
        matching_pinned = [
            item
            for item in pinned_selected
            if _matches_target(item.candidate, target)
            and _stable_item_token(item.candidate) not in matched_pinned_tokens
        ]
        if len(matching_pinned) > target.n_samples:
            raise ValueError(
                f"Sampling target '{target.name}' pins {len(matching_pinned)} repeated items, "
                f"which exceeds the requested quota of {target.n_samples}."
            )
        matching = [
            candidate
            for candidate in candidates
            if _matches_target(candidate, target)
            and _stable_item_token(candidate) not in used_source_tokens
            and _stable_item_token(candidate) not in pinned_by_token
        ]
        target_seed = allocation.random_seed + target_index
        target_selected = _select_n(
            matching,
            target.n_samples - len(matching_pinned),
            target_seed,
            target.name,
        )
        matched_pinned_tokens.update(
            _stable_item_token(item.candidate) for item in matching_pinned
        )
        selected.extend(
            [
                replace(
                    item,
                    selection_target=target.name,
                    selection_metadata={
                        **item.selection_metadata,
                        "selection_target": target.name,
                    },
                )
                for item in matching_pinned
            ]
        )
        selected.extend(target_selected)
        used_source_tokens.update(
            _stable_item_token(item.candidate) for item in target_selected
        )
    unmatched_pinned = [
        item
        for item in pinned_selected
        if _stable_item_token(item.candidate) not in matched_pinned_tokens
    ]
    if unmatched_pinned:
        raise ValueError(
            "Pinned repeated items did not match any explicit sampling target: "
            f"{[item.candidate.source_key for item in unmatched_pinned]}"
        )
    return selected


def _sample_equal_allocation(
    candidates: Sequence[PairwiseCandidateRecord],
    allocation: AllocationConfig,
    *,
    preselected_candidates: Sequence[PairwiseCandidateRecord] = (),
) -> List[SampledAnnotationItem]:
    """
    Sample candidates with strict equal allocation across requested grouping fields.

    Args:
        candidates (Sequence[PairwiseCandidateRecord]): Eligible candidates.
        allocation (AllocationConfig): Allocation configuration.

    Returns:
        List[SampledAnnotationItem]: Selected items.

    Raises:
        ValueError: If total_samples is missing or quotas cannot be satisfied.
    """
    spec = _build_equal_allocation_spec(candidates, allocation)
    fields = list(spec["fields"])
    selection_target = "joint_equal::" + "::".join(fields)
    selected: List[SampledAnnotationItem] = []
    pinned_by_stratum: Dict[Tuple[str, ...], List[PairwiseCandidateRecord]] = defaultdict(list)
    for candidate in preselected_candidates:
        pinned_by_stratum[_group_key(candidate, fields)].append(candidate)
    for stratum, pinned_candidates in pinned_by_stratum.items():
        if stratum not in spec["strata_to_candidates"]:
            raise ValueError(
                "Pinned repeated item falls outside the required equal-allocation strata: "
                f"{stratum}"
            )
        if len(pinned_candidates) > spec["per_stratum_quota"]:
            raise ValueError(
                "Pinned repeated items exceed the equal-allocation quota for stratum "
                f"{stratum}: pinned={len(pinned_candidates)} quota={spec['per_stratum_quota']}"
            )
    for stratum_index, stratum in enumerate(spec["ordered_strata"]):
        stratum_label = "::".join(
            f"{field_name}={value}" for field_name, value in zip(fields, stratum)
        )
        pinned_candidates = list(pinned_by_stratum.get(stratum, []))
        if pinned_candidates:
            selected.extend(
                [
                    replace(
                        item,
                        selection_target=f"{selection_target}::{stratum_label}",
                        selection_metadata={
                            **item.selection_metadata,
                            "selection_target": f"{selection_target}::{stratum_label}",
                        },
                    )
                    for item in _select_fixed_candidates(
                        pinned_candidates,
                        f"{selection_target}::{stratum_label}",
                    )
                ]
            )
        remaining_candidates = [
            candidate
            for candidate in spec["strata_to_candidates"][stratum]
            if candidate.source_key not in {item.source_key for item in pinned_candidates}
        ]
        stratum_selected = _select_n(
            remaining_candidates,
            n_samples=spec["per_stratum_quota"] - len(pinned_candidates),
            random_seed=allocation.random_seed + stratum_index,
            selection_target=f"{selection_target}::{stratum_label}",
        )
        selected.extend(stratum_selected)
    selected = [
        replace(item, selection_rank=selection_rank)
        for selection_rank, item in enumerate(selected, start=1)
    ]
    balance_audit = {
        "enabled": True,
        "mode": "strict_joint_strata",
        "fields": fields,
        "requested_total_samples": int(spec["total_samples"]),
        "eligible_pool_size": len(candidates),
        "eligible_counts_by_field": spec["eligible_counts_by_field"],
        "field_values": spec["field_values"],
        "required_strata": [
            {field_name: value for field_name, value in zip(fields, stratum)}
            for stratum in spec["ordered_strata"]
        ],
        "stratum_support": spec["stratum_support"],
        "per_stratum_quota": spec["per_stratum_quota"],
        "realized_counts_by_field": {
            field_name: _eligible_count_by_field(
                [item.candidate for item in selected], field_name
            )
            for field_name in fields
        },
        "realized_stratum_counts": dict(
            Counter(
                "::".join(
                    f"{field_name}={_field_value(item.candidate, field_name)}"
                    for field_name in fields
                )
                for item in selected
            )
        ),
    }
    return _attach_marginal_balance_metadata(selected, balance_audit=balance_audit)


def sample_candidates(
    candidates: List[PairwiseCandidateRecord],
    allocation: AllocationConfig,
    *,
    include_sample_types: Optional[Sequence[Any]] = None,
    pinned_source_keys: Optional[set[str]] = None,
    repeat_audit: Optional[Dict[str, Any]] = None,
    sample_type_fields: Optional[Sequence[str]] = None,
) -> List[SampledAnnotationItem]:
    """
    Sample candidates under configurable allocation rules.

    Args:
        candidates (List[PairwiseCandidateRecord]): Eligible candidates.
        allocation (AllocationConfig): Allocation configuration.

    Returns:
        List[SampledAnnotationItem]: Selected items.

    Raises:
        ValueError: If the allocation cannot be satisfied.
    """
    pinned_source_keys = set(pinned_source_keys or set())
    include_sample_types = list(include_sample_types or [])
    repeat_audit = dict(repeat_audit or {})
    sample_type_fields = list(sample_type_fields or [])
    deduped_candidates = deduplicate_candidates(candidates)
    eligible_pool = apply_stride(deduped_candidates, allocation)
    pinned_candidates = [
        candidate
        for candidate in eligible_pool
        if candidate.source_key in pinned_source_keys
    ]
    if allocation.total_samples is not None and len(pinned_candidates) > allocation.total_samples:
        raise ValueError(
            "Pinned repeated items exceed allocation.total_samples: "
            f"pinned={len(pinned_candidates)} total_samples={allocation.total_samples}"
        )
    include_quota_spec = _build_include_sample_type_quota_spec(
        eligible_pool,
        include_sample_types,
        allocation,
    )
    if include_quota_spec.get("enabled") and allocation.targets:
        raise ValueError(
            "filters.include_sample_types cannot be combined with allocation.targets."
        )
    if include_quota_spec.get("enabled"):
        if include_quota_spec.get("override_equal_allocation") and allocation.equal_allocation_by:
            logger.info(
                "Weighted include_sample_types override equal_allocation_by for fields=%s",
                include_quota_spec.get("override_fields", []),
            )
        selected = _sample_explicit_targets(
            eligible_pool,
            replace(allocation, targets=list(include_quota_spec["targets"])),
            preselected_candidates=pinned_candidates,
        )
    elif allocation.targets:
        selected = _sample_explicit_targets(
            eligible_pool,
            allocation,
            preselected_candidates=pinned_candidates,
        )
    elif allocation.equal_allocation_by:
        selected = _sample_equal_allocation(
            eligible_pool,
            allocation,
            preselected_candidates=pinned_candidates,
        )
    else:
        selected = _select_n(
            [
                candidate
                for candidate in eligible_pool
                if candidate.source_key not in pinned_source_keys
            ],
            n_samples=int(allocation.total_samples or 0) - len(pinned_candidates),
            random_seed=allocation.random_seed,
            selection_target="global",
        )
        if pinned_candidates:
            selected = _select_fixed_candidates(pinned_candidates, "repeat_pinned") + selected
            selected = [
                replace(item, selection_rank=selection_rank)
                for selection_rank, item in enumerate(selected, start=1)
            ]
    selected = _attach_selection_metadata(selected, candidates)
    if include_quota_spec.get("enabled"):
        selected = _attach_include_sample_type_metadata(selected, include_quota_spec)
    selected = _attach_repeat_metadata(
        selected,
        pinned_source_keys=pinned_source_keys,
        repeat_audit=repeat_audit,
    )
    selected = _attach_sample_type_metadata(selected, sample_type_fields)
    item_ids = [item.item_id for item in selected]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Sampling produced duplicate human-annotation item ids.")
    logger.info(
        "Sampling summary: input_rows=%d deduped_items=%d stride_items=%d sampled_items=%d",
        len(candidates),
        len(deduped_candidates),
        len(eligible_pool),
        len(selected),
    )
    logger.info(
        "Sampled composition: personas=%s prompt_types=%s model_pairs=%s",
        dict(Counter(item.candidate.persona for item in selected)),
        dict(Counter(item.candidate.prompt_type for item in selected)),
        dict(Counter(item.candidate.model_pair for item in selected)),
    )
    return selected


def sample_candidates_for_annotators(
    candidates: List[PairwiseCandidateRecord],
    allocation: AllocationConfig,
    annotator_config: AnnotatorConfig,
    *,
    include_sample_types: Optional[Sequence[SampleTypeSpec]] = None,
    pinned_source_keys: Optional[set[str]] = None,
    repeat_audit: Optional[Dict[str, Any]] = None,
    sample_type_fields: Optional[Sequence[str]] = None,
) -> List[SampledAnnotationItem]:
    """
    Sample items for a study that may include fixed calibration anchors.

    Args:
        candidates (List[PairwiseCandidateRecord]): Eligible candidates.
        allocation (AllocationConfig): Base allocation configuration.
        annotator_config (AnnotatorConfig): Annotator assignment configuration.
        include_sample_types (Optional[Sequence[SampleTypeSpec]]): Weighted packet template.
        pinned_source_keys (Optional[set[str]]): Source keys forced into sampling.
        repeat_audit (Optional[Dict[str, Any]]): Repeat audit metadata.
        sample_type_fields (Optional[Sequence[str]]): Fields used for packet balancing.

    Returns:
        List[SampledAnnotationItem]: Combined sampled items.

    Raises:
        ValueError: If the calibration/regular split is infeasible.
    """
    calibration_sample_types = list(annotator_config.calibration_sample_types)
    if not calibration_sample_types:
        return sample_candidates(
            candidates,
            allocation,
            include_sample_types=include_sample_types,
            pinned_source_keys=pinned_source_keys,
            repeat_audit=repeat_audit,
            sample_type_fields=sample_type_fields,
        )
    if annotator_config.items_per_annotator is None:
        raise ValueError(
            "annotators.items_per_annotator is required when calibration_sample_types are configured."
        )
    if annotator_config.anchor_count != len(calibration_sample_types):
        raise ValueError(
            "annotators.anchor_count must match the number of calibration_sample_types. "
            f"anchor_count={annotator_config.anchor_count} "
            f"calibration_sample_types={len(calibration_sample_types)}"
        )
    annotator_count = len(annotator_config.annotator_ids)
    regular_slots_per_annotator = (
        annotator_config.items_per_annotator
        - annotator_config.anchor_count
        - annotator_config.overlap_count
    )
    if regular_slots_per_annotator < 0:
        raise ValueError(
            "annotators.items_per_annotator must be >= anchor_count + overlap_count."
        )
    total_regular_assignments = regular_slots_per_annotator * annotator_count
    if total_regular_assignments % annotator_config.annotators_per_regular_item != 0:
        raise ValueError(
            "Regular assignment demand must be divisible by annotators_per_regular_item. "
            f"regular_assignments={total_regular_assignments} "
            f"annotators_per_regular_item={annotator_config.annotators_per_regular_item}"
        )
    calibration_items = _select_calibration_candidates(
        candidates,
        calibration_sample_types,
        random_seed=allocation.random_seed,
    )
    calibration_items = _attach_selection_metadata(calibration_items, candidates)
    calibration_items = _attach_repeat_metadata(
        calibration_items,
        pinned_source_keys=set(pinned_source_keys or set()),
        repeat_audit=dict(repeat_audit or {}),
    )
    calibration_items = _attach_sample_type_metadata(
        calibration_items,
        list(sample_type_fields or []),
    )
    calibration_items = _attach_item_role_metadata(
        calibration_items,
        item_role="calibration",
        extra_metadata={
            "calibration_selector_count": len(calibration_sample_types),
            "annotators_per_regular_item": annotator_config.annotators_per_regular_item,
        },
    )
    calibration_source_keys = {
        item.candidate.source_key for item in calibration_items
    }
    regular_candidates = [
        candidate
        for candidate in candidates
        if candidate.source_key not in calibration_source_keys
    ]
    regular_include_sample_types = _subtract_selector_instances(
        list(include_sample_types or []),
        calibration_sample_types,
    )
    regular_total_samples = (
        total_regular_assignments // annotator_config.annotators_per_regular_item
    )
    regular_selected = sample_candidates(
        regular_candidates,
        replace(allocation, total_samples=regular_total_samples),
        include_sample_types=regular_include_sample_types,
        pinned_source_keys=set(pinned_source_keys or set()) - calibration_source_keys,
        repeat_audit=repeat_audit,
        sample_type_fields=sample_type_fields,
    )
    regular_selected = _attach_item_role_metadata(
        regular_selected,
        item_role="regular",
        extra_metadata={
            "annotators_per_regular_item": annotator_config.annotators_per_regular_item,
            "regular_slots_per_annotator": regular_slots_per_annotator,
            "regular_total_assignments": total_regular_assignments,
        },
    )
    combined = calibration_items + regular_selected
    return [
        replace(item, selection_rank=selection_rank)
        for selection_rank, item in enumerate(combined, start=1)
    ]
