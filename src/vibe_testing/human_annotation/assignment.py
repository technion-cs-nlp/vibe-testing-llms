"""Annotator assignment utilities for human-annotation studies."""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from src.vibe_testing.human_annotation.schemas import (
    AnnotatorAssignment,
    AnnotatorConfig,
    SampledAnnotationItem,
)
from src.vibe_testing.human_annotation.sample_type_utils import (
    sample_type_key,
    sample_type_label,
)
from src.vibe_testing.utils import load_json

logger = logging.getLogger(__name__)


def _load_excluded_source_keys(paths: Sequence[Path]) -> Set[str]:
    """
    Load previously-used source keys from prior manifests.

    Args:
        paths (Sequence[Path]): Manifest paths to inspect.

    Returns:
        Set[str]: Source keys that should be excluded from reuse.
    """
    excluded: Set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Excluded manifest path not found: {resolved}")
        payload = load_json(str(resolved))
        items = payload.get("sampled_items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError(
                f"Excluded manifest must contain a 'sampled_items' list: {resolved}"
            )
        for item in items:
            source_key = (
                ((item.get("candidate") or {}).get("source_key"))
                if isinstance(item, dict)
                else None
            )
            if source_key:
                excluded.add(str(source_key))
    return excluded


def exclude_previously_used_items(
    sampled_items: Sequence[SampledAnnotationItem], annotator_config: AnnotatorConfig
) -> List[SampledAnnotationItem]:
    """
    Exclude sampled items whose source keys appear in prior manifests.

    Args:
        sampled_items (Sequence[SampledAnnotationItem]): Newly sampled items.
        annotator_config (AnnotatorConfig): Annotator configuration.

    Returns:
        List[SampledAnnotationItem]: Items not present in prior manifests.

    Raises:
        ValueError: If exclusion removes all sampled items.
    """
    if not annotator_config.excluded_manifest_paths:
        return list(sampled_items)
    excluded_source_keys = _load_excluded_source_keys(
        annotator_config.excluded_manifest_paths
    )
    filtered = [
        item
        for item in sampled_items
        if item.candidate.source_key not in excluded_source_keys
    ]
    if not filtered:
        raise ValueError(
            "All sampled items were excluded by excluded_manifest_paths. "
            f"Excluded source keys loaded: {len(excluded_source_keys)}"
        )
    logger.info(
        "Assignment exclusion summary: input_items=%d kept_items=%d excluded_items=%d manifests=%d",
        len(sampled_items),
        len(filtered),
        len(sampled_items) - len(filtered),
        len(annotator_config.excluded_manifest_paths),
    )
    return filtered


def _assign_unique_randomly(
    unique_items: Sequence[SampledAnnotationItem],
    annotator_ids: Sequence[str],
    *,
    items_per_annotator: int | None,
    random_seed: int,
) -> Dict[str, List[SampledAnnotationItem]]:
    """
    Assign unique items by shuffled slot filling.

    Args:
        unique_items (Sequence[SampledAnnotationItem]): Unique items to assign.
        annotator_ids (Sequence[str]): Annotator identifiers.
        items_per_annotator (int | None): Requested packet size per annotator.
        random_seed (int): Seed used for deterministic shuffling.

    Returns:
        Dict[str, List[SampledAnnotationItem]]: Unique items by annotator.
    """
    rng = random.Random(random_seed)
    unique_per_annotator: Dict[str, List[SampledAnnotationItem]] = defaultdict(list)
    unique_assignment_pool = list(unique_items)
    if items_per_annotator is not None:
        annotator_slots = [
            annotator_id
            for annotator_id in annotator_ids
            for _ in range(items_per_annotator)
        ]
    else:
        base_quota, remainder = divmod(len(unique_assignment_pool), len(annotator_ids))
        annotator_slots = [
            annotator_id for annotator_id in annotator_ids for _ in range(base_quota)
        ]
        remainder_annotators = list(annotator_ids)
        rng.shuffle(remainder_annotators)
        annotator_slots.extend(remainder_annotators[:remainder])
    rng.shuffle(unique_assignment_pool)
    rng.shuffle(annotator_slots)
    for item, annotator_id in zip(unique_assignment_pool, annotator_slots):
        unique_per_annotator[annotator_id].append(item)
    return unique_per_annotator


def _select_shared_items_by_type(
    items: Sequence[SampledAnnotationItem],
    shared_count: int,
    *,
    balance_by: Sequence[str],
    random_seed: int,
) -> Tuple[List[SampledAnnotationItem], List[SampledAnnotationItem]]:
    """
    Select shared items with round-robin type coverage.

    Args:
        items (Sequence[SampledAnnotationItem]): Candidate sampled items.
        shared_count (int): Number of shared items to take.
        balance_by (Sequence[str]): Fields that define the assignment sample type.
        random_seed (int): Seed used for deterministic shuffling.

    Returns:
        Tuple[List[SampledAnnotationItem], List[SampledAnnotationItem]]: Shared items and
            remaining items.
    """
    if shared_count == 0:
        return [], list(items)
    rng = random.Random(random_seed)
    groups: Dict[Tuple[str, ...], List[SampledAnnotationItem]] = defaultdict(list)
    for item in items:
        groups[sample_type_key(item.candidate, balance_by)].append(item)
    type_keys = sorted(groups.keys())
    rng.shuffle(type_keys)
    for type_key in type_keys:
        rng.shuffle(groups[type_key])
    shared_items: List[SampledAnnotationItem] = []
    while len(shared_items) < shared_count:
        progressed = False
        for type_key in type_keys:
            if not groups[type_key]:
                continue
            shared_items.append(groups[type_key].pop())
            progressed = True
            if len(shared_items) == shared_count:
                break
        if not progressed:
            break
    shared_tokens = {item.item_id for item in shared_items}
    remaining_items = [item for item in items if item.item_id not in shared_tokens]
    return shared_items, remaining_items


def _assign_unique_balanced_by_type(
    unique_items: Sequence[SampledAnnotationItem],
    annotator_ids: Sequence[str],
    *,
    annotator_config: AnnotatorConfig,
    random_seed: int,
) -> Dict[str, List[SampledAnnotationItem]]:
    """
    Assign unique items so each packet is balanced by sample type.

    Args:
        unique_items (Sequence[SampledAnnotationItem]): Unique items to distribute.
        annotator_ids (Sequence[str]): Annotator identifiers.
        annotator_config (AnnotatorConfig): Annotator assignment configuration.
        random_seed (int): Seed used for deterministic shuffling.

    Returns:
        Dict[str, List[SampledAnnotationItem]]: Unique items by annotator.

    Raises:
        ValueError: If the balanced assignment cannot satisfy exact packet sizes.
    """
    if annotator_config.items_per_annotator is None:
        raise ValueError(
            "annotators.items_per_annotator is required when annotator balancing is enabled."
        )
    rng = random.Random(random_seed)
    annotator_order = list(annotator_ids)
    rng.shuffle(annotator_order)
    groups: Dict[Tuple[str, ...], List[SampledAnnotationItem]] = defaultdict(list)
    for item in unique_items:
        groups[sample_type_key(item.candidate, annotator_config.balance_by)].append(
            item
        )
    type_keys = sorted(groups.keys())
    for type_key in type_keys:
        rng.shuffle(groups[type_key])
    quotas: Dict[str, Counter[Tuple[str, ...]]] = {
        annotator_id: Counter() for annotator_id in annotator_ids
    }
    extra_cursor = 0
    for type_key in type_keys:
        count = len(groups[type_key])
        base_quota, remainder = divmod(count, len(annotator_ids))
        for annotator_id in annotator_ids:
            quotas[annotator_id][type_key] += base_quota
        for remainder_index in range(remainder):
            annotator_id = annotator_order[
                (extra_cursor + remainder_index) % len(annotator_order)
            ]
            quotas[annotator_id][type_key] += 1
        extra_cursor = (extra_cursor + remainder) % len(annotator_order)

    unique_per_annotator: Dict[str, List[SampledAnnotationItem]] = defaultdict(list)
    for type_key in type_keys:
        cursor = 0
        for annotator_id in annotator_order:
            count = int(quotas[annotator_id][type_key])
            if count == 0:
                continue
            unique_per_annotator[annotator_id].extend(
                groups[type_key][cursor : cursor + count]
            )
            cursor += count

    expected_unique_slots = (
        annotator_config.items_per_annotator
        - annotator_config.anchor_count
        - annotator_config.overlap_count
    )
    for annotator_id in annotator_ids:
        assigned_count = len(unique_per_annotator[annotator_id])
        if assigned_count != expected_unique_slots:
            raise ValueError(
                "Balanced annotator assignment could not satisfy the requested unique "
                f"packet size for '{annotator_id}': expected {expected_unique_slots}, "
                f"got {assigned_count}."
            )
    return unique_per_annotator


def _pick_annotators_for_item(
    remaining_slots: Counter[str],
    annotator_ids: Sequence[str],
    *,
    multiplicity: int,
    rng: random.Random,
) -> List[str]:
    """
    Pick unique annotators for one replicated regular item.

    Args:
        remaining_slots (Counter[str]): Remaining per-annotator quota.
        annotator_ids (Sequence[str]): Annotator identifiers.
        multiplicity (int): Number of annotators required for the item.
        rng (random.Random): Deterministic RNG.

    Returns:
        List[str]: Chosen annotators.

    Raises:
        ValueError: If the multiplicity cannot be satisfied.
    """
    chosen: List[str] = []
    while len(chosen) < multiplicity:
        available = [
            annotator_id
            for annotator_id in annotator_ids
            if remaining_slots[annotator_id] > 0 and annotator_id not in chosen
        ]
        if len(available) < multiplicity - len(chosen):
            raise ValueError(
                "Not enough annotators remain to satisfy the regular-item multiplicity."
            )
        max_remaining = max(remaining_slots[annotator_id] for annotator_id in available)
        candidates = [
            annotator_id
            for annotator_id in available
            if remaining_slots[annotator_id] == max_remaining
        ]
        rng.shuffle(candidates)
        chosen.append(candidates[0])
    return chosen


def _assign_regular_items_with_multiplicity(
    regular_items: Sequence[SampledAnnotationItem],
    annotator_ids: Sequence[str],
    *,
    annotator_config: AnnotatorConfig,
    random_seed: int,
    use_balance_by: bool,
) -> Dict[str, List[SampledAnnotationItem]]:
    """
    Assign regular items so each item is labeled by multiple annotators.

    Args:
        regular_items (Sequence[SampledAnnotationItem]): Regular sampled items.
        annotator_ids (Sequence[str]): Annotator identifiers.
        annotator_config (AnnotatorConfig): Annotator assignment configuration.
        random_seed (int): Seed used for deterministic assignment.
        use_balance_by (bool): Whether to preserve type-local quotas.

    Returns:
        Dict[str, List[SampledAnnotationItem]]: Regular items by annotator.

    Raises:
        ValueError: If the assignment demand cannot be satisfied.
    """
    if annotator_config.items_per_annotator is None:
        raise ValueError(
            "annotators.items_per_annotator is required when annotators_per_regular_item > 1."
        )
    multiplicity = int(annotator_config.annotators_per_regular_item)
    regular_slots_per_annotator = (
        annotator_config.items_per_annotator
        - annotator_config.anchor_count
        - annotator_config.overlap_count
    )
    total_required_assignments = regular_slots_per_annotator * len(annotator_ids)
    if len(regular_items) * multiplicity != total_required_assignments:
        raise ValueError(
            "Regular sampled items do not match the requested annotator packet size and multiplicity. "
            f"regular_items={len(regular_items)} multiplicity={multiplicity} "
            f"required_assignments={total_required_assignments}"
        )

    rng = random.Random(random_seed)
    if use_balance_by and annotator_config.balance_by:
        groups: Dict[Tuple[str, ...], List[SampledAnnotationItem]] = defaultdict(list)
        for item in regular_items:
            groups[sample_type_key(item.candidate, annotator_config.balance_by)].append(
                item
            )
    else:
        groups = {("__all__",): list(regular_items)}
    type_keys = sorted(groups.keys())
    rng.shuffle(type_keys)

    assignments_by_annotator: Dict[str, List[SampledAnnotationItem]] = defaultdict(list)
    for type_index, type_key in enumerate(type_keys):
        type_items = list(groups[type_key])
        rng.shuffle(type_items)
        total_type_assignments = len(type_items) * multiplicity
        if total_type_assignments % len(annotator_ids) != 0:
            raise ValueError(
                "Replicated assignment requires type-local assignment demand to divide evenly "
                f"across annotators. type_key={type_key} items={len(type_items)} "
                f"multiplicity={multiplicity}"
            )
        per_annotator_quota = total_type_assignments // len(annotator_ids)
        remaining_slots = Counter(
            {annotator_id: per_annotator_quota for annotator_id in annotator_ids}
        )
        annotator_order = list(annotator_ids)
        random.Random(random_seed + type_index).shuffle(annotator_order)
        for item in type_items:
            chosen_annotators = _pick_annotators_for_item(
                remaining_slots,
                annotator_order,
                multiplicity=multiplicity,
                rng=rng,
            )
            for annotator_id in chosen_annotators:
                assignments_by_annotator[annotator_id].append(item)
                remaining_slots[annotator_id] -= 1
        if any(remaining_slots.values()):
            raise ValueError(
                "Replicated assignment finished with unfilled annotator quotas: "
                f"{dict(remaining_slots)}"
            )

    for annotator_id in annotator_ids:
        if len(assignments_by_annotator[annotator_id]) != regular_slots_per_annotator:
            raise ValueError(
                "Replicated assignment could not satisfy the requested regular packet size "
                f"for '{annotator_id}': expected {regular_slots_per_annotator}, "
                f"got {len(assignments_by_annotator[annotator_id])}."
            )
    return assignments_by_annotator


def assign_items_to_annotators(
    sampled_items: Sequence[SampledAnnotationItem],
    annotator_config: AnnotatorConfig,
    random_seed: int,
) -> List[AnnotatorAssignment]:
    """
    Assign sampled items to annotators with optional shared anchor/overlap pools.

    Args:
        sampled_items (Sequence[SampledAnnotationItem]): Sampled study items.
        annotator_config (AnnotatorConfig): Annotator assignment configuration.
        random_seed (int): Seed used for deterministic assignment.

    Returns:
        List[AnnotatorAssignment]: Flat list of annotator assignments.

    Raises:
        ValueError: If counts cannot satisfy the requested assignment policy.
    """
    items = exclude_previously_used_items(sampled_items, annotator_config)
    rng = random.Random(random_seed)
    ordered = list(items)
    rng.shuffle(ordered)
    annotator_ids = list(annotator_config.annotator_ids)

    if annotator_config.annotators_per_regular_item > 1:
        if annotator_config.overlap_count:
            raise ValueError(
                "annotators.overlap_count is not supported when annotators_per_regular_item > 1."
            )
        calibration_items = [
            item
            for item in ordered
            if item.selection_metadata.get("item_role") == "calibration"
        ]
        if len(calibration_items) != annotator_config.anchor_count:
            raise ValueError(
                "Calibration sampling must produce exactly annotators.anchor_count items. "
                f"anchor_count={annotator_config.anchor_count} "
                f"calibration_items={len(calibration_items)}"
            )
        regular_items = [
            item
            for item in ordered
            if item.selection_metadata.get("item_role") != "calibration"
        ]
        try:
            regular_per_annotator = _assign_regular_items_with_multiplicity(
                regular_items,
                annotator_ids,
                annotator_config=annotator_config,
                random_seed=random_seed,
                use_balance_by=bool(annotator_config.balance_by),
            )
        except ValueError:
            if annotator_config.balance_mode == "strict" or not annotator_config.balance_by:
                raise
            logger.warning(
                "Balanced replicated assignment was infeasible; falling back to best-effort replicated assignment.",
                exc_info=True,
            )
            regular_per_annotator = _assign_regular_items_with_multiplicity(
                regular_items,
                annotator_ids,
                annotator_config=annotator_config,
                random_seed=random_seed,
                use_balance_by=False,
            )

        assignments: List[AnnotatorAssignment] = []
        for annotator_id in annotator_ids:
            assigned_items: List[tuple[SampledAnnotationItem, str]] = []
            assigned_items.extend((item, "anchor") for item in calibration_items)
            assigned_items.extend(
                (item, "regular") for item in regular_per_annotator[annotator_id]
            )
            if annotator_config.items_per_annotator is not None and len(
                assigned_items
            ) != int(annotator_config.items_per_annotator):
                raise ValueError(
                    f"Annotator '{annotator_id}' received {len(assigned_items)} items, expected "
                    f"{annotator_config.items_per_annotator}."
                )
            rng.shuffle(assigned_items)
            for assignment_rank, (item, role) in enumerate(assigned_items, start=1):
                assignments.append(
                    AnnotatorAssignment(
                        annotator_id=annotator_id,
                        item=item,
                        assignment_rank=assignment_rank,
                        presentation_swapped=bool(rng.randint(0, 1)),
                        assignment_role=role,
                    )
                )

        per_annotator_counts = Counter(
            assignment.annotator_id for assignment in assignments
        )
        logger.info(
            "Replicated assignment summary: source_items=%d kept_after_exclusion=%d anchors=%d regular_items=%d multiplicity=%d assignments=%d annotators=%d balance_by=%s balance_scope=%s balance_mode=%s",
            len(sampled_items),
            len(items),
            len(calibration_items),
            len(regular_items),
            annotator_config.annotators_per_regular_item,
            len(assignments),
            len(annotator_ids),
            list(annotator_config.balance_by),
            annotator_config.balance_scope,
            annotator_config.balance_mode,
        )
        logger.info("Assignments per annotator: %s", dict(per_annotator_counts))
        if annotator_config.balance_by:
            per_annotator_type_counts: Dict[str, Dict[str, int]] = {}
            for annotator_id in annotator_ids:
                type_counter = Counter(
                    sample_type_label(
                        assignment.item.candidate, annotator_config.balance_by
                    )
                    for assignment in assignments
                    if assignment.annotator_id == annotator_id
                )
                per_annotator_type_counts[annotator_id] = dict(type_counter)
            logger.info(
                "Assignments by annotator sample type: %s",
                per_annotator_type_counts,
            )
        return assignments

    shared_count = annotator_config.anchor_count + annotator_config.overlap_count
    if shared_count > len(ordered):
        raise ValueError(
            "annotators.anchor_count + annotators.overlap_count exceeds sampled items."
        )

    shared_items = ordered[:shared_count]
    unique_items = ordered[shared_count:]
    anchor_items = shared_items[: annotator_config.anchor_count]
    overlap_items = shared_items[annotator_config.anchor_count :]
    if annotator_config.items_per_annotator is not None:
        unique_needed_total = (
            annotator_config.items_per_annotator
            - annotator_config.anchor_count
            - annotator_config.overlap_count
        ) * len(annotator_ids)
        if unique_needed_total < 0:
            raise ValueError(
                "annotators.items_per_annotator must be >= anchor_count + overlap_count."
            )
        if unique_needed_total > len(unique_items):
            raise ValueError(
                "Not enough unique sampled items for requested items_per_annotator."
            )
    else:
        unique_needed_total = len(unique_items)

    unique_assignment_pool = list(unique_items[:unique_needed_total])
    if annotator_config.balance_by and annotator_config.balance_scope == "all_items":
        shared_items, unique_assignment_pool = _select_shared_items_by_type(
            ordered,
            shared_count,
            balance_by=annotator_config.balance_by,
            random_seed=random_seed,
        )
        anchor_items = shared_items[: annotator_config.anchor_count]
        overlap_items = shared_items[annotator_config.anchor_count :]
        if unique_needed_total > len(unique_assignment_pool):
            raise ValueError(
                "Not enough unique sampled items remain after balanced shared-item "
                "selection for the requested items_per_annotator."
            )
        unique_assignment_pool = list(unique_assignment_pool[:unique_needed_total])

    if annotator_config.balance_by:
        try:
            unique_per_annotator = _assign_unique_balanced_by_type(
                unique_assignment_pool,
                annotator_ids,
                annotator_config=annotator_config,
                random_seed=random_seed,
            )
        except ValueError:
            if annotator_config.balance_mode == "strict":
                raise
            logger.warning(
                "Balanced annotator assignment was infeasible; falling back to best-effort shuffled assignment.",
                exc_info=True,
            )
            unique_per_annotator = _assign_unique_randomly(
                unique_assignment_pool,
                annotator_ids,
                items_per_annotator=(
                    annotator_config.items_per_annotator
                    - annotator_config.anchor_count
                    - annotator_config.overlap_count
                    if annotator_config.items_per_annotator is not None
                    else None
                ),
                random_seed=random_seed,
            )
    else:
        unique_per_annotator = _assign_unique_randomly(
            unique_assignment_pool,
            annotator_ids,
            items_per_annotator=(
                annotator_config.items_per_annotator
                - annotator_config.anchor_count
                - annotator_config.overlap_count
                if annotator_config.items_per_annotator is not None
                else None
            ),
            random_seed=random_seed,
        )

    assignments: List[AnnotatorAssignment] = []
    for annotator_id in annotator_ids:
        assigned_items: List[tuple[SampledAnnotationItem, str]] = []
        assigned_items.extend((item, "anchor") for item in anchor_items)
        assigned_items.extend((item, "overlap") for item in overlap_items)
        assigned_items.extend(
            (item, "unique") for item in unique_per_annotator[annotator_id]
        )
        if annotator_config.items_per_annotator is not None and len(
            assigned_items
        ) != int(annotator_config.items_per_annotator):
            raise ValueError(
                f"Annotator '{annotator_id}' received {len(assigned_items)} items, expected "
                f"{annotator_config.items_per_annotator}."
            )
        rng.shuffle(assigned_items)
        for assignment_rank, (item, role) in enumerate(assigned_items, start=1):
            assignments.append(
                AnnotatorAssignment(
                    annotator_id=annotator_id,
                    item=item,
                    assignment_rank=assignment_rank,
                    presentation_swapped=bool(rng.randint(0, 1)),
                    assignment_role=role,
                )
            )

    per_annotator_counts = Counter(
        assignment.annotator_id for assignment in assignments
    )
    logger.info(
        "Assignment summary: source_items=%d kept_after_exclusion=%d anchors=%d overlaps=%d unique_pool=%d assignments=%d annotators=%d balance_by=%s balance_scope=%s balance_mode=%s",
        len(sampled_items),
        len(items),
        len(anchor_items),
        len(overlap_items),
        len(unique_assignment_pool),
        len(assignments),
        len(annotator_ids),
        list(annotator_config.balance_by),
        annotator_config.balance_scope,
        annotator_config.balance_mode,
    )
    logger.info("Assignments per annotator: %s", dict(per_annotator_counts))
    if annotator_config.balance_by:
        per_annotator_type_counts: Dict[str, Dict[str, int]] = {}
        for annotator_id in annotator_ids:
            type_counter = Counter(
                sample_type_label(
                    assignment.item.candidate, annotator_config.balance_by
                )
                for assignment in assignments
                if assignment.annotator_id == annotator_id
            )
            per_annotator_type_counts[annotator_id] = dict(type_counter)
        logger.info(
            "Assignments by annotator sample type: %s",
            per_annotator_type_counts,
        )
    return assignments
