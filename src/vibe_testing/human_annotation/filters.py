"""Filtering helpers for pairwise human-annotation candidate selection."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from src.vibe_testing.human_annotation.schemas import (
    FilterConfig,
    MaxOutputCharsOverride,
    PairwiseCandidateRecord,
)

logger = logging.getLogger(__name__)

AgreementSignature = Tuple[str, Tuple[Tuple[str, str], ...]] | str


@dataclass(frozen=True)
class GroupFilterOutcome:
    """
    Audit summary for one cross-judge source group.

    Attributes:
        source_key: Stable cross-judge item identifier.
        persona: Persona for the group.
        prompt_type: Prompt type for the group.
        model_pair: Canonical model pair for the group.
        pairwise_judgment_type: Pairwise judgment type for the group.
        total_rows: Rows discovered for the group before filtering.
        individually_kept_rows: Rows remaining after per-row filters.
        rows_rejected_before_consensus: Rows removed before consensus logic.
        rows_excluded_tied_before_consensus: Rows removed because tied overall
            winners were excluded from the consensus pool.
        agreement_pool_size: Rows considered for consensus after exclusions.
        required_consensus_size: Minimum bucket size needed to pass consensus.
        consensus_bucket_size: Size of the winning consensus bucket.
        signature_counts: Signature bucket sizes for the agreement pool.
        selected_signature: Winning signature when one exists.
        outcome: Human-readable group outcome label.
    """

    source_key: str
    persona: str
    prompt_type: str
    model_pair: str
    pairwise_judgment_type: str
    total_rows: int
    individually_kept_rows: int
    rows_rejected_before_consensus: int
    rows_excluded_tied_before_consensus: int
    agreement_pool_size: int
    required_consensus_size: Optional[int]
    consensus_bucket_size: int
    signature_counts: Dict[str, int]
    selected_signature: Optional[str]
    outcome: str


@dataclass(frozen=True)
class FilterResult:
    """
    Result of applying candidate filters.

    Attributes:
        kept: Candidates that passed all filters.
        rejected: Candidates that were rejected.
        rejection_counts: Rejection counts keyed by reason.
        agreement_stats: Aggregate agreement counters for summary reporting.
        group_outcomes: Per-group audit payloads.
    """

    kept: List[PairwiseCandidateRecord]
    rejected: List[PairwiseCandidateRecord]
    rejection_counts: Dict[str, int]
    agreement_stats: Dict[str, object]
    group_outcomes: List[GroupFilterOutcome]


def _candidate_field_value(candidate: PairwiseCandidateRecord, field_name: str) -> str:
    """
    Read a comparable string field from a candidate record.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        field_name (str): Candidate attribute name.

    Returns:
        str: Comparable string value.

    Raises:
        ValueError: If the requested field is not present.
    """
    if not hasattr(candidate, field_name):
        raise ValueError(f"Unknown filter field requested: {field_name}")
    value = getattr(candidate, field_name)
    return str(value)


def _check_field_filters(
    candidate: PairwiseCandidateRecord, config: FilterConfig
) -> Tuple[bool, str]:
    """
    Apply allow/deny field filters to a candidate.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        config (FilterConfig): Filtering configuration.

    Returns:
        Tuple[bool, str]: Pass flag and failure reason.
    """
    for field_name, allowed_values in config.allow.items():
        candidate_value = _candidate_field_value(candidate, field_name)
        if candidate_value not in allowed_values:
            return False, f"allow:{field_name}"
    for field_name, denied_values in config.deny.items():
        candidate_value = _candidate_field_value(candidate, field_name)
        if candidate_value in denied_values:
            return False, f"deny:{field_name}"
    return True, ""


def _override_matches_candidate(
    candidate: PairwiseCandidateRecord, override: MaxOutputCharsOverride
) -> bool:
    """
    Check whether a max-output-chars override matches a candidate.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        override (MaxOutputCharsOverride): Override rule to evaluate.

    Returns:
        bool: True when all override fields match.
    """
    for field_name, allowed_values in override.where.items():
        candidate_value = _candidate_field_value(candidate, field_name)
        if candidate_value not in allowed_values:
            return False
    return True


def resolve_max_output_chars(
    candidate: PairwiseCandidateRecord, config: FilterConfig
) -> Optional[int]:
    """
    Resolve the effective output-length threshold for one candidate.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        config (FilterConfig): Filtering configuration.

    Returns:
        Optional[int]: Effective max length, or `None` when the filter is disabled.
    """
    resolved = config.max_output_chars
    for override in config.max_output_chars_overrides:
        if _override_matches_candidate(candidate, override):
            resolved = override.max_output_chars
    return resolved


def _group_by_source_key(
    candidates: Iterable[PairwiseCandidateRecord],
) -> Dict[str, List[PairwiseCandidateRecord]]:
    """
    Group candidate records by cross-judge identity.

    Args:
        candidates (Iterable[PairwiseCandidateRecord]): Candidate records.

    Returns:
        Dict[str, List[PairwiseCandidateRecord]]: Grouped candidates.
    """
    grouped: Dict[str, List[PairwiseCandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_key].append(candidate)
    return grouped


def _winner_signature(
    candidate: PairwiseCandidateRecord,
) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    """
    Build an agreement signature for a candidate.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.

    Returns:
        Tuple[str, Tuple[Tuple[str, str], ...]]: Overall winner plus per-dimension winners.
    """
    dims = tuple(
        sorted(
            (dim_name, str((payload or {}).get("winner", "tie")))
            for dim_name, payload in candidate.dimension_results.items()
        )
    )
    return str(candidate.overall_winner), dims


def _overall_winner_signature(candidate: PairwiseCandidateRecord) -> str:
    """
    Build an overall-winner-only agreement signature.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.

    Returns:
        str: Normalized overall-winner signature.
    """
    return str(candidate.overall_winner)


def _is_tied_overall(candidate: PairwiseCandidateRecord) -> bool:
    """
    Check whether a candidate has a tied or missing overall winner.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.

    Returns:
        bool: True when the candidate is tied overall.
    """
    return str(candidate.overall_winner).lower() in {"", "none", "tie"}


def _individual_rejection_reason(
    candidate: PairwiseCandidateRecord, config: FilterConfig
) -> Optional[str]:
    """
    Apply non-consensus filters to a candidate and return the first failure reason.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        config (FilterConfig): Filtering configuration.

    Returns:
        Optional[str]: Failure reason, or None when the candidate passes.
    """
    passed, reason = _check_field_filters(candidate, config)
    if not passed:
        return reason
    max_output_chars = resolve_max_output_chars(candidate, config)
    if max_output_chars is not None and (
        candidate.output_char_len_a > max_output_chars
        or candidate.output_char_len_b > max_output_chars
    ):
        return "max_output_chars"
    if config.require_no_overall_ties and _is_tied_overall(candidate):
        return "overall_tie"
    if config.require_no_dimension_ties and any(
        str((payload or {}).get("winner", "tie")).lower() == "tie"
        for payload in candidate.dimension_results.values()
    ):
        return "dimension_tie"
    return None


def _agreement_signature(
    candidate: PairwiseCandidateRecord, config: FilterConfig
) -> AgreementSignature:
    """
    Build the agreement signature for a candidate based on filter settings.

    Args:
        candidate (PairwiseCandidateRecord): Candidate record.
        config (FilterConfig): Filtering configuration.

    Returns:
        AgreementSignature: Comparable agreement signature.
    """
    if config.consensus_scope == "overall_winner":
        return _overall_winner_signature(candidate)
    return _winner_signature(candidate)


def _serialize_signature(signature: AgreementSignature) -> str:
    """
    Convert an agreement signature into a stable summary key.

    Args:
        signature (AgreementSignature): Comparable signature object.

    Returns:
        str: JSON-friendly signature string.
    """
    if isinstance(signature, str):
        return signature
    overall_winner, dims = signature
    dims_text = ",".join(f"{name}={winner}" for name, winner in dims)
    return f"{overall_winner}|{dims_text}"


def _build_agreement_pool(
    group_items: List[PairwiseCandidateRecord],
    config: FilterConfig,
    rejected: List[PairwiseCandidateRecord],
    rejection_counts: Counter[str],
    agreement_stats: Counter[str],
) -> tuple[List[PairwiseCandidateRecord], int]:
    """
    Build the consensus pool after optional tied-overall exclusion.

    Args:
        group_items (List[PairwiseCandidateRecord]): Rows surviving per-row filters.
        config (FilterConfig): Filtering configuration.
        rejected (List[PairwiseCandidateRecord]): Rejected rows accumulator.
        rejection_counts (Counter[str]): Rejection counter accumulator.
        agreement_stats (Counter[str]): Agreement counter accumulator.

    Returns:
        tuple[List[PairwiseCandidateRecord], int]: Agreement pool and number of
            rows removed before consensus due to tied-overall exclusion.
    """
    agreement_pool: List[PairwiseCandidateRecord] = []
    rows_excluded_tied = 0
    for candidate in group_items:
        if config.exclude_tied_overall_from_consensus and _is_tied_overall(candidate):
            rejected.append(candidate)
            rejection_counts["judge_consensus_excluded_tied_overall"] += 1
            agreement_stats["rows_excluded_tied_before_consensus"] += 1
            rows_excluded_tied += 1
            continue
        agreement_pool.append(candidate)
    return agreement_pool, rows_excluded_tied


def _build_signature_buckets(
    group_items: List[PairwiseCandidateRecord], config: FilterConfig
) -> Dict[AgreementSignature, List[PairwiseCandidateRecord]]:
    """
    Bucket agreement-pool rows by consensus signature.

    Args:
        group_items (List[PairwiseCandidateRecord]): Rows in the agreement pool.
        config (FilterConfig): Filtering configuration.

    Returns:
        Dict[AgreementSignature, List[PairwiseCandidateRecord]]: Signature buckets.
    """
    signature_buckets: Dict[AgreementSignature, List[PairwiseCandidateRecord]] = (
        defaultdict(list)
    )
    for candidate in group_items:
        signature_buckets[_agreement_signature(candidate, config)].append(candidate)
    return signature_buckets


def _resolve_required_consensus_size(
    config: FilterConfig, agreement_pool_size: int
) -> int:
    """
    Resolve the minimum winning bucket size required for consensus.

    Args:
        config (FilterConfig): Filtering configuration.
        agreement_pool_size (int): Number of judge rows in the agreement pool.

    Returns:
        int: Required winning bucket size.
    """
    return config.min_consensus_judges or agreement_pool_size


def _select_consensus_bucket(
    signature_buckets: Dict[AgreementSignature, List[PairwiseCandidateRecord]],
    required_consensus_size: int,
) -> tuple[Optional[AgreementSignature], List[PairwiseCandidateRecord]]:
    """
    Select the winning consensus bucket when one is large enough.

    Args:
        signature_buckets (Dict[AgreementSignature, List[PairwiseCandidateRecord]]):
            Consensus signature buckets.
        required_consensus_size (int): Minimum winning bucket size.

    Returns:
        tuple[Optional[AgreementSignature], List[PairwiseCandidateRecord]]: Winning
            signature and rows, or `(None, [])` when no bucket passes.
    """
    if not signature_buckets:
        return None, []
    winning_signature, winning_bucket = max(
        signature_buckets.items(), key=lambda item: len(item[1])
    )
    if len(winning_bucket) < required_consensus_size:
        return None, []
    return winning_signature, winning_bucket


def _summarize_group_filter_outcome(
    group_items: List[PairwiseCandidateRecord],
    individually_kept: List[PairwiseCandidateRecord],
    agreement_pool: List[PairwiseCandidateRecord],
    *,
    rows_excluded_tied: int,
    required_consensus_size: Optional[int],
    signature_buckets: Dict[AgreementSignature, List[PairwiseCandidateRecord]],
    selected_signature: Optional[AgreementSignature],
    selected_bucket: List[PairwiseCandidateRecord],
    outcome: str,
) -> GroupFilterOutcome:
    """
    Build a compact audit record for a source group.

    Args:
        group_items (List[PairwiseCandidateRecord]): All rows in the source group.
        individually_kept (List[PairwiseCandidateRecord]): Rows kept after per-row filters.
        agreement_pool (List[PairwiseCandidateRecord]): Rows considered for consensus.
        rows_excluded_tied (int): Rows removed by tied-overall exclusion.
        required_consensus_size (Optional[int]): Minimum consensus bucket size.
        signature_buckets (Dict[AgreementSignature, List[PairwiseCandidateRecord]]):
            Agreement buckets for the group.
        selected_signature (Optional[AgreementSignature]): Winning signature when present.
        selected_bucket (List[PairwiseCandidateRecord]): Winning bucket rows.
        outcome (str): Group outcome label.

    Returns:
        GroupFilterOutcome: Summary payload for the group.
    """
    exemplar = group_items[0]
    return GroupFilterOutcome(
        source_key=exemplar.source_key,
        persona=exemplar.persona,
        prompt_type=exemplar.prompt_type,
        model_pair=exemplar.model_pair,
        pairwise_judgment_type=exemplar.pairwise_judgment_type,
        total_rows=len(group_items),
        individually_kept_rows=len(individually_kept),
        rows_rejected_before_consensus=len(group_items) - len(individually_kept),
        rows_excluded_tied_before_consensus=rows_excluded_tied,
        agreement_pool_size=len(agreement_pool),
        required_consensus_size=required_consensus_size,
        consensus_bucket_size=len(selected_bucket),
        signature_counts={
            _serialize_signature(signature): len(bucket)
            for signature, bucket in signature_buckets.items()
        },
        selected_signature=(
            _serialize_signature(selected_signature)
            if selected_signature is not None
            else None
        ),
        outcome=outcome,
    )


def apply_candidate_filters(
    candidates: List[PairwiseCandidateRecord], config: FilterConfig
) -> FilterResult:
    """
    Apply configured filters to pairwise candidates.

    Args:
        candidates (List[PairwiseCandidateRecord]): Discovered candidates.
        config (FilterConfig): Filtering configuration.

    Returns:
        FilterResult: Kept and rejected candidates plus audit metadata.

    Raises:
        ValueError: If a required filter removes all candidates.
    """
    grouped = _group_by_source_key(candidates)
    kept: List[PairwiseCandidateRecord] = []
    rejected: List[PairwiseCandidateRecord] = []
    rejection_counts: Counter[str] = Counter()
    agreement_stats: Counter[str] = Counter()
    group_outcomes: List[GroupFilterOutcome] = []

    if config.require_judge_consensus:
        agreement_stats["groups_total"] = len(grouped)

    for source_key, group_items in grouped.items():
        individually_kept: List[PairwiseCandidateRecord] = []
        for candidate in group_items:
            reason = _individual_rejection_reason(candidate, config)
            if reason is not None:
                rejected.append(candidate)
                rejection_counts[reason] += 1
                continue
            individually_kept.append(candidate)

        if not config.require_judge_consensus:
            kept.extend(individually_kept)
            group_outcomes.append(
                _summarize_group_filter_outcome(
                    group_items,
                    individually_kept,
                    individually_kept,
                    rows_excluded_tied=0,
                    required_consensus_size=None,
                    signature_buckets={},
                    selected_signature=None,
                    selected_bucket=[],
                    outcome="kept_without_consensus_filter",
                )
            )
            continue

        # First apply per-row filters, then run consensus only on the surviving pool.
        agreement_pool, rows_excluded_tied = _build_agreement_pool(
            individually_kept,
            config,
            rejected,
            rejection_counts,
            agreement_stats,
        )
        logger.debug(
            "Consensus pool for %s: total_rows=%d individually_kept=%d pool=%d",
            source_key,
            len(group_items),
            len(individually_kept),
            len(agreement_pool),
        )

        if len(agreement_pool) < config.min_judges_in_pool:
            for candidate in agreement_pool:
                rejected.append(candidate)
                rejection_counts["judge_consensus_insufficient_pool"] += 1
            agreement_stats["groups_failing_pool_threshold"] += 1
            group_outcomes.append(
                _summarize_group_filter_outcome(
                    group_items,
                    individually_kept,
                    agreement_pool,
                    rows_excluded_tied=rows_excluded_tied,
                    required_consensus_size=config.min_judges_in_pool,
                    signature_buckets={},
                    selected_signature=None,
                    selected_bucket=[],
                    outcome="rejected_insufficient_consensus_pool",
                )
            )
            continue

        agreement_stats["groups_with_sufficient_pool"] += 1
        required_consensus_size = _resolve_required_consensus_size(
            config, len(agreement_pool)
        )
        if required_consensus_size > len(agreement_pool):
            for candidate in agreement_pool:
                rejected.append(candidate)
                rejection_counts["judge_consensus_insufficient_pool"] += 1
            agreement_stats["groups_failing_pool_threshold"] += 1
            group_outcomes.append(
                _summarize_group_filter_outcome(
                    group_items,
                    individually_kept,
                    agreement_pool,
                    rows_excluded_tied=rows_excluded_tied,
                    required_consensus_size=required_consensus_size,
                    signature_buckets={},
                    selected_signature=None,
                    selected_bucket=[],
                    outcome="rejected_required_consensus_exceeds_pool",
                )
            )
            continue

        signature_buckets = _build_signature_buckets(agreement_pool, config)
        logger.debug(
            "Consensus signatures for %s: %s",
            source_key,
            {
                _serialize_signature(signature): len(bucket)
                for signature, bucket in signature_buckets.items()
            },
        )
        selected_signature, selected_bucket = _select_consensus_bucket(
            signature_buckets, required_consensus_size
        )

        if not selected_bucket:
            for candidate in agreement_pool:
                rejected.append(candidate)
                rejection_counts["judge_consensus_no_consensus_bucket"] += 1
            agreement_stats["groups_failing_consensus"] += 1
            group_outcomes.append(
                _summarize_group_filter_outcome(
                    group_items,
                    individually_kept,
                    agreement_pool,
                    rows_excluded_tied=rows_excluded_tied,
                    required_consensus_size=required_consensus_size,
                    signature_buckets=signature_buckets,
                    selected_signature=None,
                    selected_bucket=[],
                    outcome="rejected_no_consensus_bucket",
                )
            )
            continue

        agreement_stats["groups_passing_consensus"] += 1
        agreement_stats[f"winning_consensus_size_{len(selected_bucket)}"] += 1

        winning_ids = {id(candidate) for candidate in selected_bucket}
        # When a group passes 2-of-3 consensus, dissenting judges are still rejected.
        for candidate in agreement_pool:
            if id(candidate) in winning_ids:
                kept.append(candidate)
            else:
                rejected.append(candidate)
                rejection_counts["judge_consensus_dissenting_judge"] += 1

        group_outcomes.append(
            _summarize_group_filter_outcome(
                group_items,
                individually_kept,
                agreement_pool,
                rows_excluded_tied=rows_excluded_tied,
                required_consensus_size=required_consensus_size,
                signature_buckets=signature_buckets,
                selected_signature=selected_signature,
                selected_bucket=selected_bucket,
                outcome="kept_consensus_bucket",
            )
        )

    if not kept:
        raise ValueError(
            "No candidates remained after filtering. Rejection summary: "
            f"{dict(rejection_counts)}"
        )
    logger.info(
        "Filtered candidates: discovered_rows=%d kept_rows=%d rejected_rows=%d kept_items=%d rejected_items=%d",
        len(candidates),
        len(kept),
        len(rejected),
        len({candidate.source_key for candidate in kept}),
        len({candidate.source_key for candidate in rejected}),
    )
    return FilterResult(
        kept=kept,
        rejected=rejected,
        rejection_counts=dict(rejection_counts),
        agreement_stats=dict(agreement_stats),
        group_outcomes=group_outcomes,
    )
