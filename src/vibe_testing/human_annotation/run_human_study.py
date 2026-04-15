"""CLI entrypoint for standalone human annotation workflows."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

from src.vibe_testing.human_annotation import (
    assign_items_to_annotators,
    apply_candidate_filters,
    build_study_artifact_paths,
    compute_annotation_analysis_summary,
    compute_study_stats,
    discover_pairwise_candidates,
    exclude_prior_selection_plan_candidates,
    export_human_pairwise_artifacts,
    generate_qualtrics_artifacts,
    include_sample_type_candidates,
    load_human_annotation_config,
    process_qualtrics_results,
    resolve_repeat_candidates,
    sample_candidates,
    sample_candidates_for_annotators,
    save_filter_summary,
    save_processed_outputs,
)
from src.vibe_testing.human_annotation.filters import FilterResult
from src.vibe_testing.human_annotation.schemas import (
    HumanAnnotationConfig,
    PairwiseCandidateRecord,
    ProcessedAnnotationRecord,
    SampledAnnotationItem,
)
from src.vibe_testing.utils import load_json, setup_logger

logger = logging.getLogger(__name__)


def _configure_logging(log_path: str) -> None:
    """
    Configure file and console logging for the workflow and child modules.

    Args:
        log_path (str): Log file path for this invocation.

    Returns:
        None.
    """
    configured_logger = setup_logger(log_path, logger_name="vibe.human_annotation")
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(configured_logger.level)
        for handler in configured_logger.handlers:
            root_logger.addHandler(handler)


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the human annotation module.

    Args:
        None.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Standalone pairwise human annotation workflow."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the study YAML config."
    )
    parser.add_argument(
        "--log-path",
        default="human_annotation.log",
        help="Log file path for this invocation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "prepare-study",
        help="Discover, filter, sample, assign, and generate Qualtrics files.",
    )

    process_parser = subparsers.add_parser(
        "process-results",
        help="Process Qualtrics CSV exports into canonical annotation records.",
    )
    process_parser.add_argument(
        "--responses",
        nargs="+",
        required=True,
        help="One or more Qualtrics CSV export files.",
    )
    process_parser.add_argument(
        "--annotator-name",
        default=None,
        help="Optional runtime annotator name to store in processed outputs.",
    )

    export_parser = subparsers.add_parser(
        "export-human-judge",
        help="Export processed human annotations into Stage-5b-compatible artifacts.",
    )
    export_parser.add_argument(
        "--processed-json",
        default=None,
        help="Optional processed annotation JSON path. Defaults to the study workspace file.",
    )

    full_parser = subparsers.add_parser(
        "process-and-export",
        help="Process Qualtrics CSV exports and immediately export human judge artifacts.",
    )
    full_parser.add_argument(
        "--responses",
        nargs="+",
        required=True,
        help="One or more Qualtrics CSV export files.",
    )
    full_parser.add_argument(
        "--annotator-name",
        default=None,
        help="Optional runtime annotator name to store in processed outputs.",
    )
    return parser


def _load_processed_records(path: Path) -> List[ProcessedAnnotationRecord]:
    """
    Load processed records from a saved JSON file.

    Args:
        path (Path): JSON file path.

    Returns:
        List[ProcessedAnnotationRecord]: Restored processed records.
    """
    payload = load_json(str(path))
    if not isinstance(payload, list):
        raise ValueError(f"Processed annotation JSON must contain a list: {path}")
    records: List[ProcessedAnnotationRecord] = []
    for row in payload:
        records.append(ProcessedAnnotationRecord(**row))
    logger.info("Loaded %d processed records from %s", len(records), path)
    return records


def _log_config_summary(config: HumanAnnotationConfig) -> None:
    """
    Log the high-level study configuration at startup.

    Args:
        config (HumanAnnotationConfig): Loaded study configuration.

    Returns:
        None.
    """
    paths = build_study_artifact_paths(config)
    logger.info(
        "Study config: source_dir=%s personas=%s prompt_types=%s model_pairs=%s judges=%s",
        config.source.base_results_dir,
        config.source.personas or ["*"],
        config.source.prompt_types or ["*"],
        config.source.model_pairs or ["*"],
        config.source.judges or ["*"],
    )
    logger.info(
        "Consensus config: enabled=%s scope=%s min_judges_in_pool=%d min_consensus_judges=%s exclude_tied_overall_from_consensus=%s",
        config.filters.require_judge_consensus,
        config.filters.consensus_scope,
        config.filters.min_judges_in_pool,
        config.filters.min_consensus_judges,
        config.filters.exclude_tied_overall_from_consensus,
    )
    logger.info(
        "Study outputs: workspace=%s export_base_dir=%s sampled_target=%s annotators=%d",
        paths.study_dir,
        config.outputs.export_base_dir,
        config.allocation.total_samples,
        len(config.annotators.annotator_ids),
    )


def _log_filter_summary(
    discovered: Sequence[PairwiseCandidateRecord], filter_result: FilterResult
) -> None:
    """
    Log high-signal filtering counts after candidate filtering.

    Args:
        discovered (Sequence[PairwiseCandidateRecord]): Discovered candidates.
        filter_result (FilterResult): Filter result payload.

    Returns:
        None.
    """
    logger.info(
        "Filter summary: discovered_rows=%d discovered_items=%d kept_rows=%d kept_items=%d rejected_rows=%d rejected_items=%d",
        len(discovered),
        len({candidate.source_key for candidate in discovered}),
        len(filter_result.kept),
        len({candidate.source_key for candidate in filter_result.kept}),
        len(filter_result.rejected),
        len({candidate.source_key for candidate in filter_result.rejected}),
    )
    logger.info("Top rejection counts: %s", dict(filter_result.rejection_counts))
    if filter_result.agreement_stats:
        logger.info("Agreement counters: %s", filter_result.agreement_stats)


def _prepare_study(config: HumanAnnotationConfig) -> None:
    """
    Run discovery, filtering, sampling, assignment, and Qualtrics export.

    Args:
        config (HumanAnnotationConfig): Loaded study configuration.

    Returns:
        None.
    """
    paths = build_study_artifact_paths(config)
    discovered = discover_pairwise_candidates(config)
    filter_result = apply_candidate_filters(discovered, config.filters)
    _log_filter_summary(discovered, filter_result)
    allowlisted_sample_types = list(config.filters.include_sample_types)
    allowlisted_sample_types.extend(config.annotators.calibration_sample_types)
    type_filtered_candidates, sample_type_inclusion_audit = include_sample_type_candidates(
        filter_result.kept,
        replace(
            config.filters,
            include_sample_types=allowlisted_sample_types,
        ),
    )
    repeat_candidates, pinned_source_keys, repeat_inclusion_audit = (
        resolve_repeat_candidates(type_filtered_candidates, config.filters)
    )
    eligible_candidates, prior_selection_plan_audit = (
        exclude_prior_selection_plan_candidates(
            repeat_candidates,
            config.filters,
            protected_source_keys=pinned_source_keys,
        )
    )
    sampled = sample_candidates_for_annotators(
        eligible_candidates,
        config.allocation,
        config.annotators,
        include_sample_types=config.filters.include_sample_types,
        pinned_source_keys=pinned_source_keys,
        repeat_audit=repeat_inclusion_audit,
        sample_type_fields=config.annotators.balance_by,
    )
    assignments = assign_items_to_annotators(
        sampled,
        config.annotators,
        random_seed=config.allocation.random_seed,
    )
    logger.info(
        "Assignment composition: %s",
        dict(Counter(assignment.annotator_id for assignment in assignments)),
    )
    generate_qualtrics_artifacts(config, sampled, assignments)
    save_filter_summary(
        config,
        discovered_candidates=discovered,
        filter_result=filter_result,
        prior_selection_plan_audit=prior_selection_plan_audit,
        sample_type_inclusion_audit=sample_type_inclusion_audit,
        repeat_inclusion_audit=repeat_inclusion_audit,
        sampled_items=sampled,
        assignments=assignments,
    )
    logger.info("Study preparation complete under %s", paths.study_dir)


def _process_results(
    config: HumanAnnotationConfig,
    responses: Sequence[str],
    annotator_name: str | None = None,
) -> List[ProcessedAnnotationRecord]:
    """
    Process response CSVs and persist processed outputs.

    Args:
        config (HumanAnnotationConfig): Loaded study configuration.
        responses (Sequence[str]): Response CSV paths.
        annotator_name (str | None): Optional runtime annotator name override.

    Returns:
        List[ProcessedAnnotationRecord]: Processed records.
    """
    paths = build_study_artifact_paths(config)
    logger.info("Processing response files: %s", list(responses))
    records = process_qualtrics_results(
        config=config,
        manifest_path=paths.selection_manifest_path,
        response_paths=responses,
        annotator_name=annotator_name,
    )
    stats_payload = compute_study_stats(records)
    analysis_payload = compute_annotation_analysis_summary(records)
    save_processed_outputs(
        config,
        records,
        stats_payload,
        analysis_payload=analysis_payload,
    )
    logger.info(
        "Processed responses summary: records=%d overall_choices=%s",
        len(records),
        stats_payload.get("overall_choice_counts", {}),
    )
    return records


def _export_human_judge(
    config: HumanAnnotationConfig,
    *,
    processed_json: str | None = None,
    records: Sequence[ProcessedAnnotationRecord] | None = None,
) -> None:
    """
    Export processed records into canonical human-judge Stage-5b artifacts.

    Args:
        config (HumanAnnotationConfig): Loaded study configuration.
        processed_json (str | None): Optional processed-record JSON path.
        records (Sequence[ProcessedAnnotationRecord] | None): Optional in-memory
            processed records.

    Returns:
        None.
    """
    paths = build_study_artifact_paths(config)
    if records is None:
        resolved_path = (
            Path(processed_json).expanduser().resolve()
            if processed_json
            else paths.processed_json_path
        )
        records = _load_processed_records(resolved_path)
    written = export_human_pairwise_artifacts(config, records)
    logger.info("Human judge export complete: written_artifacts=%d", len(written))


def _process_and_export(
    config: HumanAnnotationConfig,
    responses: Sequence[str],
    annotator_name: str | None = None,
) -> None:
    """
    Process Qualtrics results and export Stage-5b human-judge artifacts.

    Args:
        config (HumanAnnotationConfig): Loaded study configuration.
        responses (Sequence[str]): Response CSV paths.
        annotator_name (str | None): Optional runtime annotator name override.

    Returns:
        None.
    """
    records = _process_results(config, responses, annotator_name=annotator_name)
    _export_human_judge(config, records=records)


def main() -> None:
    """
    Run the standalone human annotation CLI.

    Args:
        None.

    Returns:
        None.
    """
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_path)
    config = load_human_annotation_config(args.config)
    _log_config_summary(config)

    if args.command == "prepare-study":
        _prepare_study(config)
        return

    if args.command == "process-results":
        _process_results(
            config,
            list(args.responses),
            annotator_name=args.annotator_name,
        )
        return

    if args.command == "export-human-judge":
        _export_human_judge(config, processed_json=args.processed_json)
        return

    if args.command == "process-and-export":
        _process_and_export(
            config,
            list(args.responses),
            annotator_name=args.annotator_name,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
