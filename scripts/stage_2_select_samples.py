"""
Stage 2: Sample Selection

This script selects a subset of benchmark samples that are relevant to a
given user profile.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path to allow direct script execution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.vibe_testing.benchmarks.benchmarks import (
    load_benchmark,
    load_and_unify_benchmarks,
)
from src.vibe_testing.data_utils import BenchmarkSample, UserProfile
from src.vibe_testing.pathing import selection_stage_dir
from src.vibe_testing.selection import SampleSelector
from src.vibe_testing.utils import (
    add_common_args,
    get_stage_logger_name,
    load_json,
    resolve_run_context,
    save_json,
    seed_everything,
    setup_logger,
)


def main(args: Optional[List[str]] = None, **kwargs):
    """Main execution function for the sample selection stage."""
    parser = argparse.ArgumentParser(description="Stage 2: Select benchmark samples.")
    parser.add_argument(
        "--user-profile",
        type=str,
        required=True,
        help="Path to the structured user profile JSON file.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="mbpp,ds1000",
        required=True,
        help="Names of the benchmarks to load, separated by commas.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Optional override path for the selected samples JSON file.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=2,
        help="Total number of samples to select.",
    )
    add_common_args(parser)
    parsed_args = parser.parse_args(args)

    seed_everything(parsed_args.seed)

    run_context = resolve_run_context("stage_2_select_samples", parsed_args)
    log_path = run_context.logs / "stage_2_select_samples.log"
    logger = setup_logger(
        str(log_path),
        logger_name=get_stage_logger_name("stage_2_select_samples"),
    )
    logger.info("Starting Stage 2: Sample Selection")
    logger.info("Run root resolved to %s", run_context.root)

    # --- Input Validation ---
    if not os.path.exists(parsed_args.user_profile):
        logger.error(f"User profile file not found: {parsed_args.user_profile}")
        sys.exit(1)

    if parsed_args.dry_run:
        logger.info("--- Dry Run Mode ---")
        logger.info(f"Would load user profile from: {parsed_args.user_profile}")
        logger.info(f"Would load benchmarks from: {parsed_args.benchmarks}")
        logger.info(
            "Would save selected samples to %s",
            parsed_args.output_file or "canonical run directory",
        )
        logger.info("Dry run complete. Exiting.")
        return

    # --- Core Logic ---
    profile_data = load_json(parsed_args.user_profile)
    user_profile = UserProfile(**profile_data[0])

    unified_dataset = load_and_unify_benchmarks(
        names=parsed_args.benchmarks, loader=load_benchmark
    )
    all_samples = []
    for s in unified_dataset:
        # Deserialize JSON string fields back into objects
        inputs = json.loads(s["inputs"]) if s.get("inputs") else None
        tests = json.loads(s["tests"]) if s.get("tests") else None
        metadata_from_benchmark = json.loads(s["metadata"]) if s.get("metadata") else {}

        metadata = {
            "language": s.get("language"),
            "difficulty": s.get("difficulty"),
            "inputs": inputs,
            "tests": tests,
        }
        if metadata_from_benchmark:
            metadata.update(metadata_from_benchmark)

        # Filter out None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        sample = BenchmarkSample(
            sample_id=str(s.get("task_id", "")),
            source_benchmark=s.get("benchmark", "unknown"),
            prompt=s.get("prompt", ""),
            ground_truth=s.get("ground_truth"),
            metadata=metadata,
        )
        all_samples.append(sample)

    selector = SampleSelector(total_samples=parsed_args.total_samples)
    selected_samples = selector.select(user_profile, all_samples)

    # --- Save Output ---   
    samples_to_save = [sample.model_dump() for sample in selected_samples]
    selection_method = parsed_args.selection_method or "default_selection"
    selection_dir = selection_stage_dir(
        parsed_args.run_base_dir, parsed_args.user_group, selection_method
    )
    selection_dir.mkdir(parents=True, exist_ok=True)
    if parsed_args.output_file:
        output_path = Path(parsed_args.output_file)
    else:
        benchmarks_token = parsed_args.benchmarks.replace(",", "-").replace(" ", "")
        detail = (
            f"persona-{run_context.user_group}:benchmarks-{benchmarks_token.lower()}"
        )
        output_path = run_context.artifact_path(
            base=selection_dir,
            artifact_type="selected_samples",
            evaluation_type="selection",
            detail=detail,
            version=0,
            ext="json",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_json(samples_to_save, str(output_path))
    logger.info(f"Successfully selected {len(selected_samples)} samples.")
    logger.info(f"Saved selected samples to {output_path}")
    print(
        f"Stage 2 Complete. Selected {len(selected_samples)} samples, saved to {output_path}"
    )


if __name__ == "__main__":
    main()
