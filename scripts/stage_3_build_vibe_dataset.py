"""
Stage 3: Build Vibe-Testing Dataset

This script personalizes the selected benchmark samples to create the final
vibe-testing dataset tailored to a specific user profile. It uses a language
model to generate and verify multiple prompt variations for each sample.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path to allow direct script execution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.vibe_testing.data_utils import BenchmarkSample, PersonalizedSample, UserProfile
from src.vibe_testing.models.base import BaseModel
from src.vibe_testing.pathing import normalize_token, vibe_dataset_stage_dir
from src.vibe_testing.personalization import TaskPersonalizer
from src.vibe_testing.utils import (
    add_common_args,
    archive_existing_directory,
    get_stage_logger_name,
    load_config,
    load_json,
    load_model_from_config,
    resolve_run_context,
    seed_everything,
    setup_logger,
)


def main(args: Optional[List[str]] = None, model: Optional[BaseModel] = None):
    """Main execution function for building the vibe-testing dataset."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Build vibe-testing dataset with personalized prompt variations."
    )
    parser.add_argument(
        "--user-profile",
        type=str,
        required=True,
        help="Path to the structured user profile JSON file.",
    )
    parser.add_argument(
        "--selected-samples",
        type=str,
        required=True,
        help="Path to the selected samples JSONL file from Stage 2.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model's configuration YAML file for personalization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Optional override directory for the final vibe-testing dataset files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to build the dataset for.",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=3,
        help="Number of prompt variations to generate per sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for personalization model calls (composition/verification).",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Archive any existing dataset contents under an 'old/<timestamp>' directory before rebuilding.",
    )
    add_common_args(parser)
    parsed_args = parser.parse_args(args)

    seed_everything(parsed_args.seed)

    run_context = resolve_run_context("stage_3_build_vibe_dataset", parsed_args)
    log_path = run_context.logs / "stage_3_build_dataset.log"
    logger = setup_logger(
        str(log_path),
        logger_name=get_stage_logger_name("stage_3_build_vibe_dataset"),
    )
    logger.info("Starting Stage 3: Vibe Dataset Construction")
    logger.info("Run root resolved to %s", run_context.root)
    generator_model = parsed_args.generator_model_name or parsed_args.model_name
    filter_model = parsed_args.filter_model_name or "none"
    if parsed_args.output_dir:
        dataset_dir = Path(parsed_args.output_dir)
    else:
        base_dataset_dir = vibe_dataset_stage_dir(
            parsed_args.run_base_dir,
            parsed_args.user_group,
            generator_model,
            filter_model,
        )
        dataset_dir = base_dataset_dir / f"dataset_{run_context.run_id}"

    def _has_active_artifacts(target_dir: Path) -> bool:
        """Checks whether the target directory contains non-archived artifacts."""
        ignored_children = {"old", "meta"}
        return target_dir.exists() and any(
            child.name not in ignored_children for child in target_dir.iterdir()
        )

    dataset_has_contents = _has_active_artifacts(dataset_dir)
    if dataset_has_contents and parsed_args.force_recreate:
        archive_dir = archive_existing_directory(dataset_dir, logger)
        dataset_has_contents = _has_active_artifacts(dataset_dir)
        if archive_dir:
            logger.warning(
                "Existing dataset artifacts detected at %s; archived to %s before rebuild.",
                dataset_dir,
                archive_dir,
            )
    if dataset_has_contents:
        logger.info(f"Dataset already exists in {dataset_dir}")
        logger.warning("Exiting and skipping Stage 3 dataset build!")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = dataset_dir / "meta"

    # --- Input Validation ---
    for f in [
        parsed_args.user_profile,
        parsed_args.selected_samples,
    ]:
        if not os.path.exists(f):
            logger.error(f"Input file not found: {f}")
            sys.exit(1)
    if model is None and not os.path.exists(parsed_args.model_config):
        logger.error(f"Model config not found: {parsed_args.model_config}")
        sys.exit(1)

    if parsed_args.dry_run:
        logger.info("--- Dry Run Mode ---")
        logger.info(f"Would load profile from: {parsed_args.user_profile}")
        logger.info(f"Would load samples from: {parsed_args.selected_samples}")
        logger.info(f"Would load model config from: {parsed_args.model_config}")
        logger.info(
            f"Would generate {parsed_args.num_variations} variations for each sample."
        )
        logger.info(f"Would save final dataset files to: {dataset_dir}")
        logger.info("Dry run complete. Exiting.")
        return

    # --- Load Model ---
    if model is None:
        logger.info(f"Loading model from config: {parsed_args.model_config}")
        model = load_model_from_config(parsed_args.model_config)
    else:
        logger.info("Using pre-loaded model.")

    # --- Core Logic ---
    profile_data = load_json(parsed_args.user_profile)
    user_profile = UserProfile(**profile_data[0])

    samples_data = load_json(parsed_args.selected_samples)
    logger.info(f"Loaded {len(samples_data)} selected samples.")
    if len(samples_data) < 1:
        raise ValueError(
            f"No selected samples found at {parsed_args.selected_samples}."
        )

    selected_samples = [BenchmarkSample(**s) for s in samples_data]

    # Propagate the run seed into the personalizer so API-backed models can
    # use deterministic sampling via their native `seed` parameter.
    personalizer = TaskPersonalizer(
        model=model,
        generation_kwargs={"seed": parsed_args.seed},
        batch_size=parsed_args.batch_size,
        artifacts_dir=str(meta_dir),
    )
    logger.info(
        f"Building dataset with {parsed_args.num_variations} variations per {len(selected_samples)} samples."
    )
    # --- Build Dataset ---
    personalized_samples = personalizer.build_dataset(
        selected_samples, user_profile, parsed_args.num_variations
    )

    # --- Save Output ---
    total_variations = 0
    personalization_model_name = getattr(model, "model_name", parsed_args.model_name)
    for personalized_sample in personalized_samples:
        sample_id = personalized_sample.original_sample.sample_id
        normalized_sample_id = normalize_token(sample_id or "sample")
        output_path = run_context.artifact_path(
            base=dataset_dir,
            artifact_type="3",
            evaluation_type="personalized",
            detail=f"sample-{personalized_sample.original_sample.source_benchmark}-{normalized_sample_id}",
            version=0,
            ext="json",
        )

        payload = personalized_sample.model_dump()
        payload["_model_metadata"] = {
            "role": "personalization",
            "model_name": personalization_model_name,
            "model_config_path": parsed_args.model_config,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        num_variations = len(personalized_sample.variations)
        total_variations += num_variations
        logger.info(
            "Saved %s variations for sample %s to %s",
            num_variations,
            sample_id,
            output_path,
        )

    logger.info(
        f"Successfully created a vibe dataset with {total_variations} total variations "
        f"across {len(personalized_samples)} samples."
    )
    logger.info(f"Saved dataset files to {dataset_dir}")
    print(
        f"Stage 3 Complete. Dataset with {total_variations} variations saved to {dataset_dir}"
    )


if __name__ == "__main__":
    main()
