"""
Stage 1: User Profiling

This script takes a user's configuration file (e.g., a YAML with a
free-text description) and generates a structured user profile.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path to allow direct script execution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.vibe_testing.models.base import BaseModel
from src.vibe_testing.pathing import profile_stage_dir
from src.vibe_testing.profiling import Profiler
from src.vibe_testing.utils import (
    add_common_args,
    get_stage_logger_name,
    load_config,
    load_model_from_config,
    resolve_run_context,
    save_json,
    seed_everything,
    setup_logger,
)


def main(args: Optional[List[str]] = None, model: Optional[BaseModel] = None):
    """Main execution function for the user profiling stage."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate a structured user profile."
    )
    parser.add_argument(
        "--user-config",
        type=str,
        required=True,
        help="Path to the user's configuration YAML file.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model's configuration YAML file for profiling.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Optional override path for the generated user profile JSON file.",
    )
    # Add arguments for generation parameter overrides
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generation max_new_tokens.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override generation top_p.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation do_sample.",
    )

    parser.add_argument(
        "--debug-mode",
        action="store_true",
        default=False,
        help="Enable debug mode. Saves the complete prompt and the json response in the run directory.",
    )

    add_common_args(parser)
    parsed_args = parser.parse_args(args)

    seed_everything(parsed_args.seed)

    run_context = resolve_run_context("stage_1_profile_user", parsed_args)
    log_path = run_context.logs / "stage_1_profile_user.log"
    logger = setup_logger(
        str(log_path),
        debug=parsed_args.debug_mode,
        logger_name=get_stage_logger_name("stage_1_profile_user"),
    )
    logger.info("Starting Stage 1: User Profiling")
    logger.info("Run root resolved to %s", run_context.root)

    # --- Input Validation ---
    if not os.path.exists(parsed_args.user_config):
        logger.error(f"User config file not found: {parsed_args.user_config}")
        sys.exit(1)

    if model is None and not os.path.exists(parsed_args.model_config):
        logger.error(f"Model config file not found: {parsed_args.model_config}")
        sys.exit(1)

    if parsed_args.dry_run:
        logger.info("--- Dry Run Mode ---")
        logger.info(f"Would load user config from: {parsed_args.user_config}")
        logger.info(f"Would load model config from: {parsed_args.model_config}")
        logger.info(f"Would instantiate Profiler with the specified model.")
        logger.info(
            "Would save structured profile to %s",
            parsed_args.output_file or "canonical run directory",
        )
        logger.info("Dry run complete. Exiting.")
        return

    # --- Core Logic ---
    user_config = load_config(parsed_args.user_config)

    if model is None:
        logger.info(f"Loading model from config: {parsed_args.model_config}")
        model = load_model_from_config(parsed_args.model_config)
    else:
        logger.info("Using pre-loaded model.")

    profiler = Profiler(model=model)

    # Collect override generation parameters
    generation_kwargs = {
        "temperature": parsed_args.temperature,
        "max_new_tokens": parsed_args.max_new_tokens,
        "top_p": parsed_args.top_p,
        "do_sample": parsed_args.do_sample,
        # Propagate the CLI seed down to the model for reproducible API calls.
        "seed": parsed_args.seed,
    }
    # Filter out any non-provided arguments except seed (which always has a value)
    generation_kwargs = {
        k: v for k, v in generation_kwargs.items() if v is not None
    }

    user_profile = profiler.create_profile(user_config, generation_kwargs)

    # --- Save Output ---
    profile_dir = profile_stage_dir(parsed_args.run_base_dir, parsed_args.user_group)
    profile_dir.mkdir(parents=True, exist_ok=True)
    if parsed_args.output_file:
        output_path = Path(parsed_args.output_file)
    else:
        detail = f"persona-{run_context.user_group}"
        output_path = run_context.artifact_path(
            base=profile_dir,
            artifact_type="user_profile",
            evaluation_type="profiling",
            detail=detail,
            version=0,
            ext="json",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    profiling_model_name = getattr(model, "model_name", parsed_args.model_name)
    profile_payload = user_profile.model_dump()
    profile_payload["_model_metadata"] = {
        "role": "profiling",
        "model_name": profiling_model_name,
        "model_config_path": parsed_args.model_config,
    }

    # Pydantic's .model_dump() is used to get a serializable dictionary
    save_json([profile_payload], str(output_path))
    logger.info(f"Successfully generated and saved user profile to:\n{output_path}")
    print(
        f"=========Stage 1 Complete========\nProfile saved to:\n{output_path}\n========\n"
    )


if __name__ == "__main__":
    main()
