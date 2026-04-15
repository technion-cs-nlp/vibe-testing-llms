"""
Orchestrator for the Vibe-Testing Pipeline

This script runs the entire vibe-testing pipeline from start to finish,
chaining the individual stages together.
"""

# IMPORTANT (Unsloth):
# Unsloth warns if it is imported after transformers, but on CPU-only machines
# importing unsloth can raise (it requires a torch accelerator). To remain usable
# on CPU-only runs, we only import unsloth when CUDA is available.
try:
    import torch

    if torch.cuda.is_available():
        try:
            import unsloth  # type: ignore  # noqa: F401
        except Exception:
            # Best-effort only: this import is an optimization hint, not a requirement.
            pass
except Exception:
    pass

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, cast

# Add src to path to allow utils import
src_path = os.path.join(os.path.dirname(__file__), "..")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.vibe_testing.utils import (
    archive_existing_directory,
    get_stage_logger_name,
    load_model_from_config,
    seed_everything,
    setup_logger,
)
from src.vibe_testing.pathing import (
    analysis_stage_dir,
    build_run_context,
    build_run_id,
    indicator_stage_dir,
    objective_stage_dir,
    profile_stage_dir,
    selection_stage_dir,
    subjective_stage_dir,
    vibe_dataset_stage_dir,
)

# Import main functions from stage scripts for direct execution
from scripts.stage_1_profile_user import main as profile_user_main
from scripts.stage_2_select_samples import main as select_samples_main
from scripts.stage_3_build_vibe_dataset import main as build_dataset_main
from scripts.stage_4_evaluate_vibe_dataset import main as evaluate_dataset_main
from scripts.stage_5_subjective_vibe_evaluation import (
    main as subjective_evaluation_main,
)
from scripts.stage_5c_indicator_scores import main as indicator_scores_main
from scripts.stage_6_analyze_results import main as analyze_results_main


def run_command(command: List[str], logger):
    """Executes a shell command and logs its output."""
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()

    if stdout:
        logger.info(f"STDOUT:\n{stdout}")
    if stderr:
        logger.warning(f"STDERR:\n{stderr}")

    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    logger.info("Command executed successfully.")


def _archive_stage3_outputs_if_needed(
    dataset_dir: Path, force_recreate: bool, logger: logging.Logger
) -> None:
    """
    Archives existing Stage 3 outputs when the caller requests a forced rebuild.

    Args:
        dataset_dir (Path): Target dataset directory selected for Stage 3 outputs.
        force_recreate (bool): Whether to archive prior artifacts.
        logger (logging.Logger): Logger used for status messages.
    """
    if not force_recreate:
        return
    archive_dir = archive_existing_directory(dataset_dir, logger)
    if archive_dir:
        logger.warning(
            "Existing Stage 3 dataset contents found at %s; archived to %s before rebuild.",
            dataset_dir,
            archive_dir,
        )


def main():
    """Main execution function to orchestrate the pipeline."""
    parser = argparse.ArgumentParser(description="Run the full vibe-testing pipeline.")
    parser.add_argument("--user-config", type=str, help="Path to the user config YAML.")
    parser.add_argument(
        "--model-config", type=str, help="Path to the model config YAML."
    )
    parser.add_argument(
        "--judge-model-config",
        type=str,
        help="Path to the judge model config YAML.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Path(s) to raw benchmark files.",
    )
    parser.add_argument(
        "--user-group",
        type=str,
        required=True,
        help="User persona or cohort identifier for structured runs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name evaluated throughout the run.",
    )
    parser.add_argument(
        "--run-base-dir",
        type=str,
        default="runs",
        help="Base directory for all structured runs (default: runs).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=False,
        help="Optional run identifier. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="no_name",
        help="Human-readable label incorporated into auto-generated run IDs.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=list(range(1, 7)),
        help="Which pipeline stage(s) to run (e.g., 1 2 5). Default is all stages.",
    )
    # parser.add_argument(
    #     "--total-samples",
    #     type=int,
    #     default=10,
    #     help="[Stage 2] Total number of samples to select.",
    # )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="[Stage 3 & 4] Number of samples to process.",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=1,
        help="[Stage 3] Number of prompt variations to generate per sample.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["function", "patch"],
        default="function",
        help="[Stage 4] Evaluation paradigm for Stage 4.",
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=["original", "personalized", "control"],
        help=(
            "[Stages 4, 5, 5B] Prompt types to operate on. "
            "When omitted, all prompt types are processed together using the legacy "
            "directory layout. When provided, higher-level orchestrators should invoke "
            "the pipeline separately per prompt type."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for model/ judge calls where supported (stages 3, 4, 5).",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["subprocess", "direct"],
        default="subprocess",
        help="Execution mode: 'subprocess' runs each stage in a new process, 'direct' calls functions internally.",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default=None,
        help="Selection strategy label used to organize Stage 2 outputs.",
    )
    parser.add_argument(
        "--generator-model-name",
        type=str,
        default=None,
        help="Generator/personalization model label for Stage 3/4/5 directories.",
    )
    parser.add_argument(
        "--filter-model-name",
        type=str,
        default=None,
        help="Filter/ranking model label for Stage 3/4/5 directories.",
    )
    parser.add_argument(
        "--evaluated-model-name",
        type=str,
        default=None,
        help="Evaluated model label for Stage 4/5/6 directories.",
    )
    parser.add_argument(
        "--judge-model-name",
        type=str,
        default=None,
        help="Judge model label for Stage 5/6 directories.",
    )
    parser.add_argument(
        "--objective-results-dir",
        type=str,
        default=None,
        help="Optional override path to Stage 4 objective results directory.",
    )
    parser.add_argument(
        "--vibe-dataset-dir",
        type=str,
        default=None,
        help="Optional path to an existing Stage 3 dataset directory. When omitted, the orchestrator builds the canonical dataset path from run metadata.",
    )
    parser.add_argument(
        "--reference-persona",
        type=str,
        default="novice_user",
        help="Optional reference persona name for optimized evaluation (default: novice_user).",
    )
    parser.add_argument(
        "--reference-results-dir",
        type=str,
        default=None,
        help="Optional reference results directory for Stages 5 and 5B.",
    )
    parser.add_argument(
        "--pairwise-correctness-mode",
        type=str,
        choices=["ignore", "dimension", "gate"],
        default="ignore",
        help=(
            "How to incorporate pass@1 correctness into per-sample pairwise "
            "winner determination (forwarded to Stage 6)."
        ),
    )
    parser.add_argument(
        "--pairwise-include-plus-correctness",
        action="store_true",
        help=(
            "Include plus-test pass@1 as additional correctness dimension "
            "(forwarded to Stage 6)."
        ),
    )
    parser.add_argument(
        "--enable-llm-code-extraction",
        action="store_true",
        help=(
            "[Stage 4] Enable LLM-assisted code extraction fallback on execution failures. "
            "When enabled, Stage 4 may re-prompt the evaluated model to extract runnable code "
            "from its own raw response and retry execution."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-prompt",
        type=str,
        default=None,
        help="[Stage 4] Optional LLM code extraction prompt template (forwarded to Stage 4).",
    )
    parser.add_argument(
        "--llm-code-extraction-prompt-file",
        type=str,
        default=None,
        help=(
            "[Stage 4] Optional file path containing the LLM code extraction prompt template "
            "(forwarded to Stage 4). Takes precedence over --llm-code-extraction-prompt."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-generation-kwargs",
        type=str,
        default=None,
        help=(
            "[Stage 4] Optional JSON generation kwargs for extraction (forwarded to Stage 4)."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-retry-on-timeout",
        type=int,
        default=None,
        help=(
            "[Stage 4] Optional override for the number of extra retries on timeout when "
            "executing LLM-extracted code (forwarded to Stage 4)."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-retry-timeout",
        type=int,
        default=None,
        help=(
            "[Stage 4] Optional override for the sandbox timeout (seconds) used when "
            "executing LLM-extracted code (forwarded to Stage 4)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Pass the dry-run flag to all stages."
    )
    parser.add_argument(
        "--force-stage3-recreate",
        action="store_true",
        help="Archive any existing Stage 3 dataset outputs into 'old/<timestamp>' before rebuilding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible pipeline runs (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Global log level for pipeline and src.vibe_testing modules (default: INFO).",
    )
    args = parser.parse_args()
    stages = set(args.stages)

    # Normalize requested log level; default is INFO.
    level_name = args.log_level.upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    # Seed random number generators once at the orchestrator level so any
    # top-level randomness (e.g., run ID generation) is reproducible.
    seed_everything(args.seed)

    # --- Argument validation ---
    if 1 in stages and not args.user_config:
        parser.error("--user-config is required for stage 1")
    if 4 in stages and not args.model_config:
        parser.error("--model-config is required for stage 4")
    if (1 in stages or 5 in stages or 6 in stages) and not args.judge_model_config:
        parser.error("--judge-model-config is required for stages 1, 5, and 6")
    if 2 in stages and not args.benchmarks:
        parser.error("--benchmarks is required for stage 2")

    run_name = args.run_name
    run_id = args.run_id or build_run_id(run_name=run_name)
    args.run_id = run_id
    user_group = args.user_group
    model_name = args.model_name
    base_dir = args.run_base_dir
    selection_method = args.selection_method or "default_selection"
    generator_model_name = args.generator_model_name or model_name
    filter_model_name = args.filter_model_name or "none"
    evaluated_model_name = args.evaluated_model_name or model_name
    judge_model_name = args.judge_model_name or evaluated_model_name

    active_prompt_type = args.prompt_types[0] if args.prompt_types else None

    stage_contexts = {
        "stage_1_profile_user": build_run_context(
            "stage_1_profile_user",
            user_group,
            model_name,
            run_id=run_id,
            base_dir=base_dir,
            run_name=run_name,
        ),
        "stage_2_select_samples": build_run_context(
            "stage_2_select_samples",
            user_group,
            model_name,
            run_id=run_id,
            base_dir=base_dir,
            run_name=run_name,
        ),
        "stage_3_build_vibe_dataset": build_run_context(
            "stage_3_build_vibe_dataset",
            user_group,
            model_name,
            run_id=run_id,
            base_dir=base_dir,
            run_name=run_name,
        ),
        "stage_4_evaluate_vibe_dataset": build_run_context(
            "stage_4_evaluate_vibe_dataset",
            user_group,
            model_name,
            run_id=run_id,
            base_dir=base_dir,
            run_name=run_name,
        ),
        "stage_5_subjective_vibe_evaluation": build_run_context(
            "stage_5_subjective_vibe_evaluation",
            user_group,
            model_name,
            run_id=run_id,
            base_dir=base_dir,
            run_name=run_name,
        ),
        "stage_5c_indicator_scores": build_run_context(
            "stage_5c_indicator_scores",
            user_group,
            model_name,
            run_id=run_id,
            base_dir=base_dir,
            run_name=run_name,
        ),
        "stage_6_analyze_results": build_run_context(
            "stage_6_analyze_results",
            user_group,
            model_name,
            run_id=run_id,
            base_dir=base_dir,
            run_name=run_name,
        ),
    }
    pipeline_ctx = build_run_context(
        "pipeline_orchestrator",
        user_group,
        model_name,
        run_id=run_id,
        base_dir=base_dir,
        run_name=run_name,
    )

    profile_dir = profile_stage_dir(base_dir, user_group)
    profile_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = selection_stage_dir(base_dir, user_group, selection_method)
    selection_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = (
        Path(args.vibe_dataset_dir).expanduser().resolve()
        if args.vibe_dataset_dir
        else vibe_dataset_stage_dir(
            base_dir, user_group, generator_model_name, filter_model_name
        )
        / f"dataset_{run_id}"
    )
    if args.vibe_dataset_dir is None or 3 in stages:
        dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_results_dir = (
        Path(args.objective_results_dir)
        if args.objective_results_dir
        else (
            objective_stage_dir(
                base_dir,
                user_group,
                evaluated_model_name,
                generator_model_name,
                filter_model_name,
                prompt_type=active_prompt_type,
            )
            / f"{args.dataset_type}_{run_id}"
        )
    )
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    subjective_dir = subjective_stage_dir(
        base_dir,
        user_group,
        evaluated_model_name,
        judge_model_name,
        generator_model_name,
        filter_model_name,
        prompt_type=active_prompt_type,
    )
    subjective_dir.mkdir(parents=True, exist_ok=True)
    indicator_dir = indicator_stage_dir(
        base_dir,
        user_group,
        evaluated_model_name,
        generator_model_name,
        filter_model_name,
        prompt_type=active_prompt_type,
    )
    indicator_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = analysis_stage_dir(
        base_dir,
        user_group,
        evaluated_model_name,
        judge_model_name,
        generator_model_name,
        filter_model_name,
        prompt_type=active_prompt_type,
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    dataset_type = args.dataset_type

    profile_path = stage_contexts["stage_1_profile_user"].artifact_path(
        base=profile_dir,
        artifact_type="user_profile",
        evaluation_type="profiling",
        detail=f"persona-{user_group}",
        version=0,
        ext="json",
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    benchmarks_arg = ",".join(args.benchmarks) if args.benchmarks else ""
    benchmarks_detail = (
        benchmarks_arg.replace(",", "-").replace(" ", "").lower()
        if benchmarks_arg
        else "unspecified"
    )
    selected_samples_path = stage_contexts["stage_2_select_samples"].artifact_path(
        base=selection_dir,
        artifact_type="selected_samples",
        evaluation_type="selection",
        detail=f"persona-{user_group}:benchmarks-{benchmarks_detail}",
        version=0,
        ext="json",
    )
    selected_samples_path.parent.mkdir(parents=True, exist_ok=True)

    subjective_results_path = stage_contexts[
        "stage_5_subjective_vibe_evaluation"
    ].artifact_path(
        base=subjective_dir,
        artifact_type="subjective_scores",
        evaluation_type="analysis",
        detail=f"model-{model_name}:persona-{user_group}",
        version=0,
        ext="json",
    )
    subjective_results_path.parent.mkdir(parents=True, exist_ok=True)
    indicator_results_path = stage_contexts["stage_5c_indicator_scores"].artifact_path(
        base=indicator_dir,
        artifact_type="indicator_scores",
        evaluation_type="analysis",
        detail=f"model-{model_name}:persona-{user_group}",
        version=0,
        ext="json",
    )
    indicator_results_path.parent.mkdir(parents=True, exist_ok=True)

    detailed_scores_path = stage_contexts["stage_6_analyze_results"].artifact_path(
        base=analysis_dir,
        artifact_type="detailed_scores",
        evaluation_type="stage6",
        detail=f"model-{model_name}",
        version=0,
        ext="json",
    )
    summary_path = stage_contexts["stage_6_analyze_results"].artifact_path(
        base=analysis_dir,
        artifact_type="summary_report",
        evaluation_type="stage6",
        detail=f"model-{model_name}",
        version=0,
        ext="json",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_scores_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        str(pipeline_ctx.logs / "pipeline_orchestrator.log"),
        logger_name=get_stage_logger_name("pipeline_orchestrator"),
    )

    # Apply the requested log level to the pipeline logger.
    logger.setLevel(numeric_level)

    # Gate console verbosity by level, but keep file handler at DEBUG so we
    # always capture detailed logs on disk.
    for handler in logger.handlers:
        # FileHandler is a subclass of StreamHandler; we only want to adjust
        # the plain stdout handler created in setup_logger.
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(numeric_level)

    # Configure a package-level logger for src.vibe_testing.* modules so that
    # library logs (e.g., personalization, models) are visible by default.
    pkg_logger = logging.getLogger("src.vibe_testing")
    pkg_logger.setLevel(numeric_level)

    # Reuse the same handlers as the pipeline logger, but avoid duplicates.
    for handler in logger.handlers:
        if handler not in pkg_logger.handlers:
            pkg_logger.addHandler(handler)

    # Prevent bubbling beyond this package logger to root, which avoids
    # double-printing if root is ever configured elsewhere.
    pkg_logger.propagate = False

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job_id:
        logger.info(
            "Detected Slurm environment: job_id=%s, array_task_id=%s",
            slurm_job_id,
            slurm_task_id or "none",
        )

    logger.info("Starting pipeline run with run_id=%s", run_id)
    logger.info("Executing stages: %s", sorted(list(stages)))
    logger.info("Using global random seed: %s", args.seed)

    paths = {
        "profile": str(profile_path),
        "selected": str(selected_samples_path),
        "dataset": str(dataset_dir),
        "raw_results": str(raw_results_dir),
        "subjective_results": str(subjective_results_path),
        "indicator_results": str(indicator_results_path),
        "scores": str(detailed_scores_path),
        "summary": str(summary_path),
    }

    common_flags = [
        "--user-group",
        user_group,
        "--model-name",
        model_name,
        "--run-base-dir",
        base_dir,
        "--run-id",
        run_id,
        "--run-name",
        run_name,
        "--selection-method",
        selection_method,
        "--generator-model-name",
        generator_model_name,
        "--filter-model-name",
        filter_model_name,
        "--evaluated-model-name",
        evaluated_model_name,
        "--judge-model-name",
        judge_model_name,
    ]
    if args.prompt_types:
        common_flags.append("--prompt-types")
        common_flags.extend(args.prompt_types)
    if args.dry_run:
        common_flags.append("--dry-run")
    common_flags.extend(["--seed", str(args.seed)])

    # --- Execute Pipeline ---
    if args.run_mode == "subprocess":
        run_pipeline_subprocess(
            args,
            stages,
            paths,
            common_flags,
            logger,
            benchmarks_arg,
            dataset_type,
            base_dir,
            user_group,
            str(analysis_dir),
        )
    else:
        run_pipeline_direct(
            args,
            stages,
            paths,
            logger,
            common_flags,
            benchmarks_arg,
            dataset_type,
            base_dir,
            user_group,
            str(analysis_dir),
        )


def run_pipeline_subprocess(
    args,
    stages,
    paths,
    common_flags,
    logger,
    benchmarks_arg: str,
    dataset_type: str,
    base_dir: str,
    user_group: str,
    analysis_dir: str,
):
    """Executes the pipeline stages as separate subprocesses."""
    logger.info("--- Running in SUBPROCESS mode ---")
    batch_size = getattr(args, "batch_size", 1)
    # --- Execute Pipeline Stages ---

    # Stage 1: Profile User
    if 1 in stages:
        logger.info("--- STAGE 1: PROFILE USER ---")
        cmd1 = [
            "python",
            "scripts/stage_1_profile_user.py",
            "--user-config",
            args.user_config,
            "--model-config",
            args.judge_model_config,
            "--output-file",
            paths["profile"],
        ] + common_flags
        run_command(cmd1, logger)

    # Stage 2: Select Samples
    if 2 in stages:
        logger.info("--- STAGE 2: SELECT SAMPLES ---")
        cmd2 = [
            "python",
            "scripts/stage_2_select_samples.py",
            "--user-profile",
            paths["profile"],
            "--benchmarks",
            benchmarks_arg,
            "--output-file",
            paths["selected"],
            "--total-samples",
            str(args.num_samples),  # to avoid duplication of arguments
        ] + common_flags
        run_command(cmd2, logger)

    # Stage 3: Build Dataset
    if 3 in stages:
        logger.info("--- STAGE 3: BUILD DATASET ---")
        _archive_stage3_outputs_if_needed(
            Path(paths["dataset"]), args.force_stage3_recreate, logger
        )
        cmd3 = [
            "python",
            "scripts/stage_3_build_vibe_dataset.py",
            "--user-profile",
            paths["profile"],
            "--selected-samples",
            paths["selected"],
            "--output-dir",
            paths["dataset"],
            "--model-config",
            args.judge_model_config,  # Using judge model for personalization
            "--num-samples",
            str(args.num_samples),
            "--num-variations",
            str(args.num_variations),
            "--batch-size",
            str(batch_size),
        ] + common_flags
        if args.force_stage3_recreate:
            cmd3.append("--force-recreate")
        run_command(cmd3, logger)

    # Stage 4: Evaluate Model
    if 4 in stages:
        logger.info("--- STAGE 4: EVALUATE MODEL ---")
        cmd4 = [
            "python",
            "scripts/stage_4_evaluate_vibe_dataset.py",
            "--vibe-dataset-dir",
            paths["dataset"],
            "--model-config",
            args.model_config,
            "--output-dir",
            paths["raw_results"],
            "--num-samples",
            str(args.num_samples),
            "--dataset-type",
            dataset_type,
            "--batch-size",
            str(batch_size),
        ] + common_flags
        if args.reference_persona:
            cmd4.extend(["--reference-persona", args.reference_persona])
        if args.enable_llm_code_extraction:
            cmd4.append("--enable-llm-code-extraction")
            if args.llm_code_extraction_prompt_file:
                cmd4.extend(
                    [
                        "--llm-code-extraction-prompt-file",
                        args.llm_code_extraction_prompt_file,
                    ]
                )
            elif args.llm_code_extraction_prompt:
                cmd4.extend(
                    [
                        "--llm-code-extraction-prompt",
                        args.llm_code_extraction_prompt,
                    ]
                )
            if args.llm_code_extraction_generation_kwargs:
                cmd4.extend(
                    [
                        "--llm-code-extraction-generation-kwargs",
                        args.llm_code_extraction_generation_kwargs,
                    ]
                )
            if args.llm_code_extraction_retry_on_timeout is not None:
                cmd4.extend(
                    [
                        "--llm-code-extraction-retry-on-timeout",
                        str(args.llm_code_extraction_retry_on_timeout),
                    ]
                )
            if args.llm_code_extraction_retry_timeout is not None:
                cmd4.extend(
                    [
                        "--llm-code-extraction-retry-timeout",
                        str(args.llm_code_extraction_retry_timeout),
                    ]
                )
        run_command(cmd4, logger)

    # Stage 5: Subjective Vibe Evaluation
    if 5 in stages:
        logger.info("--- STAGE 5: SUBJECTIVE VIBE EVALUATION ---")
        cmd5 = [
            "python",
            "scripts/stage_5_subjective_vibe_evaluation.py",
            "--user-profile",
            paths["profile"],
            "--raw-results",
            paths["raw_results"],
            "--vibe-dataset-dir",
            paths["dataset"],
            "--output-file",
            paths["subjective_results"],
            "--judge-model-config",
            args.judge_model_config,
            "--batch-size",
            str(batch_size),
        ] + common_flags
        if args.reference_results_dir:
            cmd5.extend(["--reference-results-dir", args.reference_results_dir])
        run_command(cmd5, logger)

        # Stage 5C: Non-LLM indicator scores (always generated alongside Stage 5)
        logger.info("--- STAGE 5C: INDICATOR SCORES (NON-LLM) ---")
        cmd5c = [
            "python",
            "scripts/stage_5c_indicator_scores.py",
            "--user-profile",
            paths["profile"],
            "--raw-results",
            paths["raw_results"],
            "--vibe-dataset-dir",
            paths["dataset"],
            "--output-file",
            paths["indicator_results"],
        ] + common_flags
        run_command(cmd5c, logger)

    # Stage 6: Analyze Results
    if 6 in stages:
        logger.info("--- STAGE 6: ANALYZE RESULTS ---")
        # Align Stage 6 invocation with the experiment orchestrator and
        # standalone script (scripts/3_analyze_results.sh). In this mode,
        # Stage 6 scans the full results directory for all personas and
        # uses the structured user profile JSON as input.
        cmd6 = [
            "python",
            "scripts/stage_6_analyze_results.py",
            "--results-dir",
            base_dir,
            "--personas",
            user_group,
            "--user-profiles",
            paths["profile"],
            "--analysis-output-dir",
            analysis_dir,
        ] + common_flags
        if args.pairwise_correctness_mode != "ignore":
            cmd6.extend(["--pairwise-correctness-mode", args.pairwise_correctness_mode])
        if args.pairwise_include_plus_correctness:
            cmd6.append("--pairwise-include-plus-correctness")
        run_command(cmd6, logger)


def run_pipeline_direct(
    args,
    stages,
    paths,
    logger,
    common_flags,
    benchmarks_arg: str,
    dataset_type: str,
    base_dir: str,
    user_group: str,
    analysis_dir: str,
):
    """Executes the pipeline stages by calling their main functions directly."""
    logger.info("--- Running in DIRECT mode ---")
    batch_size = getattr(args, "batch_size", 1)

    # --- Load Models ---
    model = None
    judge_model = None

    # Check if the same model config is used for both roles
    is_same_model_config = (
        args.model_config
        and args.judge_model_config
        and os.path.abspath(args.model_config)
        == os.path.abspath(args.judge_model_config)
    )

    if is_same_model_config:
        # If configs are the same, load the model only once.
        if any(s in stages for s in [1, 3, 4, 5, 6]):
            logger.info(
                f"Main model and judge model are the same. Loading once from: {args.model_config}"
            )
            shared_model = load_model_from_config(args.model_config)
            model = shared_model
            judge_model = shared_model
    else:
        # If configs are different, load them separately as needed.
        if 4 in stages and args.model_config:
            logger.info(f"Loading main model from: {args.model_config}")
            model = load_model_from_config(args.model_config)

        if any(s in stages for s in [1, 3, 5, 6]) and args.judge_model_config:
            logger.info(f"Loading judge model from: {args.judge_model_config}")
            judge_model = load_model_from_config(args.judge_model_config)

    # --- Stage 1: Profile User ---
    if 1 in stages:
        logger.info("--- STAGE 1: PROFILE USER ---")
        stage_args = [
            "--user-config",
            cast(str, args.user_config),
            "--model-config",
            cast(str, args.judge_model_config),
            "--output-file",
            paths["profile"],
        ] + common_flags
        profile_user_main(args=stage_args, model=judge_model)

    # --- Stage 2: Select Samples ---
    if 2 in stages:
        logger.info("--- STAGE 2: SELECT SAMPLES ---")
        stage_args = [
            "--user-profile",
            paths["profile"],
            "--benchmarks",
            benchmarks_arg,
            "--output-file",
            paths["selected"],
            "--total-samples",
            str(args.num_samples),
        ] + common_flags
        select_samples_main(args=stage_args)

    # --- Stage 3: Build Dataset ---
    if 3 in stages:
        logger.info("--- STAGE 3: BUILD DATASET ---")
        _archive_stage3_outputs_if_needed(
            Path(paths["dataset"]), args.force_stage3_recreate, logger
        )
        stage_args = [
            "--user-profile",
            paths["profile"],
            "--selected-samples",
            paths["selected"],
            "--output-dir",
            paths["dataset"],
            "--model-config",
            cast(str, args.judge_model_config),
            "--num-samples",
            str(args.num_samples),
            "--num-variations",
            str(args.num_variations),
            "--batch-size",
            str(batch_size),
        ] + common_flags
        if args.force_stage3_recreate:
            stage_args.append("--force-recreate")
        build_dataset_main(args=stage_args, model=judge_model)

    # --- Stage 4: Evaluate Model ---
    if 4 in stages:
        logger.info("--- STAGE 4: EVALUATE MODEL ---")
        stage_args = [
            "--vibe-dataset-dir",
            paths["dataset"],
            "--model-config",
            cast(str, args.model_config),
            "--output-dir",
            paths["raw_results"],
            "--num-samples",
            str(args.num_samples),
            "--dataset-type",
            dataset_type,
            "--batch-size",
            str(batch_size),
        ] + common_flags
        if args.reference_persona:
            stage_args.append("--reference-persona")
            stage_args.append(args.reference_persona)
        # IMPORTANT: Keep direct-mode Stage 4 flag forwarding in sync with subprocess mode.
        # Otherwise, CLI flags (like --enable-llm-code-extraction) appear on the outer
        # run, but are never seen by stage_4_evaluate_vibe_dataset.py.
        if getattr(args, "enable_llm_code_extraction", False):
            stage_args.append("--enable-llm-code-extraction")
            if getattr(args, "llm_code_extraction_prompt_file", None):
                stage_args.append("--llm-code-extraction-prompt-file")
                stage_args.append(args.llm_code_extraction_prompt_file)
            elif getattr(args, "llm_code_extraction_prompt", None):
                stage_args.append("--llm-code-extraction-prompt")
                stage_args.append(args.llm_code_extraction_prompt)
            if getattr(args, "llm_code_extraction_generation_kwargs", None):
                stage_args.append("--llm-code-extraction-generation-kwargs")
                stage_args.append(args.llm_code_extraction_generation_kwargs)
            if getattr(args, "llm_code_extraction_retry_on_timeout", None) is not None:
                stage_args.append("--llm-code-extraction-retry-on-timeout")
                stage_args.append(str(args.llm_code_extraction_retry_on_timeout))
            if getattr(args, "llm_code_extraction_retry_timeout", None) is not None:
                stage_args.append("--llm-code-extraction-retry-timeout")
                stage_args.append(str(args.llm_code_extraction_retry_timeout))
        evaluate_dataset_main(args=stage_args, model=model)

    # --- Stage 5: Subjective Vibe Evaluation ---
    if 5 in stages:
        logger.info("--- STAGE 5: SUBJECTIVE VIBE EVALUATION ---")
        stage_args = [
            "--user-profile",
            paths["profile"],
            "--raw-results",
            paths["raw_results"],
            "--vibe-dataset-dir",
            paths["dataset"],
            "--output-file",
            paths["subjective_results"],
            "--judge-model-config",
            cast(str, args.judge_model_config),
            "--batch-size",
            str(batch_size),
        ] + common_flags
        if args.reference_results_dir:
            stage_args.append("--reference-results-dir")
            stage_args.append(args.reference_results_dir)
        subjective_evaluation_main(args=stage_args, judge_model=judge_model)

        logger.info("--- STAGE 5C: INDICATOR SCORES (NON-LLM) ---")
        stage5c_args = [
            "--user-profile",
            paths["profile"],
            "--raw-results",
            paths["raw_results"],
            "--vibe-dataset-dir",
            paths["dataset"],
            "--output-file",
            paths["indicator_results"],
        ] + common_flags
        indicator_scores_main(args=stage5c_args)

    # --- Stage 6: Analyze Results ---
    if 6 in stages:
        logger.info("--- STAGE 6: ANALYZE RESULTS ---")
        # Use the same batch-scanning invocation as the subprocess mode so
        # that direct-mode runs behave identically.
        stage_args = [
            "--results-dir",
            base_dir,
            "--personas",
            user_group,
            "--user-profiles",
            paths["profile"],
            "--analysis-output-dir",
            str(analysis_dir),
        ] + common_flags
        if args.pairwise_correctness_mode != "ignore":
            stage_args.extend(
                ["--pairwise-correctness-mode", args.pairwise_correctness_mode]
            )
        if args.pairwise_include_plus_correctness:
            stage_args.append("--pairwise-include-plus-correctness")
        analyze_results_main(args=stage_args)


if __name__ == "__main__":
    main()
