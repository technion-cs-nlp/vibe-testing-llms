"""
Stage 4: Evaluate Vibe Dataset

This script now supports multiple evaluation paradigms:

1. Function-level rigor (e.g., MBPP+): measures Pass@k on single functions.
2. Repository-level patch validation (e.g., SWE-Bench): validates multi-file patches
   through an external harness.
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path to allow direct script execution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.vibe_testing.evaluation.strategies import (
    EvaluationContext,
    EvaluationOrchestrator,
    FunctionLevelStrategy,
    PatchLevelStrategy,
)
from src.vibe_testing.models.base import BaseModel
from src.vibe_testing.pathing import objective_stage_dir, vibe_dataset_stage_dir
from src.vibe_testing.utils import (
    add_common_args,
    ensure_environment,
    get_stage_logger_name,
    load_config,
    load_json,
    load_model_from_config,
    resolve_run_context,
    seed_everything,
    setup_logger,
)


def _maybe_import_unsloth_for_gpu_runs() -> None:
    """
    Best-effort import of Unsloth for GPU runs.

    IMPORTANT:
    This must NOT run at module import time. The sandbox uses multiprocessing
    (often with spawn), which re-imports the main module in child processes.
    Importing torch/unsloth at import time can dominate the sandbox startup and
    cause false timeouts in objective evaluation.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                import unsloth  # type: ignore  # noqa: F401
            except Exception:
                # Optimization hint only; ignore failures.
                pass
    except Exception:
        # If torch isn't available (CPU-only minimal env), do nothing.
        return


def load_control_prompts(
    control_prompts_dir: str,
    required_benchmarks: set,
    logger,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Load control prompts from JSON files and index them by (benchmark, task_id).

    Args:
        control_prompts_dir (str): Directory containing control prompt JSON files.
        required_benchmarks (set): Set of benchmark names that are present in the dataset.
        logger: Logger instance for warnings/errors.

    Returns:
        Dict[Tuple[str, str], List[Dict[str, Any]]]: Dictionary mapping (benchmark, task_id) to list of control variations.
    """
    control_prompts: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    control_prompts_path = Path(control_prompts_dir)

    if not control_prompts_path.exists():
        logger.warning(
            f"Control prompts directory not found: {control_prompts_dir}. "
            "Control prompts will not be loaded."
        )
        return control_prompts

    # Map from control prompt file names to benchmark names
    benchmark_file_map = {
        "mbppplus_variations.json": "mbpp_plus",
        "humanevalplus_variations.json": "humaneval_plus",
    }

    # Only load control prompts for benchmarks that exist in the dataset
    for filename, benchmark_name in benchmark_file_map.items():
        if benchmark_name not in required_benchmarks:
            logger.debug(
                f"Skipping control prompts for {benchmark_name} (not in dataset)"
            )
            continue

        file_path = control_prompts_path / filename
        if not file_path.exists():
            logger.debug(f"Control prompts file not found: {file_path}")
            # continue
            raise ValueError(f"Control prompts file not found: {file_path}")

        try:
            variations_data = load_json(str(file_path))
            if not isinstance(variations_data, list):
                logger.warning(
                    f"Control prompts file {file_path} does not contain a list. Skipping."
                )
                # continue
                raise ValueError(
                    f"Control prompts file {file_path} does not contain a list."
                )

            # Group variations by (benchmark, task_id)
            for variation_entry in variations_data:
                original_row_data = variation_entry.get("original_row_data", {})
                task_id = original_row_data.get("task_id")
                if not task_id:
                    logger.warning(
                        f"Control prompt variation missing task_id in {file_path}. Skipping."
                    )
                    # continue
                    raise ValueError(
                        f"Control prompt variation missing task_id in {file_path}. Skipping."
                    )

                # Convert task_id to string for consistent matching
                task_id_str = str(task_id)

                # Extract the prompt text
                prompt_text = variation_entry.get("prompt", "")
                if not prompt_text:
                    logger.warning(
                        f"Control prompt variation for task_id {task_id_str} missing prompt. Skipping."
                    )
                    # continue
                    raise ValueError(
                        f"Control prompt variation for task_id {task_id_str} missing prompt. Skipping."
                    )

                # Create a variation entry in the format expected by the evaluation system
                control_variation = {
                    "sample_id": task_id_str,
                    "source_benchmark": benchmark_name,
                    "variation_id": f"control_{variation_entry.get('variation_count', 'unknown')}",
                    "modified_prompt": prompt_text,
                    "prompt_type": "control",
                    "variant_label": "control",
                    "applied_changes": [],
                    "verification": {
                        "same_end_goal": True,
                        "same_ground_truth": True,
                    },
                    "original_row_data": original_row_data,
                }

                key = (benchmark_name, task_id_str)
                if key not in control_prompts:
                    control_prompts[key] = []
                control_prompts[key].append(control_variation)

            logger.info(
                f"Loaded control prompts for {benchmark_name} from {filename} "
                f"({len([k for k in control_prompts.keys() if k[0] == benchmark_name])} unique task_ids)"
            )

        except Exception as e:
            logger.error(f"Failed to load control prompts from {file_path}: {e}")
            # continue
            raise ValueError(f"Failed to load control prompts from {file_path}: {e}")

    return control_prompts


def inject_control_prompts(
    dataset_samples: List[Dict[str, Any]],
    control_prompts: Dict[Tuple[str, str], List[Dict[str, Any]]],
    logger,
) -> List[Dict[str, Any]]:
    """
    Inject control prompt variations into dataset samples.

    For each sample, if control prompts exist for its (benchmark, task_id) pair, add them as variations.
    The number of control variations added matches the number of personalized variations
    for that sample (or all available if fewer personalized variations exist).

    Args:
        dataset_samples (List[Dict[str, Any]]): List of dataset samples from Stage 3.
        control_prompts (Dict[Tuple[str, str], List[Dict[str, Any]]]): Control prompts indexed by (benchmark, task_id).
        logger: Logger instance for warnings/errors.

    Returns:
        List[Dict[str, Any]]: Dataset samples with control prompts injected as variations.
    """
    enriched_samples = []

    for sample in dataset_samples:
        original_sample = sample.get("original_sample", {})
        if not original_sample:
            # Skip samples without original_sample structure
            enriched_samples.append(sample)
            continue

        sample_id = original_sample.get("sample_id")
        source_benchmark = original_sample.get("source_benchmark")

        # Convert task_id to string for consistent matching
        task_id_str = str(sample_id)

        # Match by both benchmark and task_id
        key = (source_benchmark, task_id_str)

        # Get available control variations for this (benchmark, task_id) pair
        available_control_variations = control_prompts.get(key, [])

        # Get the number of personalized variations for this sample
        personalized_variations = sample.get("variations", [])
        num_personalized = len(personalized_variations)

        # Select the first N control variations where N = num_personalized
        # If there are fewer control variations than personalized, use all available
        num_control_to_add = min(num_personalized, len(available_control_variations))

        # Create a copy of the sample to avoid modifying the original
        enriched_sample = json.loads(json.dumps(sample))  # Deep copy

        # Add control variations to the variations list
        if "variations" not in enriched_sample:
            enriched_sample["variations"] = []

        # Add the selected control variations
        for i in range(num_control_to_add):
            control_var = json.loads(
                json.dumps(available_control_variations[i])
            )  # Deep copy
            enriched_sample["variations"].append(control_var)

        if num_control_to_add > 0:
            logger.debug(
                f"Injected {num_control_to_add} control variations for sample {sample_id} "
                f"(benchmark={source_benchmark}, had {num_personalized} personalized variations)"
            )

        enriched_samples.append(enriched_sample)

    return enriched_samples


def main(args: Optional[List[str]] = None, model: Optional[BaseModel] = None):
    """Main execution function for the model evaluation stage."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Evaluate a model on a vibe dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--vibe-dataset-dir",
        type=str,
        required=False,
        help="Optional override directory containing the vibe-testing dataset JSON files from Stage 3.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model's configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Optional override directory to save the evaluation results.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Maximum number of samples (JSON files) to evaluate. Use <=0 for all.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="function",
        choices=["function", "patch"],
        help="Which evaluation paradigm to use (function-level vs. patch-level).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        help="Optional YAML/JSON file with evaluation-specific configuration (e.g., pass@k, harness command).",
    )
    parser.add_argument(
        "--enable-llm-code-extraction",
        action="store_true",
        help=(
            "If set, enable an evaluation-only fallback that asks the evaluated model to "
            "extract runnable solution code from its own raw response when execution fails. "
            "This does not change the stored raw model response; it only affects evaluation."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-prompt",
        type=str,
        default=None,
        help=(
            "Optional prompt template string used for LLM code extraction fallback. "
            "Supports placeholders: {language}, {entry_point}, {raw_output}. "
            "Overrides any dataset-config-provided template."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-prompt-file",
        type=str,
        default=None,
        help=(
            "Optional path to a text file containing the LLM code extraction prompt template. "
            "If provided, takes precedence over --llm-code-extraction-prompt."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-generation-kwargs",
        type=str,
        default=None,
        help=(
            "Optional JSON object (as a string) with generation kwargs to use for the "
            'LLM code extraction fallback (e.g., \'{"temperature":0,"max_new_tokens":800}\').'
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-retry-on-timeout",
        type=int,
        default=1,
        help=(
            "If LLM-assisted code extraction is enabled, retry the extracted-code execution "
            "this many additional times when the sandbox returns a timeout (default: 1). "
            "Set to 0 to disable retry."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-retry-timeout",
        type=int,
        default=60,
        help=(
            "Sandbox timeout (seconds) to use when executing code extracted by the LLM. "
            "This can be higher than the default 15s to accommodate heavier plus harnesses "
            "(e.g., cold imports like numpy). Default: 60."
        ),
    )
    parser.add_argument(
        "--debug-force-raw-output",
        type=str,
        default=None,
        help=(
            "DEBUG ONLY: If provided, bypass model generation and use this string as the "
            "raw_output for every evaluation unit. Useful for testing code sanitization and "
            "LLM-assisted code extraction fallback."
        ),
    )
    parser.add_argument(
        "--debug-force-raw-output-file",
        type=str,
        default=None,
        help=(
            "DEBUG ONLY: Path to a UTF-8 text file whose contents will be used as "
            "--debug-force-raw-output. Takes precedence over --debug-force-raw-output."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for model generation (evaluation units per call).",
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=["original", "personalized", "control"],
        help=(
            "Prompt types to evaluate in this run. "
            "When omitted, all prompt types are evaluated together and results are written "
            "to the legacy Stage 4 directory layout for backward compatibility. "
            "When provided, exactly one prompt type must be specified and outputs are written "
            "under a prompt_type_<name>/ subdirectory."
        ),
    )
    parser.add_argument(
        "--control-prompts-dir",
        type=str,
        default="data/control_prompts",
        help=(
            "Optional directory containing control prompts keyed by sample_id. "
            "Each JSON file should contain at least 'sample_id' and 'prompt' fields. "
            "When provided and 'control' is selected as a prompt type, control prompts will "
            "be evaluated alongside originals/personalized prompts that share the same sample_id."
        ),
    )
    parser.add_argument(
        "--reference-persona",
        type=str,
        default="novice_user",
        help="Optional reference persona name (default: novice_user). If the current user-group is not the reference, Stage 4 will only evaluate personalized variations to save resources.",
    )
    add_common_args(parser)
    parsed_args = parser.parse_args(args)

    seed_everything(parsed_args.seed)

    # Initialize HM_HOME-backed caches and HF env vars for the main Stage 4 process.
    # NOTE: This is intentionally inside main() to avoid import-time side effects
    # in sandbox child processes.
    ensure_environment(should_print=False)

    # Best-effort optimization hint for GPU runs. Must stay inside main().
    _maybe_import_unsloth_for_gpu_runs()

    run_context = resolve_run_context("stage_4_evaluate_vibe_dataset", parsed_args)
    prompt_types = parsed_args.prompt_types
    active_prompt_type = prompt_types[0] if prompt_types else None
    generator_model = parsed_args.generator_model_name or parsed_args.model_name
    filter_model = parsed_args.filter_model_name or "none"
    evaluated_model = parsed_args.evaluated_model_name or parsed_args.model_name
    dataset_dir = (
        Path(parsed_args.vibe_dataset_dir)
        if parsed_args.vibe_dataset_dir
        else vibe_dataset_stage_dir(
            parsed_args.run_base_dir,
            parsed_args.user_group,
            generator_model,
            filter_model,
        )
        / f"dataset_{run_context.run_id}"
    )
    base_objective_dir = objective_stage_dir(
        parsed_args.run_base_dir,
        parsed_args.user_group,
        evaluated_model,
        generator_model,
        filter_model,
        prompt_type=active_prompt_type,
    )
    output_dir = (
        Path(parsed_args.output_dir)
        if parsed_args.output_dir
        else base_objective_dir / f"{parsed_args.dataset_type}_{run_context.run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = run_context.root / "tmp_stage_4"
    temp_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_context.logs / "stage_4_evaluate.log"
    logger = setup_logger(
        str(log_path),
        logger_name=get_stage_logger_name("stage_4_evaluate_vibe_dataset"),
    )
    logger.info("Starting Stage 4: Model Evaluation")
    logger.info("Run root resolved to %s", run_context.root)

    # --- Optimization: Skip original/control if not reference persona ---
    if (
        prompt_types is None
        and parsed_args.reference_persona
        and parsed_args.reference_persona != parsed_args.user_group
    ):
        logger.info(
            "Current persona '%s' is not the reference persona '%s'. "
            "Defaulting to 'personalized' prompt type to save resources.",
            parsed_args.user_group,
            parsed_args.reference_persona,
        )
        prompt_types = ["personalized"]
        active_prompt_type = "personalized"

    if prompt_types is not None and len(prompt_types) != 1:
        # For now we require a single prompt type per Stage 4 invocation to keep
        # directory layout and filtering simple. Higher-level orchestrators are
        # responsible for looping over multiple prompt types when needed.
        raise SystemExit(
            "Stage 4 currently supports exactly one --prompt-types value per run. "
            "Please invoke the stage separately for each prompt type."
        )

    # --- Input Validation ---
    if not dataset_dir.is_dir():
        logger.error(f"Input directory not found: {dataset_dir}")
        sys.exit(1)
    if model is None and not os.path.exists(parsed_args.model_config):
        logger.error(f"Model config file not found: {parsed_args.model_config}")
        sys.exit(1)

    dataset_config = load_dataset_config(parsed_args.dataset_config)
    logger.info("Loaded dataset configuration keys: %s", list(dataset_config.keys()))
    batch_size = dataset_config.get("batch_size", parsed_args.batch_size)

    if parsed_args.dry_run:
        logger.info("--- Dry Run Mode ---")
        logger.info(f"Would load dataset from: {dataset_dir}")
        logger.info(f"Would load model config from: {parsed_args.model_config}")
        logger.info(
            "Would instantiate model, select the '%s' strategy, and run evaluation.",
            parsed_args.dataset_type,
        )
        logger.info(f"Would save detailed JSON and summary CSV to: {output_dir}")
        logger.info("Dry run complete. Exiting.")
        return

    # --- Load Model ---
    if model is None:
        logger.info(f"Loading model from config: {parsed_args.model_config}")
        model = load_model_from_config(parsed_args.model_config)
    else:
        logger.info("Using pre-loaded model.")

    logger.info(f"Successfully loaded model of type '{type(model).__name__}'.")

    # --- Dataset Loading ---
    dataset_files = sorted(glob(str(dataset_dir / "*.json")))
    logger.info("Found %d dataset files.", len(dataset_files))
    if parsed_args.num_samples > 0:
        logger.info(
            "Limiting evaluation to the first %d samples.", parsed_args.num_samples
        )
    if not dataset_files:
        logger.error("No dataset files found to evaluate.")
        sys.exit(1)

    dataset_samples = [load_json(path) for path in dataset_files]

    # --- Load and Inject Control Prompts ---
    control_prompts = {}
    # Only load control prompts if we're evaluating control type or all types
    if active_prompt_type is None or active_prompt_type == "control":
        # First, identify which benchmarks are present in the dataset
        benchmarks_in_dataset = set()
        for sample in dataset_samples:
            original_sample = sample.get("original_sample", {})
            if original_sample:
                source_benchmark = original_sample.get("source_benchmark")
                if source_benchmark:
                    benchmarks_in_dataset.add(source_benchmark)

        logger.info(f"Found benchmarks in dataset: {sorted(benchmarks_in_dataset)}")

        if benchmarks_in_dataset:
            control_prompts = load_control_prompts(
                parsed_args.control_prompts_dir, benchmarks_in_dataset, logger
            )
            if control_prompts:
                logger.info(
                    f"Loaded control prompts for {len(control_prompts)} unique (benchmark, task_id) pairs. "
                    "Injecting into dataset samples..."
                )
                dataset_samples = inject_control_prompts(
                    dataset_samples, control_prompts, logger
                )
    else:
        logger.debug(
            f"Skipping control prompt loading (active prompt type: {active_prompt_type})"
        )

    # --- Strategy Orchestration ---
    context_extra = dict(dataset_config.get("context_extra", {}))
    context_extra.setdefault("run_name", run_context.run_name)
    context_extra.setdefault("evaluated_model_label", evaluated_model)
    context_extra.setdefault("model_config_path", parsed_args.model_config)
    for key in ("num_completions", "generation_kwargs"):
        if key in dataset_config:
            context_extra.setdefault(key, dataset_config[key])
    if "patch_harness" in dataset_config:
        context_extra["patch_harness"] = dataset_config["patch_harness"]
    context_extra.setdefault("batch_size", batch_size)

    # --- Optional: evaluation-only LLM code extraction fallback ---
    enable_llm_code_extraction = bool(
        parsed_args.enable_llm_code_extraction
        or dataset_config.get("enable_llm_code_extraction", False)
    )
    context_extra["enable_llm_code_extraction"] = enable_llm_code_extraction

    extraction_prompt_template = dataset_config.get("llm_code_extraction_prompt")
    if parsed_args.llm_code_extraction_prompt_file:
        try:
            with open(
                parsed_args.llm_code_extraction_prompt_file, "r", encoding="utf-8"
            ) as f:
                extraction_prompt_template = f.read()
        except OSError as exc:
            raise SystemExit(
                f"Failed to read --llm-code-extraction-prompt-file: {exc}"
            ) from exc
    if parsed_args.llm_code_extraction_prompt:
        extraction_prompt_template = parsed_args.llm_code_extraction_prompt
    if extraction_prompt_template:
        context_extra["llm_code_extraction_prompt"] = extraction_prompt_template

    # If LLM extraction is enabled and no explicit prompt template was supplied,
    # try to reuse the model config's extract_code_prompt (when available).
    # This is a practical default because many model configs already carry a
    # hardened extraction prompt.
    if (
        context_extra.get("enable_llm_code_extraction")
        and "llm_code_extraction_prompt" not in context_extra
    ):
        try:
            model_cfg = load_config(parsed_args.model_config)
            extract_prompt = model_cfg.get("extract_code_prompt")
            if isinstance(extract_prompt, str) and extract_prompt.strip():
                context_extra["llm_code_extraction_prompt"] = extract_prompt
                logger.info(
                    "Using extract_code_prompt from model config for LLM code extraction."
                )
        except Exception as exc:
            logger.debug(
                "Failed to load extract_code_prompt from model config: %s", exc
            )

    extraction_generation_kwargs = dataset_config.get(
        "llm_code_extraction_generation_kwargs"
    )
    if parsed_args.llm_code_extraction_generation_kwargs:
        try:
            extraction_generation_kwargs = json.loads(
                parsed_args.llm_code_extraction_generation_kwargs
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "Invalid JSON provided to --llm-code-extraction-generation-kwargs."
            ) from exc
    if isinstance(extraction_generation_kwargs, dict) and extraction_generation_kwargs:
        context_extra["llm_code_extraction_generation_kwargs"] = (
            extraction_generation_kwargs
        )
    context_extra["llm_code_extraction_retry_on_timeout"] = int(
        max(0, parsed_args.llm_code_extraction_retry_on_timeout)
    )
    context_extra["llm_code_extraction_retry_timeout"] = int(
        max(1, parsed_args.llm_code_extraction_retry_timeout)
    )

    # --- Debug-only: force raw output for every evaluation unit ---
    forced_raw_output = None
    if parsed_args.debug_force_raw_output_file:
        try:
            with open(
                parsed_args.debug_force_raw_output_file, "r", encoding="utf-8"
            ) as f:
                forced_raw_output = f.read()
        except OSError as exc:
            raise SystemExit(
                f"Failed to read --debug-force-raw-output-file: {exc}"
            ) from exc
    elif parsed_args.debug_force_raw_output:
        forced_raw_output = parsed_args.debug_force_raw_output
    if forced_raw_output is not None:
        context_extra["debug_force_raw_output"] = forced_raw_output

    # Ensure a deterministic seed is available for model.generate calls.
    # For API-backed models (e.g., GPT-5.1), this will be passed as the
    # native `seed` parameter; for HF models, global RNG seeding suffices.
    if "generation_kwargs" not in context_extra:
        context_extra["generation_kwargs"] = {"seed": parsed_args.seed}
    elif "seed" not in context_extra["generation_kwargs"]:
        context_extra["generation_kwargs"]["seed"] = parsed_args.seed

    context = EvaluationContext(
        run_dir=str(run_context.root),
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        extra=context_extra,
    )
    strategy = build_strategy(parsed_args.dataset_type, context, dataset_config, logger)
    strategy.bind_model(model)
    orchestrator = EvaluationOrchestrator(strategy, logger)
    evaluation_output = orchestrator.run(
        dataset_samples,
        batch_size=batch_size,
        allowed_prompt_types=[active_prompt_type] if active_prompt_type else None,
    )

    logger.info("Evaluation summary: %s", evaluation_output.get("summary"))
    print(f"Stage 4 Complete. Processed {len(dataset_files)} samples.")
    print(f"Summary artifacts: {evaluation_output.get('summary')}")


def load_dataset_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load dataset-level configuration from YAML or JSON.
    """
    if not config_path:
        return {}
    _, ext = os.path.splitext(config_path.lower())
    if ext in (".yml", ".yaml"):
        return load_config(config_path)
    return load_json(config_path)


def build_strategy(
    dataset_type: str,
    context: EvaluationContext,
    dataset_config: Dict[str, Any],
    logger,
):
    """
    Instantiate the appropriate strategy for the requested dataset type.
    """
    if dataset_type == "function":
        passk = dataset_config.get("passk", [1])
        return FunctionLevelStrategy(context, passk=passk, logger=logger)
    if dataset_type == "patch":
        harness_config = dataset_config.get("patch_harness")
        return PatchLevelStrategy(context, harness_config=harness_config, logger=logger)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


if __name__ == "__main__":
    main()
