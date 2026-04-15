"""
Stage 5: Subjective Vibe Evaluation

This script computes subjective, user-centered performance indicators that reflect
the "vibe dimensions" described in the paper.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.vibe_testing.models.base import BaseModel
from src.vibe_testing.pathing import (
    infer_prompt_type,
    subjective_stage_dir,
    vibe_dataset_stage_dir,
)
from src.vibe_testing.subjective_evaluation import (
    ModelGeneration,
    StaticEvaluationResult,
    SubjectiveVibeEvaluator,
    UserProfile,
)
from src.vibe_testing.utils import (
    add_common_args,
    get_stage_logger_name,
    load_json,
    load_model_from_config,
    resolve_run_context,
    save_json,
    seed_everything,
    setup_logger,
)


def _resolve_dataset_dir(raw_results_dir: str) -> str:
    """
    Infer the Stage 3 dataset directory relative to the raw results path.
    """
    parent = os.path.dirname(os.path.dirname(os.path.abspath(raw_results_dir)))
    return os.path.join(parent, "3_vibe_dataset")


def _unwrap_modified_prompt(payload: Any) -> str:
    """
    Normalize the stored modified prompt payload into plain text.

    Args:
        payload (Any): Raw payload found in the dataset JSON.

    Returns:
        str: Extracted modified prompt text if available.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                return decoded.get("modified_prompt", payload)
        except json.JSONDecodeError:
            return payload
        return payload
    if isinstance(payload, dict):
        return payload.get("modified_prompt", "")
    return ""


def _load_prompt_lookup(
    raw_results_dir: str, logger, dataset_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Build a lookup table from sample_id (and variations) to their prompts.

    Args:
        raw_results_dir (str): Directory containing Stage 4 raw results.
        logger: Run logger for warning messages.

    Returns:
        Dict[str, str]: Mapping from sample identifiers to prompt text.
    """
    resolved_dir = dataset_dir or _resolve_dataset_dir(raw_results_dir)
    prompt_lookup: Dict[str, str] = {}
    if not os.path.isdir(resolved_dir):
        logger.warning(
            "Dataset directory not found at %s; prompts will be limited.", resolved_dir
        )
        return prompt_lookup

    dataset_files = glob.glob(os.path.join(resolved_dir, "*.json"))
    for path in dataset_files:
        payload = load_json(path)
        original = payload.get("original_sample", {})
        sample_id = original.get("sample_id")
        prompt_text = original.get("prompt", "")
        if sample_id and prompt_text:
            prompt_lookup[sample_id] = prompt_text

        for variation in payload.get("variations", []):
            variation_id = variation.get("variation_id")
            if not variation_id:
                continue
            key = (
                f"{sample_id}::variation::{variation_id}" if sample_id else variation_id
            )
            prompt_text = _unwrap_modified_prompt(variation.get("modified_prompt"))
            if prompt_text:
                prompt_lookup[key] = prompt_text

    return prompt_lookup


def _fallback_input_text(sample_id: str, artifacts: Dict[str, Any]) -> str:
    """
    Create a fallback input text using available metadata when the prompt is missing.

    Args:
        sample_id (str): Identifier of the sample/variation.
        artifacts (Dict[str, Any]): Artifacts block from the result file.

    Returns:
        str: Synthesized instruction text.
    """
    tests = artifacts.get("tests", {})
    entry_point = tests.get("entry_point", "unknown_entry_point")
    base_tests = tests.get("base_tests") or []
    variant_label = artifacts.get("sample_metadata", {}).get("variant_label", "unknown")
    test_str = "; ".join(base_tests[:3])
    return (
        f"Sample {sample_id} ({variant_label}). "
        f"Implement entry point `{entry_point}`. Key tests: {test_str}"
    )


def _extract_pass_flag(result_block: Dict[str, Any]) -> Optional[float]:
    """Convert a result block into a scalar pass indicator."""
    if not result_block:
        return None
    if "passed" in result_block:
        return 1.0 if result_block["passed"] else 0.0
    return None


def _extract_pass_at_k(metric_block: Dict[str, Any]) -> Optional[float]:
    """Pull pass@1 (or the first numeric) from a metric block."""
    if not metric_block:
        return None
    pass_at_k = metric_block.get("pass_at_k", {})
    if not pass_at_k:
        return None
    # Prefer k=1, otherwise take the first available value.
    if "1" in pass_at_k:
        return float(pass_at_k["1"])
    first_val = next(iter(pass_at_k.values()), None)
    return float(first_val) if first_val is not None else None


def _compute_correctness(record: Dict[str, Any], metrics: Dict[str, Any]) -> float:
    """Combine per-attempt execution signals into a correctness scalar."""
    values: List[float] = []
    for key in ("base_result", "plus_result"):
        val = _extract_pass_flag(record.get(key, {}))
        if val is not None:
            values.append(val)

    if values:
        return sum(values) / len(values)

    for key in ("base", "plus"):
        val = _extract_pass_at_k(metrics.get(key, {}))
        if val is not None:
            values.append(val)

    return sum(values) / len(values) if values else 0.0


def _build_accuracy_metrics(
    record: Dict[str, Any], metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble accuracy metrics payload for downstream analysis."""
    accuracy = {
        "base_pass": record.get("base_result", {}).get("passed"),
        "plus_pass": record.get("plus_result", {}).get("passed"),
        "base_pass_at_1": _extract_pass_at_k(metrics.get("base", {})),
        "plus_pass_at_1": _extract_pass_at_k(metrics.get("plus", {})),
    }
    return accuracy


def _infer_error_types(record: Dict[str, Any]) -> List[str]:
    """Derive coarse error labels from execution results."""
    errors: List[str] = []
    if record.get("base_result") and not record["base_result"].get("passed", True):
        errors.append("base_failure")
    if record.get("plus_result") and not record["plus_result"].get("passed", True):
        errors.append("plus_failure")
    return errors


def _extract_generations_from_result(
    data: Dict[str, Any],
    user_profile: UserProfile,
    prompt_lookup: Dict[str, str],
    model_name: str,
    logger,
) -> Tuple[List[ModelGeneration], List[StaticEvaluationResult]]:
    """
    Convert a Stage-4 result JSON payload into ModelGeneration/StaticEvaluationResult lists.
    """
    sample_id = data.get("sample_id")
    artifacts = data.get("artifacts", {})
    records = artifacts.get("records", [])
    metrics = data.get("metrics", {}) or {}
    if bool(metrics.get("verification_failed", False)):
        # Verification-gated units are expected and should not be treated as missing data.
        return [], []
    if not sample_id or not records:
        logger.warning("Skipping sample with missing sample_id or records.")
        return [], []

    prompt_text = prompt_lookup.get(sample_id) or _fallback_input_text(
        sample_id, artifacts
    )
    shared_metadata = {
        "metrics": metrics,
        "tests": artifacts.get("tests", {}),
        "sample_metadata": artifacts.get("sample_metadata", {}),
    }

    generations: List[ModelGeneration] = []
    static_results: List[StaticEvaluationResult] = []

    multiple_attempts = len(records) > 1
    for idx, record in enumerate(records):
        task_id = f"{sample_id}::attempt::{idx}" if multiple_attempts else sample_id
        generated_output = record.get("raw_output") or record.get("sanitized_code", "")
        if not generated_output:
            logger.warning(
                "Sample %s attempt %s missing generated output.", sample_id, idx
            )
            continue

        metadata = {
            **shared_metadata,
            "record_index": idx,
            "sanitized_code": record.get("sanitized_code"),
        }

        mg = ModelGeneration(
            user_id=user_profile.user_id,
            task_id=task_id,
            input_text=prompt_text,
            generated_output=generated_output,
            model_name=model_name,
            metadata=metadata,
        )

        correctness = _compute_correctness(record, metrics)
        sr = StaticEvaluationResult(
            user_id=user_profile.user_id,
            task_id=task_id,
            correctness_score=correctness,
            accuracy_metrics=_build_accuracy_metrics(record, metrics),
            error_types=_infer_error_types(record),
            metadata=metadata,
        )

        generations.append(mg)
        static_results.append(sr)

    return generations, static_results


def main(args: Optional[List[str]] = None, judge_model: Optional[BaseModel] = None):
    """Main execution function for subjective evaluation."""
    parser = argparse.ArgumentParser(description="Stage 5: Subjective Vibe Evaluation")
    parser.add_argument(
        "--user-profile",
        type=str,
        required=True,
        help="Path to the user profile JSON file from Stage 1.",
    )
    parser.add_argument(
        "--raw-results",
        type=str,
        required=True,
        help="Path to the directory containing raw model outputs from Stage 4.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Optional override path for the subjective evaluation results (JSON).",
    )
    parser.add_argument(
        "--vibe-dataset-dir",
        type=str,
        required=False,
        help="Optional override path to the Stage 3 vibe dataset (defaults to canonical run directory).",
    )
    parser.add_argument(
        "--judge-model-config",
        type=str,
        required=True,
        help="Config for the model to use for subjective judging.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for judge model calls during subjective evaluation.",
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=["original", "personalized", "control"],
        help=(
            "Prompt types to include in subjective evaluation. When omitted, all prompt "
            "types present in the raw results are evaluated together using the legacy "
            "directory layout. When provided, exactly one prompt type must be specified "
            "and outputs are written under a prompt_type_<name>/ subdirectory."
        ),
    )
    parser.add_argument(
        "--reference-results-dir",
        type=str,
        help="Optional directory containing reference Stage 4 results (e.g., original/control from another persona).",
    )
    add_common_args(parser)
    parsed_args = parser.parse_args(args)

    seed_everything(parsed_args.seed)

    run_context = resolve_run_context("stage_5_subjective_vibe_evaluation", parsed_args)
    prompt_types = parsed_args.prompt_types
    if prompt_types is not None and len(prompt_types) != 1:
        raise SystemExit(
            "Stage 5 currently supports exactly one --prompt-types value per run. "
            "Please invoke the stage separately for each prompt type."
        )
    active_prompt_type = prompt_types[0] if prompt_types else None
    log_path = run_context.logs / "stage_5_subjective_eval.log"
    logger = setup_logger(
        str(log_path),
        logger_name=get_stage_logger_name("stage_5_subjective_vibe_evaluation"),
    )
    logger.info("Starting Stage 5: Subjective Vibe Evaluation")
    logger.info("Run root resolved to %s", run_context.root)
    generator_model = parsed_args.generator_model_name or parsed_args.model_name
    filter_model = parsed_args.filter_model_name or "none"
    evaluated_model = parsed_args.evaluated_model_name or parsed_args.model_name
    judge_model_label = parsed_args.judge_model_name or evaluated_model
    subjective_dir = subjective_stage_dir(
        parsed_args.run_base_dir,
        parsed_args.user_group,
        evaluated_model,
        judge_model_label,
        generator_model,
        filter_model,
        prompt_type=active_prompt_type,
    )
    subjective_dir.mkdir(parents=True, exist_ok=True)
    default_dataset_dir = (
        vibe_dataset_stage_dir(
            parsed_args.run_base_dir,
            parsed_args.user_group,
            generator_model,
            filter_model,
        )
        / f"dataset_{run_context.run_id}"
    )

    # --- Input Validation ---
    if not os.path.exists(parsed_args.user_profile):
        logger.error(f"User profile not found: {parsed_args.user_profile}")
        sys.exit(1)
    if not os.path.isdir(parsed_args.raw_results):
        logger.error(f"Raw results directory not found: {parsed_args.raw_results}")
        sys.exit(1)
    if judge_model is None and not os.path.exists(parsed_args.judge_model_config):
        logger.error(f"Judge model config not found: {parsed_args.judge_model_config}")
        sys.exit(1)

    if parsed_args.dry_run:
        logger.info("--- Dry Run Mode ---")
        return

    # --- Load Data ---
    logger.info("Loading user profile...")
    profile_data = load_json(parsed_args.user_profile)
    # Adapt profile data to schema
    if isinstance(profile_data, list):
        if len(profile_data) > 0:
            profile_data = profile_data[0]
        else:
            logger.error("Empty user profile list")
            sys.exit(1)

    # Instantiate the unified UserProfile. The Pydantic model handles
    # mapping from profiling output to evaluation fields automatically.
    user_profile = UserProfile(**profile_data)

    # Injected persona type from the user group for rubric-based scoring
    if not user_profile.persona.get("type"):
        user_profile.persona["type"] = parsed_args.user_group

    # Validate that persona-related content is not empty
    if not user_profile.persona.get("description") and not user_profile.context:
        logger.error(
            f"Persona-related content is empty for user {user_profile.user_id}. "
            "Check Stage 1 profiling output."
        )
        raise ValueError(f"Empty persona content for user {user_profile.user_id}")

    logger.info("Loading raw results...")
    dataset_dir_override = (
        parsed_args.vibe_dataset_dir
        if parsed_args.vibe_dataset_dir
        else str(default_dataset_dir)
    )
    prompt_lookup = _load_prompt_lookup(
        parsed_args.raw_results, logger, dataset_dir=dataset_dir_override
    )
    # Search recursively for JSON files in raw_results and subdirectories (e.g., function_eval/)
    result_files = sorted(Path(parsed_args.raw_results).rglob("*.json"))

    if parsed_args.reference_results_dir:
        logger.info(
            "Loading reference results from %s", parsed_args.reference_results_dir
        )
        ref_files = sorted(Path(parsed_args.reference_results_dir).rglob("*.json"))
        # We append reference files at the end, but we will deduplicate by sample_id
        # prioritizing the first one found.
        result_files.extend(ref_files)

    # Convert Path objects to strings for compatibility
    result_files = [str(f) for f in result_files]
    model_generations: List[ModelGeneration] = []
    static_results: List[StaticEvaluationResult] = []

    # Track seen sample IDs to avoid duplicates when loading from multiple directories
    seen_sample_ids = set()
    verification_failed_units = 0

    for fpath in result_files:
        data = load_json(fpath)
        if not isinstance(data, dict) or "artifacts" not in data:
            # Skip aggregate metrics or unrecognized files.
            logger.debug(
                "Skipping %s; no per-sample artifacts found.", os.path.basename(fpath)
            )
            continue

        sample_id = data.get("sample_id", "")
        if sample_id in seen_sample_ids:
            logger.debug("Skipping duplicate result for sample %s", sample_id)
            continue
        if bool((data.get("metrics", {}) or {}).get("verification_failed", False)):
            verification_failed_units += 1
            seen_sample_ids.add(sample_id)
            continue

        # Extract model name from the canonical Stage 4 path:
        # .../4_objective_evaluation/evaluated_model_<name>/gen_model_.../filter_model_.../...
        try:
            # Try to find evaluated_model_<name> in the path
            parts = fpath.split("/")
            evaluated_model = None
            for part in parts:
                if part.startswith("evaluated_model_"):
                    evaluated_model = part.replace("evaluated_model_", "")
                    break
            if not evaluated_model:
                # Fallback to the one from parsed_args
                evaluated_model = (
                    parsed_args.evaluated_model_name or parsed_args.model_name
                )
        except Exception:
            evaluated_model = parsed_args.evaluated_model_name or parsed_args.model_name

        model_name = evaluated_model

        # Filter by prompt type when requested. We infer the prompt type from
        # the sample identifier and metadata written by Stage 4.
        inferred_type = infer_prompt_type(sample_id)

        if active_prompt_type and inferred_type != active_prompt_type:
            continue

        gens, stats = _extract_generations_from_result(
            data,
            user_profile,
            prompt_lookup,
            model_name,
            logger,
        )
        if gens:
            model_generations.extend(gens)
            static_results.extend(stats)
            seen_sample_ids.add(sample_id)

    logger.info(
        f"Loaded {len(model_generations)} generations from {len(result_files)} files."
    )
    if verification_failed_units:
        logger.info(
            "Excluded %d verification-gated unit(s) (failed_verification) from subjective evaluation.",
            verification_failed_units,
        )
    if not model_generations:
        logger.error(
            "No per-sample generations found for subjective evaluation. "
            "raw_results=%s prompt_type=%s (reference_results_dir=%s). "
            "Verify Stage 4 produced per-sample JSON artifacts under the provided directory.",
            parsed_args.raw_results,
            active_prompt_type or "all",
            parsed_args.reference_results_dir or "none",
        )
        if verification_failed_units:
            logger.error(
                "All candidate units appear to have been excluded due to failed Stage-3 verification. "
                "verification_failed_units=%d",
                verification_failed_units,
            )
        raise SystemExit(1)

    # --- Load Judge Model ---
    if judge_model is None:
        logger.info(
            f"Loading judge model from config: {parsed_args.judge_model_config}"
        )
        judge_model = load_model_from_config(parsed_args.judge_model_config)

    # --- Run Evaluation ---
    # Propagate the run seed into the subjective evaluator so that any
    # API-backed judge model can use deterministic sampling via its
    # native `seed` parameter.
    evaluator = SubjectiveVibeEvaluator(
        judge_model,
        config={
            "generation_kwargs": {"seed": parsed_args.seed},
            "batch_size": parsed_args.batch_size,
            # Stage 5 is now judge-only: no rubric blending, and indicators are split out to Stage 5c.
            "disable_rubric": True,
            "include_indicators": False,
        },
    )

    # We pass single user profile effectively broadcasting it to all generations
    # Or if generations have distinct user_ids, we should handle that.
    # For now, assume single user session.
    profiles = [user_profile] * len(model_generations)

    logger.info("Running subjective evaluation...")
    results = evaluator.evaluate_batch(profiles, model_generations, static_results)

    # --- Save Results ---
    output_data = [r.model_dump() for r in results]

    for record in output_data:
        record.setdefault("_model_metadata", {})
        record["_model_metadata"].update(
            {
                "role": "subjective_judging",
                "evaluated_model_name": record.get("model_name"),
                "judge_model_name": judge_model_label,
                "judge_model_config_path": parsed_args.judge_model_config,
            }
        )
        # Stage 5 is now judge-only. Ensure indicator payloads are not emitted here.
        record.pop("input_side_indicators", None)
        record.pop("output_side_indicators", None)

    if parsed_args.output_file:
        output_path = Path(parsed_args.output_file)
    else:
        detail = f"model-{run_context.model_name}:persona-{run_context.user_group}"
        output_path = run_context.artifact_path(
            base=subjective_dir,
            artifact_type="subjective_scores",
            evaluation_type="analysis",
            detail=detail,
            version=0,
            ext="json",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_data, str(output_path))
    logger.info("Saved subjective results to %s", output_path)

    print("Stage 5 Complete.")


if __name__ == "__main__":
    main()
