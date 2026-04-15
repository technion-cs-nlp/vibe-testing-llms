"""
Stage 5B: Pairwise Comparison Evaluation

This script performs pairwise comparison of model outputs, having a judge model
determine which response better serves the user persona for each vibe dimension.

Unlike Stage 5 (absolute scoring), this stage compares two models directly,
reducing bias from individual rating tendencies.
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
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.vibe_testing.models.base import BaseModel
from src.vibe_testing.model_names import canonicalize_model_name
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_GENERAL_USER,
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
    uses_shared_general_user_artifacts,
)
from src.vibe_testing.pathing import (
    infer_prompt_type,
    normalize_token,
    pairwise_stage_dir,
    vibe_dataset_stage_dir,
)
from src.vibe_testing.subjective_evaluation import (
    GeneralUserPairwiseJudge,
    PairwiseComparisonInput,
    PairwiseVibeEvaluator,
    PairwiseJudge,
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


def _infer_gen_filter_from_path(
    results_path: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Infer generator and filter model names from a Stage 4 results path.

    Args:
        results_path: Directory path to Stage 4 results.

    Returns:
        Tuple containing (generator_model, filter_model), each optionally None.
    """
    if not results_path:
        return None, None
    try:
        parts = Path(results_path).parts
    except TypeError:
        return None, None

    generator = next(
        (
            segment.split("gen_model_", 1)[1]
            for segment in parts
            if segment.startswith("gen_model_")
        ),
        None,
    )
    filter_model = next(
        (
            segment.split("filter_model_", 1)[1]
            for segment in parts
            if segment.startswith("filter_model_")
        ),
        None,
    )
    return generator, filter_model


def _infer_user_group_from_results_path(results_path: Optional[str]) -> Optional[str]:
    """
    Infer the user group/persona from a Stage 4 results path.

    Args:
        results_path: Directory path to Stage 4 results.

    Returns:
        User group string if detected, otherwise None.
    """
    if not results_path:
        return None
    try:
        parts = Path(results_path).parts
    except TypeError:
        return None

    if "4_objective_evaluation" in parts:
        idx = parts.index("4_objective_evaluation")
        if idx > 0:
            return parts[idx - 1]
    return None


def _load_control_prompt_lookup(
    control_prompts_dir: str,
    logger,
) -> Dict[str, str]:
    """
    Load control prompts keyed by sample_id::variation::control_<n>.

    Args:
        control_prompts_dir: Directory containing control prompt JSON files.
        logger: Logger for status messages.

    Returns:
        Dict mapping control variation keys to prompt text.
    """
    control_prompts: Dict[str, str] = {}
    control_path = Path(control_prompts_dir)
    if not control_path.exists():
        logger.debug("Control prompts directory not found: %s", control_prompts_dir)
        return control_prompts

    benchmark_file_map = {
        "mbppplus_variations.json": "mbpp_plus",
        "humanevalplus_variations.json": "humaneval_plus",
    }

    for filename, _benchmark in benchmark_file_map.items():
        file_path = control_path / filename
        if not file_path.exists():
            continue
        try:
            variations_data = load_json(str(file_path))
            if not isinstance(variations_data, list):
                logger.warning(
                    "Control prompts file %s does not contain a list. Skipping.",
                    file_path,
                )
                continue

            for variation_entry in variations_data:
                original_row_data = variation_entry.get("original_row_data", {})
                task_id = original_row_data.get("task_id")
                if task_id is None:
                    continue
                task_id_str = str(task_id)
                prompt_text = variation_entry.get("prompt", "")
                variation_count = variation_entry.get("variation_count", "unknown")
                if not prompt_text:
                    continue

                key = f"{task_id_str}::variation::control_{variation_count}"
                control_prompts[key] = prompt_text
        except Exception as exc:
            logger.warning("Failed to load control prompts from %s: %s", file_path, exc)
    if control_prompts:
        logger.info(
            "Loaded %d control prompts from %s",
            len(control_prompts),
            control_prompts_dir,
        )
    return control_prompts


def _infer_dataset_dir_from_results(results_dir: Optional[str]) -> Optional[Path]:
    """
    Infer the Stage 3 dataset directory from a Stage 4 results path.

    Expected Stage 4 layout (legacy and prompt_type-aware):
        <base>/<user_group>/4_objective_evaluation/evaluated_model_*/gen_model_*/filter_model_*/[prompt_type_*/]<dataset_type>_<run_id>

    We map that to Stage 3 dataset:
        <base>/<user_group>/3_vibe_dataset/gen_model_*/filter_model_*/dataset_<run_id>

    Args:
        results_dir: Path to Stage 4 results directory.

    Returns:
        Path to the inferred Stage 3 dataset directory, or None if not inferrable.
    """
    if not results_dir:
        return None
    try:
        p = Path(results_dir)
        parts = p.parts
        if "4_objective_evaluation" not in parts:
            return None
        idx = parts.index("4_objective_evaluation")
        if idx == 0:
            return None
        user_group = parts[idx - 1]
        gen_seg = next((seg for seg in parts if seg.startswith("gen_model_")), None)
        filter_seg = next(
            (seg for seg in parts if seg.startswith("filter_model_")), None
        )
        if not gen_seg or not filter_seg:
            return None
        leaf = p.name
        run_id = None
        if "_" in leaf:
            run_id = leaf.split("_", 1)[1]
        if not run_id:
            return None
        base = Path(*parts[: idx - 1]) if idx >= 1 else Path("/")
        return (
            base
            / user_group
            / "3_vibe_dataset"
            / gen_seg
            / filter_seg
            / f"dataset_{run_id}"
        )
    except Exception:
        return None


def _load_model_outputs(
    results_dir: str,
    model_name: str,
    logger,
    reference_results_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load model outputs from a Stage 4 results directory.

    The reference_results_dir is used to backfill samples that don't exist in
    the primary results_dir. This is critical for multi-persona setups where:
    - The reference persona (e.g., novice_user) evaluates all prompt types
      (original, personalized, control)
    - Other personas only evaluate personalized prompts
    - Stages 5b/6 need to load original/control from reference and personalized
      from the current persona

    IMPORTANT: From the reference directory, we ONLY load original and control
    samples. Personalized samples from the reference persona are excluded because
    each persona has their own personalized variations.

    Args:
        results_dir: Path to the objective evaluation results (primary).
        model_name: Name of the evaluated model.
        logger: Logger for status messages.
        reference_results_dir: Optional directory containing reference Stage 4 results.

    Returns:
        Dict mapping sample_id to output data.

    Raises:
        ValueError: If no valid results directories are found.
    """
    outputs: Dict[str, Dict[str, Any]] = {}

    # Collect all result files from both primary and reference directories.
    # We process reference FIRST so that primary results can override them.
    search_dirs = []
    if reference_results_dir and os.path.isdir(reference_results_dir):
        search_dirs.append(("reference", reference_results_dir))
    if os.path.isdir(results_dir):
        search_dirs.append(("primary", results_dir))

    if not search_dirs:
        raise ValueError(
            f"No valid results directories found. "
            f"Primary: '{results_dir}' (exists={os.path.isdir(results_dir)}), "
            f"Reference: '{reference_results_dir}' (exists={os.path.isdir(reference_results_dir) if reference_results_dir else 'not provided'})"
        )

    primary_count = 0
    reference_count = 0
    reference_skipped_personalized = 0
    verification_failed_units = 0

    for source_label, dir_path in search_dirs:
        result_files = list(Path(dir_path).rglob("*.json"))
        logger.debug(
            "Scanning %s directory '%s': found %d JSON files",
            source_label,
            dir_path,
            len(result_files),
        )

        for fpath in result_files:
            try:
                data = load_json(str(fpath))
                if not isinstance(data, dict) or "artifacts" not in data:
                    continue

                sample_id = data.get("sample_id")
                if not sample_id:
                    continue

                # From reference, only load original and control samples.
                # Personalized samples should come from the current persona only.
                if source_label == "reference":
                    prompt_type = infer_prompt_type(sample_id)
                    if prompt_type == "personalized":
                        reference_skipped_personalized += 1
                        continue

                if bool(
                    (data.get("metrics", {}) or {}).get("verification_failed", False)
                ):
                    verification_failed_units += 1
                    continue

                records = data.get("artifacts", {}).get("records", [])
                if not records:
                    continue

                # Use first record's output
                record = records[0]
                generated_output = record.get("raw_output") or record.get(
                    "sanitized_code", ""
                )

                # Primary results override reference results for the same sample_id.
                # Reference results are loaded first, so primary will overwrite them.
                is_new = sample_id not in outputs
                outputs[sample_id] = {
                    "sample_id": sample_id,
                    "generated_output": generated_output,
                    "model_name": model_name,
                    "metadata": data.get("metadata", {}),
                    "_source": source_label,
                }

                if source_label == "primary":
                    primary_count += 1
                elif is_new:
                    reference_count += 1

            except Exception as exc:
                logger.warning("Failed to load %s: %s", fpath, exc)

    if reference_skipped_personalized > 0:
        logger.debug(
            "Skipped %d personalized samples from reference (using current persona's instead)",
            reference_skipped_personalized,
        )

    # Categorize outputs by prompt type
    original_count = sum(1 for k in outputs if infer_prompt_type(k) == "original")
    personalized_count = sum(
        1 for k in outputs if infer_prompt_type(k) == "personalized"
    )
    control_count = sum(1 for k in outputs if infer_prompt_type(k) == "control")

    logger.info(
        "Loaded %d outputs for model %s: %d original, %d personalized, %d control "
        "(primary=%d, reference=%d)",
        len(outputs),
        model_name,
        original_count,
        personalized_count,
        control_count,
        primary_count,
        reference_count,
    )
    if verification_failed_units:
        logger.info(
            "Excluded %d verification-gated unit(s) (failed_verification) while loading outputs for model %s.",
            verification_failed_units,
            model_name,
        )

    if not outputs:
        raise ValueError(
            f"No valid outputs found for model '{model_name}'. "
            f"Checked directories: {[d for _, d in search_dirs]}"
        )

    return outputs


def _load_prompt_lookup(
    dataset_dir: str,
    logger,
) -> Dict[str, str]:
    """
    Build a lookup table from sample_id (and variations) to their prompts.

    This loads both the original sample prompts and all variation prompts,
    using the same key format as Stage 4 results (e.g., "104::variation::104-var1").

    Args:
        dataset_dir: Path to the Stage 3 vibe dataset.
        logger: Logger for status messages.

    Returns:
        Dict mapping sample identifiers to prompt text.
    """
    prompts: Dict[str, str] = {}

    if not os.path.isdir(dataset_dir):
        logger.warning("Dataset directory not found: %s", dataset_dir)
        return prompts

    dataset_files = glob.glob(os.path.join(dataset_dir, "*.json"))
    for fpath in dataset_files:
        try:
            data = load_json(fpath)
            original = data.get("original_sample", {})
            sample_id = original.get("sample_id")
            prompt_text = original.get("prompt", "")

            # Store original sample prompt
            if sample_id and prompt_text:
                prompts[sample_id] = prompt_text

            # Store variation prompts
            for variation in data.get("variations", []):
                variation_id = variation.get("variation_id")
                if not variation_id:
                    continue
                # Use the same key format as Stage 4 results
                key = (
                    f"{sample_id}::variation::{variation_id}"
                    if sample_id
                    else variation_id
                )
                var_prompt = _unwrap_modified_prompt(variation.get("modified_prompt"))
                if var_prompt:
                    prompts[key] = var_prompt

        except Exception as exc:
            logger.warning("Failed to load prompt from %s: %s", fpath, exc)

    original_count = sum(1 for k in prompts if "::" not in k)
    variation_count = len(prompts) - original_count
    logger.info(
        "Loaded %d prompts from dataset (%d original, %d variations)",
        len(prompts),
        original_count,
        variation_count,
    )
    return prompts


def _build_comparison_inputs(
    model_a_outputs: Dict[str, Dict[str, Any]],
    model_b_outputs: Dict[str, Dict[str, Any]],
    prompts: Dict[str, str],
    model_a_name: str,
    model_b_name: str,
    user_profile: UserProfile,
    logger,
) -> List[PairwiseComparisonInput]:
    """
    Build comparison inputs from model outputs.

    Args:
        model_a_outputs: Outputs from model A.
        model_b_outputs: Outputs from model B.
        prompts: Task prompts.
        model_a_name: Name of model A.
        model_b_name: Name of model B.
        user_profile: User profile for metadata.
        logger: Logger for status messages.

    Returns:
        List of PairwiseComparisonInput objects.

    Raises:
        ValueError: If prompts are missing for any common samples.
    """
    inputs: List[PairwiseComparisonInput] = []

    # Find common samples between model A and model B
    common_samples = set(model_a_outputs.keys()) & set(model_b_outputs.keys())
    only_a = set(model_a_outputs.keys()) - set(model_b_outputs.keys())
    only_b = set(model_b_outputs.keys()) - set(model_a_outputs.keys())

    if only_a or only_b:
        logger.warning(
            "Sample mismatch between models. "
            "Only in %s: %d samples. Only in %s: %d samples.",
            model_a_name,
            len(only_a),
            model_b_name,
            len(only_b),
        )
        # Categorize missing samples by prompt type for better diagnostics
        only_a_by_type = {"original": 0, "personalized": 0, "control": 0}
        only_b_by_type = {"original": 0, "personalized": 0, "control": 0}
        for sid in only_a:
            only_a_by_type[infer_prompt_type(sid)] += 1
        for sid in only_b:
            only_b_by_type[infer_prompt_type(sid)] += 1

        if only_a:
            logger.debug(
                "Only in %s: %d samples (original=%d, personalized=%d, control=%d). First 10: %s",
                model_a_name,
                len(only_a),
                only_a_by_type["original"],
                only_a_by_type["personalized"],
                only_a_by_type["control"],
                sorted(only_a)[:10],
            )
        if only_b:
            logger.debug(
                "Only in %s: %d samples (original=%d, personalized=%d, control=%d). First 10: %s",
                model_b_name,
                len(only_b),
                only_b_by_type["original"],
                only_b_by_type["personalized"],
                only_b_by_type["control"],
                sorted(only_b)[:10],
            )

        # Warn if control samples exist in one model but not the other
        if only_a_by_type["control"] > 0 and only_b_by_type["control"] == 0:
            logger.warning(
                "%s has %d control samples but %s has none. "
                "Control samples will be excluded from comparison.",
                model_a_name,
                only_a_by_type["control"],
                model_b_name,
            )
        elif only_b_by_type["control"] > 0 and only_a_by_type["control"] == 0:
            logger.warning(
                "%s has %d control samples but %s has none. "
                "Control samples will be excluded from comparison.",
                model_b_name,
                only_b_by_type["control"],
                model_a_name,
            )

    # Categorize common samples by prompt type
    common_by_type = {"original": [], "personalized": [], "control": []}
    for sid in common_samples:
        prompt_type = infer_prompt_type(sid)
        common_by_type[prompt_type].append(sid)

    logger.info(
        "Common samples: %d total (%d original, %d personalized, %d control)",
        len(common_samples),
        len(common_by_type["original"]),
        len(common_by_type["personalized"]),
        len(common_by_type["control"]),
    )

    missing_prompts: List[str] = []
    for sample_id in sorted(common_samples):
        output_a = model_a_outputs[sample_id]
        output_b = model_b_outputs[sample_id]
        prompt = prompts.get(sample_id, f"[Prompt for {sample_id}]")

        if prompt.startswith("[Prompt for"):
            missing_prompts.append(sample_id)
            continue

        inputs.append(
            PairwiseComparisonInput(
                task_id=sample_id,
                input_text=prompt,
                model_a_name=model_a_name,
                model_a_output=output_a["generated_output"],
                model_b_name=model_b_name,
                model_b_output=output_b["generated_output"],
                metadata={
                    "user_id": user_profile.user_id,
                    "persona_description": user_profile.persona_description,
                },
            )
        )

    if missing_prompts:
        # Group missing prompts by type for better diagnostics
        missing_by_type = {"original": [], "personalized": [], "control": []}
        for sid in missing_prompts:
            prompt_type = infer_prompt_type(sid)
            missing_by_type[prompt_type].append(sid)

        error_details = []
        for ptype, sids in missing_by_type.items():
            if sids:
                error_details.append(f"{ptype}={len(sids)}")

        missing_list = ", ".join(sorted(missing_prompts)[:20])
        if len(missing_prompts) > 20:
            missing_list += f"... ({len(missing_prompts) - 20} more)"

        raise ValueError(
            f"Missing prompts for {len(missing_prompts)} samples ({', '.join(error_details)}): "
            f"{missing_list}. "
            "Ensure the Stage 3 dataset directory matches the Stage 4 results. "
            "For cross-persona runs, ensure the reference persona's dataset is accessible. "
            "Override with --vibe-dataset-dir if needed."
        )

    return inputs


def _build_pairwise_judge(
    pairwise_judgment_type: str,
    judge_model: BaseModel,
    generation_kwargs: Dict[str, Any],
):
    """
    Build the concrete pairwise judge implementation for a run.

    Args:
        pairwise_judgment_type: Normalized judgment type token.
        judge_model: LLM used for pairwise judging.
        generation_kwargs: Generation kwargs forwarded to the judge.

    Returns:
        Concrete pairwise judge instance.
    """
    if pairwise_judgment_type == PAIRWISE_JUDGMENT_TYPE_PERSONA:
        return PairwiseJudge(judge_model, generation_kwargs=generation_kwargs)
    if pairwise_judgment_type == PAIRWISE_JUDGMENT_TYPE_GENERAL_USER:
        return GeneralUserPairwiseJudge(
            judge_model, generation_kwargs=generation_kwargs
        )
    raise ValueError(
        "Unsupported pairwise judgment type "
        f"{pairwise_judgment_type!r} while constructing pairwise judge."
    )


def main(args: Optional[List[str]] = None, judge_model: Optional[BaseModel] = None):
    """Main execution function for pairwise comparison evaluation."""
    parser = argparse.ArgumentParser(
        description="Stage 5B: Pairwise Comparison Evaluation"
    )
    parser.add_argument(
        "--user-profile",
        type=str,
        required=True,
        help="Path to the user profile JSON file from Stage 1.",
    )
    parser.add_argument(
        "--model-a-results",
        type=str,
        required=True,
        help="Path to Stage 4 results directory for model A.",
    )
    parser.add_argument(
        "--model-b-results",
        type=str,
        required=True,
        help="Path to Stage 4 results directory for model B.",
    )
    parser.add_argument(
        "--model-a-reference-results",
        type=str,
        help="Optional reference results for model A.",
    )
    parser.add_argument(
        "--model-b-reference-results",
        type=str,
        help="Optional reference results for model B.",
    )
    parser.add_argument(
        "--model-a-name",
        type=str,
        required=True,
        help="Name of model A.",
    )
    parser.add_argument(
        "--model-b-name",
        type=str,
        required=True,
        help="Name of model B.",
    )
    parser.add_argument(
        "--vibe-dataset-dir",
        type=str,
        required=False,
        help="Path to the Stage 3 vibe dataset for prompt lookup.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Optional override path for the pairwise comparison results (JSON).",
    )
    parser.add_argument(
        "--judge-model-config",
        type=str,
        required=True,
        help="Config for the model to use for pairwise judging.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for pairwise judge calls (number of comparisons per batch).",
    )
    parser.add_argument(
        "--no-position-swap",
        action="store_true",
        default=False,
        help=(
            "Disable position swap for position bias mitigation. "
            "By default, each comparison is run twice with swapped positions "
            "to detect and mitigate position bias. Use this flag to run only once "
            "(faster but potentially biased)."
        ),
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=["original", "personalized", "control"],
        help=(
            "Prompt types to include in pairwise comparison. When omitted, all prompt "
            "types are compared together using the legacy directory layout. When "
            "provided, exactly one prompt type must be specified and outputs are "
            "written under a prompt_type_<name>/ subdirectory."
        ),
    )
    parser.add_argument(
        "--control-prompts-dir",
        type=str,
        default="data/control_prompts",
        help="Directory containing control prompt variations (mbppplus_variations.json, humanevalplus_variations.json).",
    )
    parser.add_argument(
        "--pairwise-judgment-type",
        type=str,
        choices=["persona", "general_user"],
        default="persona",
        help=(
            "Pairwise judgment mode. 'persona' preserves the existing persona-aware "
            "judge. 'general_user' evaluates responses for a general user without "
            "persona-specific context."
        ),
    )
    add_common_args(parser)
    parsed_args = parser.parse_args(args)
    level_name = parsed_args.log_level.upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    seed_everything(parsed_args.seed)

    run_context = resolve_run_context("stage_5b_pairwise_comparison", parsed_args)
    prompt_types = parsed_args.prompt_types
    if prompt_types is not None and len(prompt_types) != 1:
        raise SystemExit(
            "Stage 5B currently supports exactly one --prompt-types value per run. "
            "Please invoke the stage separately for each prompt type."
        )
    active_prompt_type = prompt_types[0] if prompt_types else None
    pairwise_judgment_type = normalize_pairwise_judgment_type(
        parsed_args.pairwise_judgment_type
    )
    if (
        pairwise_judgment_type == PAIRWISE_JUDGMENT_TYPE_GENERAL_USER
        and active_prompt_type is None
    ):
        raise SystemExit(
            "Stage 5B general-user judgments require an explicit single "
            "--prompt-types value so original/control artifacts remain reusable "
            "and separated from persona-specific personalized judgments."
        )
    log_path = run_context.logs / "stage_5b_pairwise_comparison.log"
    logger = setup_logger(
        str(log_path),
        logger_name=get_stage_logger_name("stage_5b_pairwise_comparison"),
    )
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(numeric_level)
    logger.info("Starting Stage 5B: Pairwise Comparison Evaluation")
    logger.info(
        "Comparing: %s vs %s", parsed_args.model_a_name, parsed_args.model_b_name
    )
    logger.info("Pairwise judgment type: %s", pairwise_judgment_type)

    # Derive directory paths
    generator_model = parsed_args.generator_model_name
    filter_model = parsed_args.filter_model_name

    for candidate_path in (
        parsed_args.model_a_results,
        parsed_args.model_b_results,
        parsed_args.model_a_reference_results,
        parsed_args.model_b_reference_results,
    ):
        gen_candidate, filt_candidate = _infer_gen_filter_from_path(candidate_path)
        if not generator_model and gen_candidate:
            generator_model = gen_candidate
        if not filter_model and filt_candidate:
            filter_model = filt_candidate

    generator_model = generator_model or parsed_args.model_name
    filter_model = filter_model or "none"
    logger.info(
        "Using generator_model=%s filter_model=%s for dataset lookup",
        generator_model,
        filter_model,
    )
    judge_model_label = parsed_args.judge_model_name or parsed_args.model_name
    canonical_judge_model_name = canonicalize_model_name(judge_model_label)

    pairwise_dir = pairwise_stage_dir(
        parsed_args.run_base_dir,
        parsed_args.user_group,
        parsed_args.model_a_name,
        parsed_args.model_b_name,
        judge_model_label,
        generator_model,
        filter_model,
        judgment_type=pairwise_judgment_type,
        prompt_type=active_prompt_type,
    )
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    # Build default dataset dir path
    default_dataset_dir = (
        vibe_dataset_stage_dir(
            parsed_args.run_base_dir,
            parsed_args.user_group,
            generator_model,
            filter_model,
        )
        / f"dataset_{run_context.run_id}"
    )

    # Input validation
    if not os.path.exists(parsed_args.user_profile):
        logger.error("User profile not found: %s", parsed_args.user_profile)
        sys.exit(1)
    if not os.path.isdir(parsed_args.model_a_results):
        logger.error("Model A results not found: %s", parsed_args.model_a_results)
        sys.exit(1)
    if not os.path.isdir(parsed_args.model_b_results):
        logger.error("Model B results not found: %s", parsed_args.model_b_results)
        sys.exit(1)
    if judge_model is None and not os.path.exists(parsed_args.judge_model_config):
        logger.error("Judge model config not found: %s", parsed_args.judge_model_config)
        sys.exit(1)

    if parsed_args.dry_run:
        logger.info("--- Dry Run Mode ---")
        return

    # Load user profile
    logger.info("Loading user profile...")
    profile_data = load_json(parsed_args.user_profile)
    if isinstance(profile_data, list):
        profile_data = profile_data[0] if profile_data else {}

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

    # Load prompts
    dataset_dir = parsed_args.vibe_dataset_dir or str(default_dataset_dir)
    prompts: Dict[str, str] = {}

    # Helper to merge prompt lookups without overwriting.
    def merge_prompts(src_dir: Optional[str]) -> None:
        if not src_dir:
            return
        logger.info("Loading prompts from dataset directory: %s", src_dir)
        new_prompts = _load_prompt_lookup(src_dir, logger)
        for key, val in new_prompts.items():
            prompts.setdefault(key, val)

    # 1) Primary (explicit or default) dataset dir.
    merge_prompts(dataset_dir)

    # 2) Dataset dirs inferred from model results (primary and reference) to match actual run_ids.
    for candidate in (
        _infer_dataset_dir_from_results(parsed_args.model_a_results),
        _infer_dataset_dir_from_results(parsed_args.model_b_results),
        _infer_dataset_dir_from_results(parsed_args.model_a_reference_results),
        _infer_dataset_dir_from_results(parsed_args.model_b_reference_results),
    ):
        if candidate:
            merge_prompts(str(candidate))

    # 3) Reference persona fallback (if different persona and no explicit dataset dir).
    reference_user_group = _infer_user_group_from_results_path(
        parsed_args.model_a_reference_results
    ) or _infer_user_group_from_results_path(parsed_args.model_b_reference_results)
    if (
        reference_user_group
        and reference_user_group != parsed_args.user_group
        and not parsed_args.vibe_dataset_dir
    ):
        fallback_dataset_dir = (
            vibe_dataset_stage_dir(
                parsed_args.run_base_dir,
                reference_user_group,
                generator_model,
                filter_model,
            )
            / f"dataset_{run_context.run_id}"
        )
        merge_prompts(str(fallback_dataset_dir))

    # 4) Control prompts (do not overwrite).
    control_prompts_dir = parsed_args.control_prompts_dir
    control_prompts_lookup = _load_control_prompt_lookup(control_prompts_dir, logger)
    for key, val in control_prompts_lookup.items():
        prompts.setdefault(key, val)

    # Summarize loaded prompts by type
    prompt_counts = {"original": 0, "personalized": 0, "control": 0}
    for pid in prompts:
        ptype = infer_prompt_type(pid)
        prompt_counts[ptype] = prompt_counts.get(ptype, 0) + 1

    logger.info(
        "Loaded %d prompts total: %d original, %d personalized, %d control",
        len(prompts),
        prompt_counts["original"],
        prompt_counts["personalized"],
        prompt_counts["control"],
    )

    if not prompts:
        raise ValueError(
            "No prompts could be loaded for pairwise comparison. "
            "Verify the dataset directory or provide --vibe-dataset-dir explicitly."
        )

    # Load model outputs
    model_a_outputs = _load_model_outputs(
        parsed_args.model_a_results,
        parsed_args.model_a_name,
        logger,
        reference_results_dir=parsed_args.model_a_reference_results,
    )
    model_b_outputs = _load_model_outputs(
        parsed_args.model_b_results,
        parsed_args.model_b_name,
        logger,
        reference_results_dir=parsed_args.model_b_reference_results,
    )

    if active_prompt_type:
        filtered_a: Dict[str, Dict[str, Any]] = {}
        filtered_b: Dict[str, Dict[str, Any]] = {}
        for sid, payload in model_a_outputs.items():
            if infer_prompt_type(sid) == active_prompt_type:
                filtered_a[sid] = payload
        for sid, payload in model_b_outputs.items():
            if infer_prompt_type(sid) == active_prompt_type:
                filtered_b[sid] = payload
        model_a_outputs = filtered_a
        model_b_outputs = filtered_b

    # Build comparison inputs
    comparison_inputs = _build_comparison_inputs(
        model_a_outputs,
        model_b_outputs,
        prompts,
        parsed_args.model_a_name,
        parsed_args.model_b_name,
        user_profile,
        logger,
    )

    if not comparison_inputs:
        logger.error("No common samples found between models")
        sys.exit(1)

    # Load judge model
    if judge_model is None:
        logger.info("Loading judge model from: %s", parsed_args.judge_model_config)
        judge_model = load_model_from_config(parsed_args.judge_model_config)
        # change judge model max tokens to a reasonable value
        judge_model.config["max_length"] = 100

    # Run evaluation
    use_position_swap = not parsed_args.no_position_swap
    logger.info(
        "Position swap for bias mitigation: %s",
        "enabled" if use_position_swap else "disabled",
    )

    evaluator = PairwiseVibeEvaluator(
        judge_model,
        config={
            "generation_kwargs": {"seed": parsed_args.seed},
            "use_position_swap": use_position_swap,
            "pairwise_judgment_type": pairwise_judgment_type,
        },
        judge=_build_pairwise_judge(
            pairwise_judgment_type,
            judge_model,
            generation_kwargs={"seed": parsed_args.seed},
        ),
    )

    logger.info("Running pairwise evaluation...")
    results = evaluator.evaluate_batch(
        [user_profile], comparison_inputs, batch_size=parsed_args.batch_size
    )

    # Save results
    output_data = [r.model_dump() for r in results]

    # Add metadata
    for record in output_data:
        record.setdefault("_model_metadata", {})
        record["_model_metadata"].update(
            {
                "role": "pairwise_comparison",
                "model_a_name": parsed_args.model_a_name,
                "model_b_name": parsed_args.model_b_name,
                "judge_model_name": canonical_judge_model_name,
                "raw_judge_model_name": judge_model_label,
                "judge_model_config_path": parsed_args.judge_model_config,
                "position_swap_enabled": use_position_swap,
                "pairwise_judgment_type": pairwise_judgment_type,
                "source_persona": parsed_args.user_group,
                "shared_general_user_artifact": uses_shared_general_user_artifacts(
                    pairwise_judgment_type, active_prompt_type
                ),
                "prompt_type": active_prompt_type,
            }
        )
        record["judge_model_name"] = canonical_judge_model_name
        record["pairwise_judgment_type"] = pairwise_judgment_type

    if parsed_args.output_file:
        output_path = Path(parsed_args.output_file)
    else:
        # Build a detail token from normalized model labels so artifact filenames
        # remain valid even when model labels contain special characters.
        model_a_token = normalize_token(parsed_args.model_a_name)
        model_b_token = normalize_token(parsed_args.model_b_name)
        safe_detail = f"model-{model_a_token}-vs-{model_b_token}"[:50]
        output_path = run_context.artifact_path(
            base=pairwise_dir,
            artifact_type="pairwise-comparison",
            evaluation_type="analysis",
            detail=safe_detail,
            version=0,
            ext="json",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_data, str(output_path))
    logger.info("Saved pairwise results to %s", output_path)

    # Print summary
    if results:
        total_a = sum(r.win_counts["model_a"] for r in results)
        total_b = sum(r.win_counts["model_b"] for r in results)
        total_tie = sum(r.win_counts["tie"] for r in results)
        logger.info(
            "Summary: %s wins=%d, %s wins=%d, ties=%d",
            parsed_args.model_a_name,
            total_a,
            parsed_args.model_b_name,
            total_b,
            total_tie,
        )

    print("Stage 5B Complete.")


if __name__ == "__main__":
    main()
