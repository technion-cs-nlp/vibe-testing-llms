"""
Task execution for experiments.

This module handles the actual execution of experiment tasks by
wrapping calls to run_pipeline.py with the appropriate arguments.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.experiment.config import ExperimentConfig
from scripts.experiment.tasks import Task
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
)
from src.vibe_testing.pathing import (
    normalize_token,
    indicator_stage_dir,
    objective_stage_dir,
    pairwise_stage_dir,
    subjective_stage_dir,
    vibe_dataset_stage_dir,
)
from src.vibe_testing.utils import archive_existing_directory

logger = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    """Raised when a task fails to execute."""

    pass


class ExperimentRunner:
    """
    Executes experiment tasks by calling run_pipeline.py.

    Handles the translation from Task objects to the appropriate
    command-line arguments for each stage.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        dry_run: bool = False,
        verbose: bool = False,
        log_level: str = "INFO",
        seed: int = 42,
        batch_size_override: Optional[int] = None,
        force_dataset: bool = False,
        force_objective: bool = False,
        force_subjective: bool = False,
        force_indicators: bool = False,
        force_pairwise: bool = False,
        indicator_include_rubric: bool = False,
        analysis_pairwise_source: str = "stage5b",
        analysis_pairwise_indicator_score_field: str = "rubric_dimension_scores",
        analysis_output_dir: Optional[str] = None,
        analysis_include_pairwise: bool = False,
        analysis_skip_subjective_figures: bool = False,
        analysis_only_joint_preference_long: bool = False,
        analysis_pairwise_tie_breaker: str = "strict",
        analysis_omit_figure_dimensions: Optional[List[str]] = None,
        analysis_no_figure_titles: bool = False,
        analysis_figure_font_scale: Optional[str] = None,
        analysis_figure_label_size: Optional[str] = None,
        analysis_figure_tick_size: Optional[str] = None,
        analysis_pairwise_dimension_weighted_winner: bool = False,
        analysis_pairwise_dimension_weighted_configs: Optional[List[str]] = None,
        analysis_pairwise_correctness_mode: str = "ignore",
        analysis_pairwise_include_plus_correctness: bool = False,
        enable_llm_code_extraction: bool = False,
        llm_code_extraction_prompt: Optional[str] = None,
        llm_code_extraction_prompt_file: Optional[str] = None,
        llm_code_extraction_generation_kwargs: Optional[str] = None,
        llm_code_extraction_retry_on_timeout: Optional[int] = None,
        llm_code_extraction_retry_timeout: Optional[int] = None,
    ):
        """
        Initialize the runner.

        Args:
            config: Experiment configuration.
            dry_run: If True, print commands instead of executing.
            verbose: If True, show more detailed output.
            log_level: Log level to pass to stage scripts (default: INFO).
            batch_size_override: Optional batch size override. When provided, this
                value is used for all stage invocations that accept --batch-size,
                overriding any per-model or experiment default batch_size.
            force_dataset: If True, archive existing dataset results.
            force_objective: If True, archive existing objective results.
            force_subjective: If True, archive existing subjective results.
            force_pairwise: If True, archive existing pairwise results.
        """
        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose
        self.log_level = log_level
        if int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got: {seed}")
        self.seed = int(seed)
        if batch_size_override is not None and int(batch_size_override) < 1:
            raise ValueError(
                f"batch_size_override must be >= 1 when provided, got: {batch_size_override}"
            )
        self.batch_size_override = (
            int(batch_size_override) if batch_size_override is not None else None
        )
        self.force_dataset = force_dataset
        self.force_objective = force_objective
        self.force_subjective = force_subjective
        self.force_indicators = force_indicators
        self.force_pairwise = force_pairwise
        self.indicator_include_rubric = bool(indicator_include_rubric)
        self.analysis_pairwise_source = analysis_pairwise_source
        self.analysis_pairwise_indicator_score_field = (
            analysis_pairwise_indicator_score_field
        )
        self.analysis_output_dir = analysis_output_dir
        self.analysis_include_pairwise = bool(analysis_include_pairwise)
        self.analysis_skip_subjective_figures = bool(analysis_skip_subjective_figures)
        self.analysis_only_joint_preference_long = bool(
            analysis_only_joint_preference_long
        )
        self.analysis_pairwise_tie_breaker = analysis_pairwise_tie_breaker
        self.analysis_omit_figure_dimensions = analysis_omit_figure_dimensions
        self.analysis_no_figure_titles = bool(analysis_no_figure_titles)
        self.analysis_figure_font_scale = analysis_figure_font_scale
        self.analysis_figure_label_size = analysis_figure_label_size
        self.analysis_figure_tick_size = analysis_figure_tick_size
        self.analysis_pairwise_dimension_weighted_winner = bool(
            analysis_pairwise_dimension_weighted_winner
        )
        self.analysis_pairwise_dimension_weighted_configs = (
            analysis_pairwise_dimension_weighted_configs
        )
        self.analysis_pairwise_correctness_mode = analysis_pairwise_correctness_mode
        self.analysis_pairwise_include_plus_correctness = bool(
            analysis_pairwise_include_plus_correctness
        )
        self.enable_llm_code_extraction = enable_llm_code_extraction
        self.llm_code_extraction_prompt = llm_code_extraction_prompt
        self.llm_code_extraction_prompt_file = llm_code_extraction_prompt_file
        self.llm_code_extraction_generation_kwargs = (
            llm_code_extraction_generation_kwargs
        )
        self.llm_code_extraction_retry_on_timeout = llm_code_extraction_retry_on_timeout
        self.llm_code_extraction_retry_timeout = llm_code_extraction_retry_timeout
        self._scripts_dir = Path(__file__).parent.parent

    def run_tasks(self, tasks: List[Task]) -> dict:
        """
        Execute a list of tasks.

        Args:
            tasks: List of tasks to execute.

        Returns:
            Dict with execution summary (completed, failed, skipped counts).
        """
        results = {"completed": 0, "failed": 0, "skipped": 0}

        for task in tasks:
            try:
                logger.info("[Task %d] %s", task.task_id, task.describe())
                self._run_task(task)
                results["completed"] += 1
                logger.info("[Task %d] Completed successfully", task.task_id)
            except TaskExecutionError as e:
                results["failed"] += 1
                logger.error("[Task %d] Failed: %s", task.task_id, e)
                # Continue with other tasks
            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                raise

        logger.info(
            "Execution complete: %d completed, %d failed, %d skipped",
            results["completed"],
            results["failed"],
            results["skipped"],
        )
        return results

    def _get_batch_size(self, model_cfg: Optional[Dict[str, Any]] = None) -> int:
        """
        Get batch size with fallback logic: override -> model config -> defaults -> 1.

        Args:
            model_cfg: Optional model configuration dict. If provided, checks
                for batch_size in the model config first.

        Returns:
            Batch size to use (defaults to 1 if not specified anywhere).
        """
        if self.batch_size_override is not None:
            return int(self.batch_size_override)

        # First check model-specific batch_size if model config provided
        if model_cfg and "batch_size" in model_cfg:
            return int(model_cfg["batch_size"])

        # Fall back to experiment defaults
        defaults = self.config.defaults
        if "batch_size" in defaults:
            return int(defaults["batch_size"])

        # Final fallback to 1
        return 1

    def _run_task(self, task: Task) -> None:
        """
        Execute a single task.

        Args:
            task: Task to execute.

        Raises:
            TaskExecutionError: If the task fails.
        """
        if task.stage == "dataset":
            self._run_dataset_task(task)
        elif task.stage == "objective":
            self._run_objective_task(task)
        elif task.stage == "subjective":
            self._run_subjective_task(task)
        elif task.stage == "indicators":
            self._run_indicators_task(task)
        elif task.stage == "pairwise":
            self._run_pairwise_task(task)
        elif task.stage == "analyze":
            self._run_analyze_task(task)
        else:
            raise TaskExecutionError(f"Unknown task stage: {task.stage}")

    def _run_indicators_task(self, task: Task) -> None:
        """
        Run Stage 5c indicator scoring for a (persona, model) pair.

        This is deterministic/offline and does not call an LLM judge.
        """
        defaults = self.config.defaults

        # Find the dataset directory
        dataset_dir = self._find_dataset_dir(task.persona, task.prompt_type)
        if not dataset_dir:
            raise TaskExecutionError(
                f"Dataset not found for persona '{task.persona}'. Run dataset generation first."
            )

        # Find objective results (must exist)
        obj_results_path = self._find_objective_results_dir(
            task.persona, task.model, task.prompt_type
        )
        if not obj_results_path:
            raise TaskExecutionError(
                f"Objective results not found for {task.persona}/{task.model}. "
                "Run objective evaluation first."
            )

        run_id = f"eval_{task.model}_{self.config.name}"

        # Archive existing results if forced
        if self.force_indicators and not self.dry_run:
            filter_model = defaults.get("filter_model", "none")
            out_dir = indicator_stage_dir(
                self.config.base_dir,
                task.persona,
                task.model,
                self.config.generator,
                filter_model,
                prompt_type=task.prompt_type,
            )
            archive_existing_directory(out_dir, logger)

        cmd = [
            sys.executable,
            str(self._scripts_dir / "stage_5c_indicator_scores.py"),
            "--user-profile",
            str(self._get_profile_path(task.persona)),
            "--raw-results",
            str(obj_results_path),
            "--vibe-dataset-dir",
            str(dataset_dir),
            "--user-group",
            task.persona,
            "--model-name",
            task.model,
            "--evaluated-model-name",
            task.model,
            "--generator-model-name",
            self.config.generator,
            "--filter-model-name",
            defaults.get("filter_model", "none"),
            "--run-base-dir",
            str(self.config.base_dir),
            "--run-id",
            run_id,
            "--log-level",
            self.log_level,
            "--seed",
            str(self.seed),
        ]
        if task.prompt_type:
            cmd.extend(["--prompt-types", task.prompt_type])
        if self.indicator_include_rubric:
            cmd.append("--include-rubric")

        label = f"indicators:{task.persona}/{task.model}"
        if task.prompt_type:
            label += f"/prompt_type={task.prompt_type}"
        self._execute_command(cmd, label)

    def _run_dataset_task(self, task: Task) -> None:
        """
        Generate dataset for a persona (stages 1-3).

        Args:
            task: Dataset generation task.
        """
        persona_cfg = self.config.get_persona_config(task.persona)
        gen_cfg = self.config.get_generator_config()
        defaults = self.config.defaults
        batch_size = self._get_batch_size(gen_cfg)

        # Build the dataset ID for this experiment
        dataset_id = self.config.get_dataset_id(task.persona)

        # Archive existing results if forced
        if self.force_dataset and not self.dry_run:
            filter_model = defaults.get("filter_model", "none")
            dataset_dir = vibe_dataset_stage_dir(
                self.config.base_dir,
                task.persona,
                self.config.generator,
                filter_model,
            )
            # The actual dataset files are inside a dataset_<run_id> subdirectory
            # But vibe_dataset_stage_dir returns the parent.
            # We want to archive the specific dataset run if possible.
            specific_dir = dataset_dir / f"dataset_{dataset_id}"
            if specific_dir.exists():
                archive_existing_directory(specific_dir, logger)
            elif dataset_dir.exists():
                archive_existing_directory(dataset_dir, logger)

        cmd = [
            sys.executable,
            str(self._scripts_dir / "run_pipeline.py"),
            "--stages",
            "1",
            "2",
            "3",
            "--run-mode",
            "direct",
            "--user-group",
            task.persona,
            "--user-config",
            persona_cfg["config"],
            "--model-name",
            self.config.generator,
            "--generator-model-name",
            self.config.generator,
            "--judge-model-config",
            gen_cfg["config"],
            "--run-base-dir",
            str(self.config.base_dir),
            "--run-id",
            dataset_id,
            "--benchmarks",
            *defaults["benchmarks"],
            "--num-samples",
            str(defaults["num_samples"]),
            "--num-variations",
            str(defaults["num_variations"]),
            "--filter-model-name",
            defaults.get("filter_model", "none"),
            "--log-level",
            self.log_level,
            "--seed",
            str(self.seed),
            "--batch-size",
            str(batch_size),
        ]

        self._execute_command(cmd, f"dataset:{task.persona}")

    def _run_objective_task(self, task: Task) -> None:
        """
        Run objective evaluation for a (persona, model) pair (stage 4).

        Args:
            task: Objective evaluation task.
        """
        persona_cfg = self.config.get_persona_config(task.persona)
        model_cfg = self.config.get_model_config(task.model)
        gen_cfg = self.config.get_generator_config()
        defaults = self.config.defaults
        batch_size = self._get_batch_size(model_cfg)
        # Find the dataset directory
        dataset_dir = self._find_dataset_dir(task.persona, task.prompt_type)
        if not dataset_dir:
            raise TaskExecutionError(
                f"Dataset not found for persona '{task.persona}'. "
                "Run dataset generation first."
            )

        # Build base run ID for this evaluation. When multiple prompt types are
        # requested, they will share this run_id but write to distinct
        # prompt_type_<name>/ subdirectories.
        run_id = f"eval_{task.model}_{self.config.name}"

        # Archive existing results if forced
        if self.force_objective and not self.dry_run:
            generator_model = self.config.generator
            filter_model = defaults.get("filter_model", "none")
            obj_dir = objective_stage_dir(
                self.config.base_dir,
                task.persona,
                task.model,
                generator_model,
                filter_model,
                prompt_type=task.prompt_type,
            )
            # Similar to dataset, the actual files are in a subdirectory.
            # objective_stage_dir handles prompt_type, but not the run_id part.
            # However, Stage 4 typically writes into obj_dir directly or via run_id subdirs.
            # Looking at objective_stage_dir: it returns the base path for that model/prompt_type.
            archive_existing_directory(obj_dir, logger)

        cmd = [
            sys.executable,
            str(self._scripts_dir / "run_pipeline.py"),
            "--stages",
            "4",
            "--run-mode",
            "direct",
            "--user-group",
            task.persona,
            "--user-config",
            persona_cfg["config"],
            "--vibe-dataset-dir",
            str(dataset_dir),
            "--model-name",
            task.model,
            "--model-config",
            model_cfg["config"],
            "--evaluated-model-name",
            task.model,
            "--generator-model-name",
            self.config.generator,
            "--filter-model-name",
            defaults.get("filter_model", "none"),
            "--run-base-dir",
            str(self.config.base_dir),
            "--run-id",
            run_id,
            "--judge-model-config",
            gen_cfg["config"],
            "--num-samples",
            str(defaults["num_samples"]),
            "--dataset-type",
            defaults.get("dataset_type", "function"),
            "--log-level",
            self.log_level,
            "--seed",
            str(self.seed),
            "--batch-size",
            str(batch_size),
        ]

        if self.enable_llm_code_extraction:
            cmd.append("--enable-llm-code-extraction")
            if self.llm_code_extraction_prompt_file:
                cmd.extend(
                    [
                        "--llm-code-extraction-prompt-file",
                        self.llm_code_extraction_prompt_file,
                    ]
                )
            elif self.llm_code_extraction_prompt:
                cmd.extend(
                    [
                        "--llm-code-extraction-prompt",
                        self.llm_code_extraction_prompt,
                    ]
                )
            if self.llm_code_extraction_generation_kwargs:
                cmd.extend(
                    [
                        "--llm-code-extraction-generation-kwargs",
                        self.llm_code_extraction_generation_kwargs,
                    ]
                )
            if self.llm_code_extraction_retry_on_timeout is not None:
                cmd.extend(
                    [
                        "--llm-code-extraction-retry-on-timeout",
                        str(self.llm_code_extraction_retry_on_timeout),
                    ]
                )
            if self.llm_code_extraction_retry_timeout is not None:
                cmd.extend(
                    [
                        "--llm-code-extraction-retry-timeout",
                        str(self.llm_code_extraction_retry_timeout),
                    ]
                )

        # Pass reference persona to allow skipping original/control evaluation
        # Use novice_user as reference if present, otherwise fallback to the first persona
        reference_persona = (
            "novice_user"
            if "novice_user" in self.config.use_personas
            else (self.config.use_personas[0] if self.config.use_personas else None)
        )
        if reference_persona:
            cmd.extend(["--reference-persona", reference_persona])

        if task.prompt_type:
            cmd.extend(["--prompt-types", task.prompt_type])

        label = f"objective:{task.persona}/{task.model}"
        if task.prompt_type:
            label += f"/prompt_type={task.prompt_type}"
        self._execute_command(cmd, label)

    def _run_subjective_task(self, task: Task) -> None:
        """
        Run subjective evaluation for a (persona, model, judge) triple (stage 5).

        Args:
            task: Subjective evaluation task.
        """
        persona_cfg = self.config.get_persona_config(task.persona)
        model_cfg = self.config.get_model_config(task.model)
        judge_cfg = self.config.get_model_config(task.judge)
        gen_cfg = self.config.get_generator_config()
        defaults = self.config.defaults
        # Use judge model batch_size for subjective evaluation (judge does the work)
        batch_size = self._get_batch_size(judge_cfg)
        # Find the dataset directory
        dataset_dir = self._find_dataset_dir(task.persona, task.prompt_type)
        if not dataset_dir:
            raise TaskExecutionError(
                f"Dataset not found for persona '{task.persona}'. "
                "Run dataset generation first."
            )

        # Find objective results (must exist)
        obj_results_path = self._find_objective_results_dir(
            task.persona, task.model, task.prompt_type
        )
        if not obj_results_path:
            raise TaskExecutionError(
                f"Objective results not found for {task.persona}/{task.model}. "
                "Run objective evaluation first."
            )

        # Build run ID matching the objective evaluation
        run_id = f"eval_{task.model}_{self.config.name}"

        # Archive existing results if forced
        if self.force_subjective and not self.dry_run:
            generator_model = self.config.generator
            filter_model = self.config.defaults.get("filter_model", "none")
            subj_dir = subjective_stage_dir(
                self.config.base_dir,
                task.persona,
                task.model,
                task.judge,
                generator_model,
                filter_model,
                prompt_type=task.prompt_type,
            )
            archive_existing_directory(subj_dir, logger)

        cmd = [
            sys.executable,
            str(self._scripts_dir / "run_pipeline.py"),
            "--stages",
            "5",
            "--run-mode",
            "direct",
            "--user-group",
            task.persona,
            "--user-config",
            persona_cfg["config"],
            "--vibe-dataset-dir",
            str(dataset_dir),
            "--model-name",
            task.model,
            "--model-config",
            model_cfg["config"],
            "--evaluated-model-name",
            task.model,
            "--generator-model-name",
            self.config.generator,
            "--filter-model-name",
            defaults.get("filter_model", "none"),
            "--judge-model-name",
            task.judge,
            "--judge-model-config",
            judge_cfg["config"],
            "--run-base-dir",
            str(self.config.base_dir),
            "--run-id",
            run_id,
            "--num-samples",
            str(defaults["num_samples"]),
            "--dataset-type",
            defaults.get("dataset_type", "function"),
            "--log-level",
            self.log_level,
            "--seed",
            str(self.seed),
            "--batch-size",
            str(batch_size),
            "--objective-results-dir",
            str(obj_results_path),
        ]

        # Pass reference results directory if needed
        # Use novice_user as reference if present, otherwise fallback to the first persona
        reference_persona = (
            "novice_user"
            if "novice_user" in self.config.use_personas
            else (self.config.use_personas[0] if self.config.use_personas else None)
        )
        if reference_persona and task.persona != reference_persona:
            ref_results_path = self._find_objective_results_dir(
                reference_persona, task.model, task.prompt_type
            )
            if ref_results_path:
                cmd.extend(["--reference-results-dir", str(ref_results_path)])

        if task.prompt_type:
            cmd.extend(["--prompt-types", task.prompt_type])

        label = f"subjective:{task.persona}/{task.model}/judge={task.judge}"
        if task.prompt_type:
            label += f"/prompt_type={task.prompt_type}"
        self._execute_command(cmd, label)

    def _run_pairwise_task(self, task: Task) -> None:
        """
        Run pairwise comparison for a (persona, model_pair, judge) triple (stage 5b).

        Args:
            task: Pairwise comparison task.
        """
        # Parse model pair from task.model (e.g., "gpt4_vs_gpt35")
        if not task.model or "_vs_" not in task.model:
            raise TaskExecutionError(
                f"Invalid model pair format: {task.model}. Expected 'model_a_vs_model_b'"
            )
        model_a, model_b = task.model.split("_vs_", 1)

        persona_cfg = self.config.get_persona_config(task.persona)
        judge_cfg = self.config.get_model_config(task.judge)
        defaults = self.config.defaults
        # Use judge model batch_size for pairwise evaluation (judge does the work)
        batch_size = self._get_batch_size(judge_cfg)
        pairwise_judgment_type = (
            task.pairwise_judgment_type or PAIRWISE_JUDGMENT_TYPE_PERSONA
        )

        # Find the dataset directory
        dataset_dir = self._find_dataset_dir(task.persona, task.prompt_type)
        if not dataset_dir:
            raise TaskExecutionError(
                f"Dataset not found for persona '{task.persona}'. "
                "Run dataset generation first."
            )

        # Find objective results directories for both models
        model_a_results = self._find_objective_results_dir(
            task.persona, model_a, task.prompt_type
        )
        model_b_results = self._find_objective_results_dir(
            task.persona, model_b, task.prompt_type
        )

        if not model_a_results:
            raise TaskExecutionError(
                f"Objective results not found for {task.persona}/{model_a}. "
                "Run objective evaluation first."
            )
        if not model_b_results:
            raise TaskExecutionError(
                f"Objective results not found for {task.persona}/{model_b}. "
                "Run objective evaluation first."
            )

        # Build run ID for this evaluation
        run_id = f"pairwise_{model_a}_vs_{model_b}_{self.config.name}"

        # Archive existing results if forced
        if self.force_pairwise and not self.dry_run:
            generator_model = self.config.generator
            filter_model = self.config.defaults.get("filter_model", "none")
            pair_dir = pairwise_stage_dir(
                self.config.base_dir,
                task.persona,
                model_a,
                model_b,
                task.judge,
                generator_model,
                filter_model,
                judgment_type=pairwise_judgment_type,
                prompt_type=task.prompt_type,
            )
            archive_existing_directory(pair_dir, logger)

        cmd = [
            sys.executable,
            str(self._scripts_dir / "stage_5b_pairwise_comparison.py"),
            "--user-group",
            task.persona,
            "--user-profile",
            str(self._get_profile_path(task.persona)),
            "--model-a-name",
            model_a,
            "--model-b-name",
            model_b,
            "--model-a-results",
            str(model_a_results),
            "--model-b-results",
            str(model_b_results),
            "--vibe-dataset-dir",
            str(dataset_dir),
            "--judge-model-config",
            judge_cfg["config"],
            "--judge-model-name",
            task.judge,
            "--generator-model-name",
            self.config.generator,
            "--filter-model-name",
            defaults.get("filter_model", "none"),
            "--run-base-dir",
            str(self.config.base_dir),
            "--run-id",
            run_id,
            "--model-name",
            f"{model_a}_vs_{model_b}",
            "--log-level",
            self.log_level,
            "--seed",
            str(self.seed),
            "--batch-size",
            str(batch_size),
            "--pairwise-judgment-type",
            pairwise_judgment_type,
        ]

        # Pass reference results directories if needed
        # Use novice_user as reference if present, otherwise fallback to the first persona
        reference_persona = (
            "novice_user"
            if "novice_user" in self.config.use_personas
            else (self.config.use_personas[0] if self.config.use_personas else None)
        )
        if reference_persona and task.persona != reference_persona:
            ref_a_results = self._find_objective_results_dir(
                reference_persona, model_a, task.prompt_type
            )
            ref_b_results = self._find_objective_results_dir(
                reference_persona, model_b, task.prompt_type
            )
            if ref_a_results:
                cmd.extend(["--model-a-reference-results", str(ref_a_results)])
            if ref_b_results:
                cmd.extend(["--model-b-reference-results", str(ref_b_results)])

        if task.prompt_type:
            cmd.extend(["--prompt-types", task.prompt_type])

        label = (
            f"pairwise:{task.persona}/{model_a}_vs_{model_b}"
            f"/judge={task.judge}/judgment_type={pairwise_judgment_type}"
        )
        if task.prompt_type:
            label += f"/prompt_type={task.prompt_type}"
        self._execute_command(cmd, label)

    def _find_objective_results_dir(
        self, persona: str, model: str, prompt_type: Optional[str] = None
    ) -> Optional[Path]:
        """
        Find the objective results directory for a (persona, model) pair.

        Args:
            persona: Persona name.
            model: Model name.
            prompt_type: Optional prompt type filter.

        Returns:
            Path to results directory, or None if not found.
        """
        filter_model = self.config.defaults.get("filter_model", "none")

        personas_to_check = [persona]
        # OPTIMIZATION: If prompt_type is original or control, also check the reference persona
        # Use novice_user as reference if present, otherwise fallback to the first persona
        reference_persona = (
            "novice_user"
            if "novice_user" in self.config.use_personas
            else (self.config.use_personas[0] if self.config.use_personas else None)
        )
        if (
            prompt_type in ["original", "control"]
            and reference_persona
            and persona != reference_persona
        ):
            personas_to_check.append(reference_persona)

        for check_persona in personas_to_check:
            obj_base = (
                self.config.base_dir
                / check_persona
                / "4_objective_evaluation"
                / f"evaluated_model_{model}"
                / f"gen_model_{self.config.generator}"
                / f"filter_model_{filter_model}"
            )

            if not obj_base.exists():
                continue

            candidate_roots: List[Path] = []
            if prompt_type:
                normalized_prompt = normalize_token(prompt_type)
                candidate_roots.append(obj_base / f"prompt_type_{normalized_prompt}")
            else:
                candidate_roots.append(obj_base)

            # Ensure we don't duplicate roots
            seen_roots = set()
            for root in candidate_roots:
                if not root.exists() or root in seen_roots:
                    continue
                seen_roots.add(root)
                for run_dir in sorted(root.iterdir(), reverse=True):
                    if run_dir.is_dir():
                        if any(run_dir.glob("*.json")) or any(
                            run_dir.glob("**/*.json")
                        ):
                            return run_dir
                if self._has_json_artifacts(root):
                    return root

        return None

    def _has_json_artifacts(self, directory: Path) -> bool:
        """
        Check whether a directory directly contains JSON artifacts.
        """
        return any(directory.glob("*.json")) or any(directory.glob("**/*.json"))

    def _get_profile_path(self, persona: str) -> Path:
        """Get the path to a persona's profile file."""
        return (
            self.config.base_dir
            / persona
            / "1_profile"
            / f"user_profile_profiling_persona-{persona}_v00.json"
        )

    def _run_analyze_task(self, task: Task) -> None:
        """
        Run analysis across all personas (stage 6).

        Args:
            task: Analysis task.
        """
        personas = task.persona.split(",")
        gen_cfg = self.config.get_generator_config()

        # Collect profile paths for all personas
        profile_paths = []
        for persona in personas:
            persona_cfg = self.config.get_persona_config(persona)
            # The profile is generated in stage 1 under:
            # {base_dir}/{persona}/1_profile/user_profile_profiling_persona-{persona}_v00.json
            profile_path = (
                self.config.base_dir
                / persona
                / "1_profile"
                / f"user_profile_profiling_persona-{persona}_v00.json"
            )
            if profile_path.exists():
                profile_paths.append(str(profile_path))
            else:
                logger.warning("Profile not found for %s: %s", persona, profile_path)

        if not profile_paths:
            raise TaskExecutionError("No user profiles found for analysis")

        # Build analysis output directory
        from datetime import datetime

        if self.analysis_output_dir:
            analysis_dir = Path(self.analysis_output_dir)
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            analysis_dir = self.config.base_dir / f"final_analysis_{date_str}"

        # Identify reference persona
        # Use novice_user as reference if present, otherwise fallback to the first persona
        reference_persona = (
            "novice_user"
            if "novice_user" in self.config.use_personas
            else (self.config.use_personas[0] if self.config.use_personas else None)
        )

        cmd = [
            sys.executable,
            str(self._scripts_dir / "stage_6_analyze_results.py"),
            "--results-dir",
            str(self.config.base_dir),
            "--personas",
            *personas,
            "--user-profiles",
            *profile_paths,
            "--analysis-output-dir",
            str(analysis_dir),
            "--run-base-dir",
            str(self.config.base_dir),
            "--user-group",
            "comparative_personas" if len(personas) > 1 else personas[0],
            "--model-name",
            "multi_model_batch",
            "--judge-model-config",
            gen_cfg["config"],
            "--seed",
            str(self.seed),
        ]

        if reference_persona:
            cmd.extend(["--reference-persona", reference_persona])

        if self.analysis_include_pairwise:
            cmd.append("--include-pairwise")
        if self.analysis_skip_subjective_figures:
            cmd.append("--skip-subjective-figures")
        if self.analysis_only_joint_preference_long:
            cmd.append("--only-joint-preference-long")
        if self.analysis_pairwise_tie_breaker:
            cmd.extend(["--pairwise-tie-breaker", self.analysis_pairwise_tie_breaker])
        if self.analysis_omit_figure_dimensions:
            cmd.extend(
                ["--omit-figure-dimensions", *self.analysis_omit_figure_dimensions]
            )
        if self.analysis_no_figure_titles:
            cmd.append("--no-figure-titles")
        if self.analysis_figure_font_scale is not None:
            cmd.extend(["--figure-font-scale", str(self.analysis_figure_font_scale)])
        if self.analysis_figure_label_size is not None:
            cmd.extend(["--figure-label-size", str(self.analysis_figure_label_size)])
        if self.analysis_figure_tick_size is not None:
            cmd.extend(["--figure-tick-size", str(self.analysis_figure_tick_size)])
        if self.analysis_pairwise_dimension_weighted_winner:
            cmd.append("--pairwise-dimension-weighted-winner")
            if self.analysis_pairwise_dimension_weighted_configs:
                cmd.extend(
                    [
                        "--pairwise-dimension-weighted-configs",
                        *self.analysis_pairwise_dimension_weighted_configs,
                    ]
                )

        if self.analysis_pairwise_correctness_mode != "ignore":
            cmd.extend(
                [
                    "--pairwise-correctness-mode",
                    self.analysis_pairwise_correctness_mode,
                ]
            )
        if self.analysis_pairwise_include_plus_correctness:
            cmd.append("--pairwise-include-plus-correctness")

        if self.analysis_pairwise_source:
            cmd.extend(["--pairwise-source", self.analysis_pairwise_source])
            if self.analysis_pairwise_source == "indicator_scores":
                cmd.extend(
                    [
                        "--pairwise-indicator-score-field",
                        self.analysis_pairwise_indicator_score_field,
                    ]
                )

        self._execute_command(cmd, f"analyze:{','.join(personas)}")

    def _find_dataset_dir(
        self, persona: str, prompt_type: Optional[str] = None
    ) -> Optional[Path]:
        """
        Find the dataset directory for a persona.

        Args:
            persona: Persona name.
            prompt_type: Optional prompt type to allow fallback for original/control.

        Returns:
            Path to dataset directory, or None if not found.
        """
        filter_model = self.config.defaults.get("filter_model", "none")

        personas_to_check = [persona]
        # OPTIMIZATION: If prompt_type is original or control, also check the reference persona
        # Use novice_user as reference if present, otherwise fallback to the first persona
        reference_persona = (
            "novice_user"
            if "novice_user" in self.config.use_personas
            else (self.config.use_personas[0] if self.config.use_personas else None)
        )
        if (
            prompt_type in ["original", "control"]
            and reference_persona
            and persona != reference_persona
        ):
            personas_to_check.append(reference_persona)

        for check_persona in personas_to_check:
            dataset_base = (
                self.config.base_dir
                / check_persona
                / "3_vibe_dataset"
                / f"gen_model_{self.config.generator}"
                / f"filter_model_{filter_model}"
            )

            if not dataset_base.exists():
                continue

            # Look for any dataset directory with files
            for dataset_dir in sorted(dataset_base.iterdir(), reverse=True):
                if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset_"):
                    if any(dataset_dir.glob("*.json")):
                        return dataset_dir

        return None

    def _execute_command(self, cmd: List[str], label: str) -> None:
        """
        Execute a command or print it in dry-run mode.

        Args:
            cmd: Command and arguments to execute.
            label: Human-readable label for logging.

        Raises:
            TaskExecutionError: If the command fails.
        """
        cmd_str = " ".join(cmd)

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_job_id:
            logger.debug(
                "Executing task under Slurm job %s (array task: %s)",
                slurm_job_id,
                slurm_task_id or "none",
            )

        if self.dry_run:
            print(f"[DRY-RUN] {label}")
            print(f"  Command: {cmd_str}")
            return

        if self.verbose:
            logger.info("Executing: %s", cmd_str)

        try:
            # Don't capture output - let it stream to console for progress visibility
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with exit code {e.returncode}"
            raise TaskExecutionError(error_msg) from e
