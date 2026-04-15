"""
Task generation and filtering for experiments.

This module defines the Task dataclass and TaskGenerator for creating
the full experiment task matrix with support for filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Set, Tuple

from scripts.experiment.config import ExperimentConfig
from scripts.experiment.state import ExperimentState
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_GENERAL_USER,
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
    uses_shared_general_user_artifacts,
)
from src.vibe_testing.pathing import normalize_token

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    A single executable unit of work in an experiment.

    Attributes:
        task_id: Unique identifier for this task.
        stage: Stage name ('dataset', 'objective', 'subjective', 'analyze').
        persona: Persona name (or comma-separated list for analyze).
        model: Model name (for objective/subjective stages).
        judge: Judge model name (for subjective stage).
    """

    task_id: int
    stage: str
    persona: str
    model: Optional[str] = None
    judge: Optional[str] = None
    prompt_type: Optional[str] = None
    pairwise_judgment_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization."""
        d: Dict[str, Any] = {
            "task_id": self.task_id,
            "stage": self.stage,
            "persona": self.persona,
        }
        if self.model:
            d["model"] = self.model
        if self.judge:
            d["judge"] = self.judge
        if self.prompt_type:
            d["prompt_type"] = self.prompt_type
        if self.pairwise_judgment_type:
            d["pairwise_judgment_type"] = self.pairwise_judgment_type
        return d

    def describe(self) -> str:
        """Get human-readable description of the task."""
        if self.stage == "dataset":
            return f"Generate dataset for {self.persona}"
        elif self.stage == "objective":
            desc = f"Objective eval: {self.persona} / {self.model}"
            if self.prompt_type:
                desc += f" / prompt_type={self.prompt_type}"
            return desc
        elif self.stage == "subjective":
            desc = (
                f"Subjective eval: {self.persona} / {self.model} / judge={self.judge}"
            )
            if self.prompt_type:
                desc += f" / prompt_type={self.prompt_type}"
            return desc
        elif self.stage == "indicators":
            desc = f"Indicator scores: {self.persona} / {self.model}"
            if self.prompt_type:
                desc += f" / prompt_type={self.prompt_type}"
            return desc
        elif self.stage == "pairwise":
            desc = f"Pairwise comparison: {self.persona} / {self.model} / judge={self.judge}"
            if self.pairwise_judgment_type:
                desc += f" / judgment_type={self.pairwise_judgment_type}"
            if self.prompt_type:
                desc += f" / prompt_type={self.prompt_type}"
            return desc
        elif self.stage == "analyze":
            return f"Analyze results for: {self.persona}"
        else:
            return f"Unknown task: {self.stage}"

    def __hash__(self):
        return hash(
            (
                self.stage,
                self.persona,
                self.model,
                self.judge,
                self.prompt_type,
                self.pairwise_judgment_type,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return (
            self.stage == other.stage
            and self.persona == other.persona
            and self.model == other.model
            and self.judge == other.judge
            and self.prompt_type == other.prompt_type
            and self.pairwise_judgment_type == other.pairwise_judgment_type
        )


class TaskGenerator:
    """
    Generates the full task list for an experiment.

    Creates tasks for each stage based on the experiment configuration,
    with support for filtering by specific items, tags, and completion state.
    """

    # Stage order for dependency tracking
    STAGE_ORDER = [
        "dataset",
        "objective",
        "subjective",
        "indicators",
        "pairwise",
        "analyze",
    ]

    def __init__(self, config: ExperimentConfig):
        """
        Initialize task generator.

        Args:
            config: Experiment configuration.
        """
        self.config = config

    def _get_pairwise_judgment_types(self) -> List[str]:
        """
        Get configured pairwise judgment types with a safe fallback for mocks.

        Returns:
            List[str]: Pairwise judgment types to expand.
        """
        getter = getattr(self.config, "get_pairwise_judgment_types", None)
        if callable(getter):
            judgment_types = getter()
            if isinstance(judgment_types, (list, tuple)) and judgment_types:
                return [str(item) for item in judgment_types]
        return ["persona"]

    def generate_all_tasks(
        self,
        personas: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        judges: Optional[List[str]] = None,
        stages: Optional[List[str]] = None,
        prompt_types_override: Optional[List[str]] = None,
        pairwise_judgment_types_override: Optional[List[str]] = None,
    ) -> List[Task]:
        """
        Generate all tasks for the experiment matrix.

        Args:
            personas: Specific personas to include (default: all from config).
            models: Specific models to include (default: all from config).
            judges: Specific judges to include (default: all from config).
            stages: Specific stages to include (default: all).
            pairwise_judgment_types_override: Optional override for pairwise
                judgment types.

        Returns:
            List of Task objects in execution order.
        """
        # Apply filters or use config defaults
        use_personas = personas if personas is not None else self.config.use_personas
        use_models = models if models is not None else self.config.use_models
        use_judges = judges if judges is not None else self.config.use_judges
        use_stages = stages if stages is not None else self.STAGE_ORDER

        # Normalize stage names
        if "all" in use_stages:
            use_stages = self.STAGE_ORDER

        tasks: List[Task] = []
        task_id = 1

        # Stage: Dataset generation (one per persona)
        if "dataset" in use_stages:
            for persona in use_personas:
                tasks.append(Task(task_id, "dataset", persona))
                task_id += 1

        # Stage: Objective evaluation (persona x model)
        if "objective" in use_stages:
            prompt_types = self._resolve_prompt_types(
                "objective", prompt_types_override
            )

            # Use novice_user as reference if present, otherwise fallback to the first persona
            reference_persona = (
                "novice_user"
                if "novice_user" in use_personas
                else (use_personas[0] if use_personas else None)
            )

            for persona, model in product(use_personas, use_models):
                for prompt_type in prompt_types:
                    # OPTIMIZATION: only run original and control for the reference persona
                    if (
                        prompt_type in ["original", "control"]
                        and persona != reference_persona
                    ):
                        logger.debug(
                            "Skipping %s evaluation for %s (will reuse from %s)",
                            prompt_type,
                            persona,
                            reference_persona,
                        )
                        continue

                    tasks.append(
                        Task(
                            task_id,
                            "objective",
                            persona,
                            model=model,
                            prompt_type=prompt_type,
                        )
                    )
                    task_id += 1

        # Stage: Subjective evaluation (persona x model x judge)
        if "subjective" in use_stages:
            prompt_types = self._resolve_prompt_types(
                "subjective", prompt_types_override
            )
            for persona, model, judge in product(use_personas, use_models, use_judges):
                for prompt_type in prompt_types:
                    tasks.append(
                        Task(
                            task_id,
                            "subjective",
                            persona,
                            model=model,
                            judge=judge,
                            prompt_type=prompt_type,
                        )
                    )
                    task_id += 1

        # Stage: Indicator scores (persona x model)
        if "indicators" in use_stages:
            prompt_types = self._resolve_prompt_types(
                "indicators", prompt_types_override
            )
            for persona, model in product(use_personas, use_models):
                for prompt_type in prompt_types:
                    tasks.append(
                        Task(
                            task_id,
                            "indicators",
                            persona,
                            model=model,
                            prompt_type=prompt_type,
                        )
                    )
                    task_id += 1

        # Stage: Pairwise comparison (persona x model_pair x judge)
        if "pairwise" in use_stages:
            # Generate all unique pairs of models
            model_pairs = list(combinations(use_models, 2))
            prompt_types = self._resolve_prompt_types("pairwise", prompt_types_override)
            pairwise_judgment_types = (
                [
                    normalize_pairwise_judgment_type(item)
                    for item in pairwise_judgment_types_override
                ]
                if pairwise_judgment_types_override is not None
                else self._get_pairwise_judgment_types()
            )
            reference_persona = (
                "novice_user"
                if "novice_user" in use_personas
                else (use_personas[0] if use_personas else None)
            )
            for persona in use_personas:
                for model_a, model_b in model_pairs:
                    for judge in use_judges:
                        # Encode model pair in the model field
                        model_pair_str = f"{model_a}_vs_{model_b}"
                        for judgment_type in pairwise_judgment_types:
                            for prompt_type in prompt_types:
                                if (
                                    reference_persona
                                    and persona != reference_persona
                                    and uses_shared_general_user_artifacts(
                                        judgment_type, prompt_type
                                    )
                                ):
                                    logger.debug(
                                        "Skipping pairwise prompt_type=%s for %s "
                                        "judgment_type=%s persona=%s (will reuse from %s)",
                                        prompt_type,
                                        (
                                            "shared general-user"
                                            if judgment_type
                                            == PAIRWISE_JUDGMENT_TYPE_GENERAL_USER
                                            else "persona"
                                        ),
                                        judgment_type,
                                        persona,
                                        reference_persona,
                                    )
                                    continue
                                tasks.append(
                                    Task(
                                        task_id,
                                        "pairwise",
                                        persona,
                                        model=model_pair_str,
                                        judge=judge,
                                        prompt_type=prompt_type,
                                        pairwise_judgment_type=judgment_type,
                                    )
                                )
                                task_id += 1

        # Stage: Analysis (one task covering all personas)
        if "analyze" in use_stages:
            all_personas = ",".join(use_personas)
            tasks.append(Task(task_id, "analyze", all_personas))

        logger.info(
            "Generated %d tasks: %d dataset, %d objective, %d subjective, %d indicators, "
            "%d pairwise, %d analyze",
            len(tasks),
            sum(1 for t in tasks if t.stage == "dataset"),
            sum(1 for t in tasks if t.stage == "objective"),
            sum(1 for t in tasks if t.stage == "subjective"),
            sum(1 for t in tasks if t.stage == "indicators"),
            sum(1 for t in tasks if t.stage == "pairwise"),
            sum(1 for t in tasks if t.stage == "analyze"),
        )

        return tasks

    def _resolve_prompt_types(
        self, stage: str, prompt_types_override: Optional[List[str]]
    ) -> List[Optional[str]]:
        """
        Determine which prompt types should be expanded for a stage.

        Args:
            stage: The stage identifier.
            prompt_types_override: Optional override list from the CLI.

        Returns:
            List of prompt type tokens. Defaults to [None] when no prompt types are configured.
        """
        if prompt_types_override is not None:
            source_list = prompt_types_override
        else:
            source_list = self.config.get_prompt_types(stage)

        if not source_list:
            return [None]

        normalized: List[Optional[str]] = []
        seen: Set[Optional[str]] = set()
        for prompt_type in source_list:
            token = normalize_token(prompt_type)
            if token not in seen:
                normalized.append(token)
                seen.add(token)

        return normalized if normalized else [None]

    def filter_by_tags(
        self,
        tasks: List[Task],
        model_tags: Optional[List[str]] = None,
        persona_tags: Optional[List[str]] = None,
    ) -> List[Task]:
        """
        Filter tasks by model and/or persona tags.

        Args:
            tasks: List of tasks to filter.
            model_tags: Keep only tasks with models having these tags.
            persona_tags: Keep only tasks with personas having these tags.

        Returns:
            Filtered list of tasks.
        """
        result = tasks

        if model_tags:
            allowed_models = set(self.config.filter_models_by_tags(model_tags))
            # Also include allowed models as judges
            allowed_judges = set(self.config.filter_judges_by_tags(model_tags))

            filtered = []
            for task in result:
                # Dataset and analyze tasks don't have model constraints
                if task.stage == "dataset":
                    filtered.append(task)
                elif task.stage == "analyze":
                    filtered.append(task)
                elif task.stage == "objective":
                    if task.model in allowed_models:
                        filtered.append(task)
                elif task.stage == "subjective":
                    # For subjective, filter by evaluated model tag
                    if task.model in allowed_models:
                        filtered.append(task)
                elif task.stage == "pairwise":
                    # For pairwise, check if both models in the pair are allowed
                    if task.model and "_vs_" in task.model:
                        model_a, model_b = task.model.split("_vs_", 1)
                        if model_a in allowed_models and model_b in allowed_models:
                            filtered.append(task)
            result = filtered

            logger.debug(
                "After model tag filter (%s): %d tasks", model_tags, len(result)
            )

        if persona_tags:
            allowed_personas = set(self.config.filter_personas_by_tags(persona_tags))

            filtered = []
            for task in result:
                if task.stage == "analyze":
                    # Analyze task - check if any of its personas match
                    task_personas = set(task.persona.split(","))
                    if task_personas & allowed_personas:
                        filtered.append(task)
                else:
                    if task.persona in allowed_personas:
                        filtered.append(task)
            result = filtered

            logger.debug(
                "After persona tag filter (%s): %d tasks", persona_tags, len(result)
            )

        return result

    def filter_completed(
        self,
        tasks: List[Task],
        state: ExperimentState,
        force_stages: Optional[List[str]] = None,
    ) -> List[Task]:
        """
        Remove tasks that have already been completed.

        Args:
            tasks: List of tasks to filter.
            state: Experiment state with completion information.
            force_stages: List of stages to skip completion check for.

        Returns:
            List of tasks that still need to be executed.
        """
        existing_datasets = state.get_existing_datasets()
        existing_objective = state.get_existing_objective_evals()
        existing_subjective = state.get_existing_subjective_evals()
        existing_indicators = (
            state.get_existing_indicator_scores()
            if hasattr(state, "get_existing_indicator_scores")
            else set()
        )
        existing_pairwise = state.get_existing_pairwise_evals()

        # NOTE: Tests sometimes pass a MagicMock as `state`. MagicMock will happily
        # fabricate attributes on demand, which would make `getattr(state, ...)`
        # return a callable mock even when the real ExperimentState does not
        # provide the method. We therefore check support on the concrete class.
        supports_model_canon = hasattr(state.__class__, "canonicalize_task_model_key")
        supports_judge_canon = hasattr(state.__class__, "canonicalize_task_judge_key")

        canonicalize_model = (
            getattr(state, "canonicalize_task_model_key", None)
            if supports_model_canon
            else None
        )
        canonicalize_judge = (
            getattr(state, "canonicalize_task_judge_key", None)
            if supports_judge_canon
            else None
        )

        def _canon_model(model_key: Optional[str]) -> Optional[str]:
            """
            Canonicalize a model key if the ExperimentState supports it.

            Args:
                model_key (Optional[str]): Model key from a task.

            Returns:
                Optional[str]: Canonical key when possible, else the original input.
            """
            if callable(canonicalize_model):
                return canonicalize_model(model_key)
            return model_key

        def _canon_judge(judge_key: Optional[str]) -> Optional[str]:
            """
            Canonicalize a judge key if the ExperimentState supports it.

            Args:
                judge_key (Optional[str]): Judge key from a task.

            Returns:
                Optional[str]: Canonical key when possible, else the original input.
            """
            if callable(canonicalize_judge):
                return canonicalize_judge(judge_key)
            return judge_key

        pending: List[Task] = []
        skipped = 0
        force_stages = force_stages or []

        for task in tasks:
            # Skip check if stage is forced
            if task.stage in force_stages:
                pending.append(task)
                continue

            is_completed = False

            if task.stage == "dataset":
                is_completed = task.persona in existing_datasets
            elif task.stage == "objective":
                model_key = _canon_model(task.model)
                is_completed = (
                    task.persona,
                    model_key,
                    task.prompt_type,
                ) in existing_objective
            elif task.stage == "subjective":
                model_key = _canon_model(task.model)
                judge_key = _canon_judge(task.judge)
                is_completed = (
                    task.persona,
                    model_key,
                    judge_key,
                    task.prompt_type,
                ) in existing_subjective
            elif task.stage == "indicators":
                model_key = _canon_model(task.model)
                is_completed = (
                    task.persona,
                    model_key,
                    task.prompt_type,
                ) in existing_indicators
            elif task.stage == "pairwise":
                # Parse model pair from task.model (e.g., "gpt4_vs_gpt35")
                if task.model and "_vs_" in task.model:
                    raw_a, raw_b = task.model.split("_vs_", 1)
                    model_a = _canon_model(raw_a) or raw_a
                    model_b = _canon_model(raw_b) or raw_b
                    judge_key = _canon_judge(task.judge)

                    # ExperimentState historically stored pairwise tuples in a canonical
                    # (sorted) model-pair order, but some legacy discovery paths may
                    # preserve directory order. To avoid re-running completed tasks,
                    # treat either ordering as equivalent for completion checks.
                    key_unsorted = (
                        task.persona,
                        model_a,
                        model_b,
                        judge_key,
                        task.prompt_type,
                        task.pairwise_judgment_type or PAIRWISE_JUDGMENT_TYPE_PERSONA,
                    )
                    model_x, model_y = sorted([model_a, model_b])
                    key_sorted = (
                        task.persona,
                        model_x,
                        model_y,
                        judge_key,
                        task.prompt_type,
                        task.pairwise_judgment_type or PAIRWISE_JUDGMENT_TYPE_PERSONA,
                    )
                    is_completed = (
                        key_unsorted in existing_pairwise
                        or key_sorted in existing_pairwise
                    )
            # analyze stage is never skipped - always runs

            if is_completed:
                skipped += 1
                logger.debug("Skipping completed task: %s", task.describe())
            else:
                pending.append(task)

        logger.info(
            "Filtered tasks: %d pending, %d already completed", len(pending), skipped
        )
        return pending

    def filter_by_task_ids(
        self,
        tasks: List[Task],
        task_ids: List[int],
    ) -> List[Task]:
        """
        Filter tasks to only those with specific IDs.

        Args:
            tasks: List of tasks to filter.
            task_ids: List of task IDs to keep.

        Returns:
            Tasks matching the specified IDs.
        """
        id_set = set(task_ids)
        return [t for t in tasks if t.task_id in id_set]

    def parse_task_id_spec(self, spec: str) -> List[int]:
        """
        Parse a task ID specification string.

        Supports:
        - Single ID: "5"
        - Range: "3-7"
        - Comma-separated: "1,3,5"
        - Mixed: "1,3-5,8"

        Args:
            spec: Task ID specification string.

        Returns:
            List of task IDs.
        """
        ids: List[int] = []

        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                ids.extend(range(int(start), int(end) + 1))
            else:
                ids.append(int(part))

        return sorted(set(ids))

    def get_task_dependencies(self, task: Task) -> List[str]:
        """
        Get the stages that must complete before this task can run.

        Args:
            task: Task to check dependencies for.

        Returns:
            List of stage names that are dependencies.
        """
        deps: List[str] = []

        if task.stage == "objective":
            # Objective requires dataset
            deps.append("dataset")
        elif task.stage == "subjective":
            # Subjective requires objective (which requires dataset)
            deps.extend(["dataset", "objective"])
        elif task.stage == "indicators":
            # Indicator scores require objective (which requires dataset)
            deps.extend(["dataset", "objective"])
        elif task.stage == "pairwise":
            # Pairwise requires objective for both models (which requires dataset)
            deps.extend(["dataset", "objective"])
        elif task.stage == "analyze":
            # Analyze can run with whatever data exists
            pass

        return deps
