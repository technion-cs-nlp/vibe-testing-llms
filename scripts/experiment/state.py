"""
Experiment state detection.

This module scans experiment directories to determine which tasks
have already been completed, enabling incremental experiment runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from scripts.experiment.config import ExperimentConfig
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
)
from src.vibe_testing.pathing import infer_prompt_type, normalize_token
from src.vibe_testing.utils import load_json

logger = logging.getLogger(__name__)


class ExperimentState:
    """
    Discovers completed work in experiment directories.

    Scans the canonical directory structure to identify:
    - Which personas have datasets generated
    - Which (persona, model) pairs have objective evaluations
    - Which (persona, model, judge) triples have subjective evaluations
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize state scanner for an experiment.

        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.base_dir = config.base_dir

    def _get_pairwise_judgment_types(self) -> List[str]:
        """
        Get configured pairwise judgment types with a safe fallback for mocks.

        Returns:
            List[str]: Pairwise judgment types.
        """
        getter = getattr(self.config, "get_pairwise_judgment_types", None)
        if callable(getter):
            judgment_types = getter()
            if isinstance(judgment_types, (list, tuple)) and judgment_types:
                return [str(item) for item in judgment_types]
        return [PAIRWISE_JUDGMENT_TYPE_PERSONA]

    # NOTE: The state scanner internally canonicalizes model/judge keys it finds on disk
    # back to the *experiment-config keys* (see `_canonicalize_model_key`). The task
    # generator (and CLI overrides like --only-models/--only-judges) may use aliases
    # that are not present in `config.use_models`/`config.use_judges`, even when they
    # point to the same underlying model config path. Exposing these helpers lets the
    # orchestrator compare tasks to discovered artifacts using the same canonical keys.

    def canonicalize_task_model_key(self, model_key: Optional[str]) -> Optional[str]:
        """
        Canonicalize a task-provided model key to the experiment's configured model keys.

        This is primarily used to make completion checks robust when callers pass
        alias model keys via CLI overrides (e.g., ``gpt4-o``) while the experiment
        configuration uses a different key for the same underlying model config
        (e.g., ``gpt4``).

        Args:
            model_key (Optional[str]): Model key from a Task or CLI override.

        Returns:
            Optional[str]: Canonical model key (from ``config.use_models``) if resolvable;
                otherwise returns the original key (or None if input is None).
        """
        if model_key is None:
            return None
        canonical = self._canonicalize_model_key(model_key, self.config.use_models)
        return canonical or model_key

    def canonicalize_task_judge_key(self, judge_key: Optional[str]) -> Optional[str]:
        """
        Canonicalize a task-provided judge key to the experiment's configured judge keys.

        Args:
            judge_key (Optional[str]): Judge key from a Task or CLI override.

        Returns:
            Optional[str]: Canonical judge key (from ``config.use_judges``) if resolvable;
                otherwise returns the original key (or None if input is None).
        """
        if judge_key is None:
            return None
        canonical = self._canonicalize_model_key(judge_key, self.config.use_judges)
        return canonical or judge_key

    def get_existing_datasets(self) -> Set[str]:
        """
        Find personas that have existing datasets.

        Scans for non-empty dataset directories under:
        {base_dir}/{persona}/3_vibe_dataset/gen_model_{generator}/filter_model_{filter}/

        Returns:
            Set of persona names with existing datasets.
        """
        existing: Set[str] = set()
        generator = self.config.generator
        filter_model = self.config.defaults.get("filter_model", "none")

        for persona in self.config.use_personas:
            dataset_base = (
                self.base_dir
                / persona
                / "3_vibe_dataset"
                / f"gen_model_{generator}"
                / f"filter_model_{filter_model}"
            )

            if not dataset_base.exists():
                continue

            # Check for any dataset directory with JSON files
            for dataset_dir in dataset_base.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset_"):
                    json_files = list(dataset_dir.glob("*.json"))
                    if json_files:
                        logger.debug(
                            "Found existing dataset for %s: %s (%d files)",
                            persona,
                            dataset_dir,
                            len(json_files),
                        )
                        existing.add(persona)
                        break

        logger.info("Found %d personas with existing datasets", len(existing))
        return existing

    def get_existing_objective_evals(
        self,
    ) -> Set[Tuple[str, str, Optional[str]]]:
        """
        Find (persona, model) pairs with existing objective evaluations.

        Scans for result files under:
        {base_dir}/{persona}/4_objective_evaluation/evaluated_model_{model}/...

        Returns:
            Set of (persona, model, prompt_type) tuples with existing evaluations.
        """
        existing: Set[Tuple[str, str, Optional[str]]] = set()
        generator = self.config.generator
        filter_model = self.config.defaults.get("filter_model", "none")

        # Only count objective evaluations as "completed" when we see a summary/metrics
        # artifact. This avoids false positives from partially-written sample files,
        # temporary artifacts, or logs.
        objective_completion_patterns = [
            "function_eval_metrics_*.json",
            "function_eval_metrics_*.csv",
            "patch_eval_metrics_*.json",
            "patch_eval_metrics_*.csv",
        ]

        for persona in self.config.use_personas:
            obj_root = self.base_dir / persona / "4_objective_evaluation"
            if not obj_root.exists():
                continue

            for eval_model_dir in obj_root.iterdir():
                if not eval_model_dir.is_dir():
                    continue
                if not eval_model_dir.name.startswith("evaluated_model_"):
                    continue

                raw_model = eval_model_dir.name.replace("evaluated_model_", "")
                model = self._canonicalize_model_key(raw_model, self.config.use_models)
                if model is None:
                    logger.debug(
                        "Skipping objective eval dir with unknown model '%s' (persona=%s)",
                        raw_model,
                        persona,
                    )
                    continue

                # Check for expected nested structure
                gen_dir = eval_model_dir / f"gen_model_{generator}"
                if not gen_dir.exists():
                    continue

                filter_dir = gen_dir / f"filter_model_{filter_model}"
                if not filter_dir.exists():
                    continue

                prompt_type_dirs = sorted(
                    [
                        entry
                        for entry in filter_dir.iterdir()
                        if entry.is_dir() and entry.name.startswith("prompt_type_")
                    ],
                    reverse=True,
                )
                for prompt_dir in prompt_type_dirs:
                    raw_prompt = prompt_dir.name.replace("prompt_type_", "", 1)
                    try:
                        prompt_token = normalize_token(raw_prompt)
                    except ValueError:
                        prompt_token = raw_prompt
                    if self._has_artifacts(prompt_dir, objective_completion_patterns):
                        logger.debug(
                            "Found existing objective eval: %s / %s / %s",
                            persona,
                            model,
                            prompt_token,
                        )
                        existing.add((persona, model, prompt_token))
                    else:
                        self._cleanup_empty_dirs(prompt_dir)

                # Legacy (non-prompt_type) run directories
                legacy_dirs = sorted(
                    [
                        entry
                        for entry in filter_dir.iterdir()
                        if entry.is_dir() and not entry.name.startswith("prompt_type_")
                    ],
                    reverse=True,
                )
                for run_dir in legacy_dirs:
                    found_any_pt = False
                    # Check for prompt-specific summary files first (new naming scheme)
                    for pt in ["original", "personalized", "control"]:
                        if self._has_artifacts(
                            run_dir, [f"*-{pt}-metrics_*.json", f"*-{pt}-metrics_*.csv"]
                        ):
                            logger.debug(
                                "Found prompt-specific objective summary in legacy dir: %s / %s / %s",
                                persona,
                                model,
                                pt,
                            )
                            existing.add((persona, model, pt))
                            found_any_pt = True

                    # If no specific summaries found, check individual result files
                    # but only if we didn't find any specialized summary yet.
                    # Or even if we did, there might be other types.
                    if not found_any_pt:
                        # Check a sample of files to infer types present in this legacy run
                        # Search recursively since they might be in a 'function_eval' subfolder
                        sample_files = list(
                            run_dir.rglob("function_eval_objective_sample-*.json")
                        )[:20]
                        if sample_files:
                            pts = {infer_prompt_type(f.name) for f in sample_files}
                            for pt in pts:
                                logger.debug(
                                    "Inferred objective prompt type '%s' from files in legacy dir: %s / %s",
                                    pt,
                                    persona,
                                    model,
                                )
                                existing.add((persona, model, pt))
                            found_any_pt = True

                    if not found_any_pt and self._has_artifacts(
                        run_dir, objective_completion_patterns
                    ):
                        logger.debug(
                            "Found generic objective eval: %s / %s", persona, model
                        )
                        existing.add((persona, model, None))
                        break

                # Direct artifacts under filter_dir (rare but possible)
                if (
                    not prompt_type_dirs
                    and not legacy_dirs
                    and self._has_artifacts(filter_dir, objective_completion_patterns)
                ):
                    logger.debug(
                        "Found legacy objective eval at root: %s / %s", persona, model
                    )
                    existing.add((persona, model, None))

        logger.info("Found %d existing objective evaluations", len(existing))
        return existing

    def get_existing_subjective_evals(
        self,
    ) -> Set[Tuple[str, str, str, Optional[str]]]:
        """
        Find (persona, model, judge) triples with existing subjective evaluations.

        Scans for result files under:
        {base_dir}/{persona}/5_subjective_evaluation/evaluated_model_{model}/
            sub_judge_model_{judge}/gen_model_{generator}/filter_model_{filter}/

        Returns:
            Set of (persona, model, judge, prompt_type) tuples with existing evaluations.
        """
        existing: Set[Tuple[str, str, str, Optional[str]]] = set()
        generator = self.config.generator
        filter_model = self.config.defaults.get("filter_model", "none")

        # Only count subjective evaluations as "completed" when we see the
        # canonical subjective_scores artifact(s). This avoids counting logs,
        # intermediate artifacts, or empty placeholder files.
        subjective_completion_patterns = [
            "subjective_scores_*.json",
            "subjective_scores_*.jsonl",
        ]

        for persona in self.config.use_personas:
            subj_root = self.base_dir / persona / "5_subjective_evaluation"
            if not subj_root.exists():
                continue

            for eval_model_dir in subj_root.iterdir():
                if not eval_model_dir.is_dir():
                    continue
                if not eval_model_dir.name.startswith("evaluated_model_"):
                    continue

                raw_model = eval_model_dir.name.replace("evaluated_model_", "")
                model = self._canonicalize_model_key(raw_model, self.config.use_models)
                if model is None:
                    logger.debug(
                        "Skipping subjective eval dir with unknown model '%s' (persona=%s)",
                        raw_model,
                        persona,
                    )
                    continue

                for judge_dir in eval_model_dir.iterdir():
                    if not judge_dir.is_dir():
                        continue
                    if not judge_dir.name.startswith("sub_judge_model_"):
                        continue

                    raw_judge = judge_dir.name.replace("sub_judge_model_", "")
                    judge = self._canonicalize_model_key(
                        raw_judge, self.config.use_judges
                    )
                    if judge is None:
                        logger.debug(
                            "Skipping subjective eval judge dir with unknown judge '%s' (persona=%s model=%s)",
                            raw_judge,
                            persona,
                            model,
                        )
                        continue

                    # Check for expected nested structure
                    gen_dir = judge_dir / f"gen_model_{generator}"
                    if not gen_dir.exists():
                        continue

                    filter_dir = gen_dir / f"filter_model_{filter_model}"
                    if not filter_dir.exists():
                        continue

                    prompt_type_dirs = sorted(
                        [
                            entry
                            for entry in filter_dir.iterdir()
                            if entry.is_dir() and entry.name.startswith("prompt_type_")
                        ],
                        reverse=True,
                    )
                    for prompt_dir in prompt_type_dirs:
                        raw_prompt = prompt_dir.name.replace("prompt_type_", "", 1)
                        try:
                            prompt_token = normalize_token(raw_prompt)
                        except ValueError:
                            prompt_token = raw_prompt
                        if self._has_artifacts(
                            prompt_dir, subjective_completion_patterns
                        ):
                            logger.debug(
                                "Found existing subjective eval: %s / %s / %s / %s",
                                persona,
                                model,
                                judge,
                                prompt_token,
                            )
                            existing.add((persona, model, judge, prompt_token))
                        else:
                            self._cleanup_empty_dirs(prompt_dir)

                    legacy_dirs = sorted(
                        [
                            entry
                            for entry in filter_dir.iterdir()
                            if entry.is_dir()
                            and not entry.name.startswith("prompt_type_")
                        ],
                        reverse=True,
                    )
                    for run_dir in legacy_dirs:
                        found_any_pt = False
                        # Check for prompt-specific summary files first
                        for pt in ["original", "personalized", "control"]:
                            if self._has_artifacts(
                                run_dir, [f"*subjective_scores_*-{pt}_*.json"]
                            ):
                                logger.debug(
                                    "Found prompt-specific subjective summary in legacy dir: %s / %s / %s / %s",
                                    persona,
                                    model,
                                    judge,
                                    pt,
                                )
                                existing.add((persona, model, judge, pt))
                                found_any_pt = True

                        if not found_any_pt:
                            # Try to infer from files if any
                            # subjective results are usually in one or more JSON files
                            subj_files = list(run_dir.rglob("*.json"))[:5]
                            for f in subj_files:
                                try:
                                    data = load_json(str(f))
                                    if isinstance(data, list) and data:
                                        # Inspect first record
                                        task_id = data[0].get("task_id", "")
                                        pt = infer_prompt_type(task_id)
                                        logger.debug(
                                            "Inferred subjective prompt type '%s' from %s: %s / %s / %s",
                                            pt,
                                            f.name,
                                            persona,
                                            model,
                                            judge,
                                        )
                                        existing.add((persona, model, judge, pt))
                                        found_any_pt = True
                                        # Don't break, there might be mixed types in different files
                                        # although usually they are separated.
                                except Exception:
                                    continue

                        if not found_any_pt and self._has_artifacts(
                            run_dir, subjective_completion_patterns
                        ):
                            logger.debug(
                                "Found generic subjective eval: %s / %s / %s",
                                persona,
                                model,
                                judge,
                            )
                            existing.add((persona, model, judge, None))
                            break
                        self._cleanup_empty_dirs(run_dir)

                    if (
                        not prompt_type_dirs
                        and not legacy_dirs
                        and self._has_artifacts(
                            filter_dir, subjective_completion_patterns
                        )
                    ):
                        logger.debug(
                            "Found legacy subjective eval at root: %s / %s / %s",
                            persona,
                            model,
                            judge,
                        )
                        existing.add((persona, model, judge, None))

        logger.info("Found %d existing subjective evaluations", len(existing))
        return existing

    def get_existing_pairwise_evals(
        self,
    ) -> Set[Tuple[str, str, str, str, Optional[str], str]]:
        """
        Find (persona, model_a, model_b, judge) tuples with existing pairwise evaluations.

        Scans for result files under:
        {base_dir}/{persona}/5b_pairwise_evaluation/models_{model_a}_vs_{model_b}/
            judge_model_{judge}/gen_model_{generator}/filter_model_{filter}/

        Returns:
            Set of (persona, model_a, model_b, judge, prompt_type, judgment_type)
            tuples with existing evaluations.
        """
        existing: Set[Tuple[str, str, str, str, Optional[str], str]] = set()
        generator = self.config.generator
        filter_model = self.config.defaults.get("filter_model", "none")

        # Pairwise completion is represented by a single, canonical artifact
        # written by Stage 5B: pairwise-comparison_*.json
        pairwise_completion_patterns = ["pairwise-comparison_*.json"]

        for persona in self.config.use_personas:
            pairwise_root = self.base_dir / persona / "5b_pairwise_evaluation"
            if not pairwise_root.exists():
                continue

            for models_dir in pairwise_root.iterdir():
                if not models_dir.is_dir():
                    continue
                if not models_dir.name.startswith("models_"):
                    continue

                # Parse model pair from directory name (e.g., "models_gpt4_vs_gpt35")
                try:
                    pair_part = models_dir.name.replace("models_", "")
                    if "_vs_" not in pair_part:
                        continue
                    parts = pair_part.split("_vs_", 1)
                    if len(parts) != 2:
                        continue
                    raw_a, raw_b = parts
                    model_a = self._canonicalize_model_key(
                        raw_a, self.config.use_models
                    )
                    model_b = self._canonicalize_model_key(
                        raw_b, self.config.use_models
                    )
                    if model_a is None or model_b is None:
                        logger.debug(
                            "Skipping pairwise models dir with unknown model(s): %s (persona=%s)",
                            models_dir.name,
                            persona,
                        )
                        continue
                except ValueError:
                    continue

                for judge_dir in models_dir.iterdir():
                    if not judge_dir.is_dir():
                        continue
                    if not judge_dir.name.startswith("judge_model_"):
                        continue

                    raw_judge = judge_dir.name.replace("judge_model_", "")
                    judge = self._canonicalize_model_key(
                        raw_judge, self.config.use_judges
                    )
                    if judge is None:
                        logger.debug(
                            "Skipping pairwise judge dir with unknown judge '%s' (persona=%s models=%s)",
                            raw_judge,
                            persona,
                            models_dir.name,
                        )
                        continue

                    # Check for expected nested structure
                    gen_dir = judge_dir / f"gen_model_{generator}"
                    if not gen_dir.exists():
                        continue

                    filter_dir = gen_dir / f"filter_model_{filter_model}"
                    if not filter_dir.exists():
                        continue

                    judgment_dirs = sorted(
                        [
                            entry
                            for entry in filter_dir.iterdir()
                            if entry.is_dir()
                            and entry.name.startswith("judgment_type_")
                        ],
                        reverse=True,
                    )
                    filter_roots: List[Tuple[Path, str]] = []
                    if judgment_dirs:
                        for judgment_dir in judgment_dirs:
                            raw_judgment = judgment_dir.name.replace(
                                "judgment_type_", "", 1
                            )
                            judgment_type = normalize_pairwise_judgment_type(
                                raw_judgment
                            )
                            filter_roots.append((judgment_dir, judgment_type))
                    else:
                        filter_roots.append(
                            (filter_dir, PAIRWISE_JUDGMENT_TYPE_PERSONA)
                        )

                    for filter_root, judgment_type in filter_roots:
                        prompt_type_dirs = sorted(
                            [
                                entry
                                for entry in filter_root.iterdir()
                                if entry.is_dir()
                                and entry.name.startswith("prompt_type_")
                            ],
                            reverse=True,
                        )
                        for prompt_dir in prompt_type_dirs:
                            raw_prompt = prompt_dir.name.replace("prompt_type_", "", 1)
                            try:
                                prompt_token = normalize_token(raw_prompt)
                            except ValueError:
                                prompt_token = raw_prompt
                            if self._has_artifacts(
                                prompt_dir, pairwise_completion_patterns
                            ):
                                logger.debug(
                                    "Found existing pairwise eval: %s / %s vs %s / judge=%s / %s / judgment_type=%s",
                                    persona,
                                    model_a,
                                    model_b,
                                    judge,
                                    prompt_token,
                                    judgment_type,
                                )
                                model_x, model_y = sorted([model_a, model_b])
                                existing.add(
                                    (
                                        persona,
                                        model_x,
                                        model_y,
                                        judge,
                                        prompt_token,
                                        judgment_type,
                                    )
                                )
                            else:
                                self._cleanup_empty_dirs(prompt_dir)

                        legacy_dirs = sorted(
                            [
                                entry
                                for entry in filter_root.iterdir()
                                if entry.is_dir()
                                and not entry.name.startswith("prompt_type_")
                            ],
                            reverse=True,
                        )
                        for run_dir in legacy_dirs:
                            found_any_pt = False
                            # Check for prompt-specific summary files first
                            for pt in ["original", "personalized", "control"]:
                                if self._has_artifacts(
                                    run_dir, [f"*pairwise-comparison_*-{pt}_*.json"]
                                ):
                                    logger.debug(
                                        "Found prompt-specific pairwise summary in legacy dir: %s / %s vs %s / %s / %s / judgment_type=%s",
                                        persona,
                                        model_a,
                                        model_b,
                                        judge,
                                        pt,
                                        judgment_type,
                                    )
                                    existing.add(
                                        (
                                            persona,
                                            model_a,
                                            model_b,
                                            judge,
                                            pt,
                                            judgment_type,
                                        )
                                    )
                                    found_any_pt = True

                            if not found_any_pt:
                                # Try to infer from files if any
                                pair_files = list(run_dir.rglob("*.json"))[:5]
                                for f in pair_files:
                                    try:
                                        data = load_json(str(f))
                                        if isinstance(data, list) and data:
                                            # Inspect first record
                                            task_id = data[0].get("task_id", "")
                                            pt = infer_prompt_type(task_id)
                                            logger.debug(
                                                "Inferred pairwise prompt type '%s' from %s: %s / %s vs %s / %s / judgment_type=%s",
                                                pt,
                                                f.name,
                                                persona,
                                                model_a,
                                                model_b,
                                                judge,
                                                judgment_type,
                                            )
                                            existing.add(
                                                (
                                                    persona,
                                                    model_a,
                                                    model_b,
                                                    judge,
                                                    pt,
                                                    judgment_type,
                                                )
                                            )
                                            found_any_pt = True
                                    except Exception:
                                        continue

                            if not found_any_pt and self._has_artifacts(
                                run_dir, pairwise_completion_patterns
                            ):
                                logger.debug(
                                    "Found generic pairwise eval: %s / %s vs %s / %s / judgment_type=%s",
                                    persona,
                                    model_a,
                                    model_b,
                                    judge,
                                    judgment_type,
                                )
                                model_x, model_y = sorted([model_a, model_b])
                                existing.add(
                                    (
                                        persona,
                                        model_x,
                                        model_y,
                                        judge,
                                        None,
                                        judgment_type,
                                    )
                                )
                                break
                            self._cleanup_empty_dirs(run_dir)

                        if (
                            not prompt_type_dirs
                            and not legacy_dirs
                            and self._has_artifacts(
                                filter_root, pairwise_completion_patterns
                            )
                        ):
                            logger.debug(
                                "Found legacy pairwise eval at root: %s / %s vs %s / judge=%s / judgment_type=%s",
                                persona,
                                model_a,
                                model_b,
                                judge,
                                judgment_type,
                            )
                            model_x, model_y = sorted([model_a, model_b])
                            existing.add(
                                (
                                    persona,
                                    model_x,
                                    model_y,
                                    judge,
                                    None,
                                    judgment_type,
                                )
                            )

        logger.info("Found %d existing pairwise evaluations", len(existing))
        return existing

    def get_existing_indicator_scores(
        self,
    ) -> Set[Tuple[str, str, Optional[str]]]:
        """
        Find (persona, model, prompt_type) tuples with existing Stage 5c indicator scores.

        Scans for result files under:
        {base_dir}/{persona}/5c_indicator_scores/evaluated_model_{model}/gen_model_{generator}/
            filter_model_{filter}/[prompt_type_{pt}/]indicator_scores_*.json

        Returns:
            Set of (persona, model, prompt_type) tuples with existing evaluations.
        """
        existing: Set[Tuple[str, str, Optional[str]]] = set()
        generator = self.config.generator
        filter_model = self.config.defaults.get("filter_model", "none")
        completion_patterns = ["indicator_scores_*.json"]

        for persona in self.config.use_personas:
            root = self.base_dir / persona / "5c_indicator_scores"
            if not root.exists():
                continue

            for eval_model_dir in root.iterdir():
                if not eval_model_dir.is_dir() or not eval_model_dir.name.startswith(
                    "evaluated_model_"
                ):
                    continue

                raw_model = eval_model_dir.name.replace("evaluated_model_", "")
                model = self._canonicalize_model_key(raw_model, self.config.use_models)
                if model is None:
                    continue

                gen_dir = eval_model_dir / f"gen_model_{generator}"
                if not gen_dir.exists():
                    continue

                filter_dir = gen_dir / f"filter_model_{filter_model}"
                if not filter_dir.exists():
                    continue

                prompt_type_dirs = sorted(
                    [
                        entry
                        for entry in filter_dir.iterdir()
                        if entry.is_dir() and entry.name.startswith("prompt_type_")
                    ],
                    reverse=True,
                )
                for prompt_dir in prompt_type_dirs:
                    raw_prompt = prompt_dir.name.replace("prompt_type_", "", 1)
                    try:
                        prompt_token = normalize_token(raw_prompt)
                    except ValueError:
                        prompt_token = raw_prompt
                    if self._has_artifacts(prompt_dir, completion_patterns):
                        existing.add((persona, model, prompt_token))
                    else:
                        self._cleanup_empty_dirs(prompt_dir)

                legacy_dirs = sorted(
                    [
                        entry
                        for entry in filter_dir.iterdir()
                        if entry.is_dir() and not entry.name.startswith("prompt_type_")
                    ],
                    reverse=True,
                )
                for run_dir in legacy_dirs:
                    found_any_pt = False
                    for pt in ["original", "personalized", "control"]:
                        if self._has_artifacts(
                            run_dir, [f"*indicator_scores_*-{pt}_*.json"]
                        ):
                            existing.add((persona, model, pt))
                            found_any_pt = True
                    if not found_any_pt and self._has_artifacts(
                        run_dir, completion_patterns
                    ):
                        existing.add((persona, model, None))

                if (
                    not prompt_type_dirs
                    and not legacy_dirs
                    and self._has_artifacts(filter_dir, completion_patterns)
                ):
                    existing.add((persona, model, None))

        logger.info("Found %d existing indicator score artifacts", len(existing))
        return existing

    def get_completion_summary(self) -> dict:
        """
        Get a summary of experiment completion status.

        Returns:
            Dict with counts and lists of completed/pending items.
        """
        datasets = self.get_existing_datasets()
        objective = self.get_existing_objective_evals()
        subjective = self.get_existing_subjective_evals()
        indicators = self.get_existing_indicator_scores()
        pairwise = self.get_existing_pairwise_evals()

        # Calculate expected counts
        num_personas = len(self.config.use_personas)
        num_models = len(self.config.use_models)
        num_judges = len(self.config.use_judges)
        # Number of unique model pairs: C(n,2) = n*(n-1)/2
        num_model_pairs = num_models * (num_models - 1) // 2

        objective_prompt_types = self._prompt_types_for_stage("objective")
        subjective_prompt_types = self._prompt_types_for_stage("subjective")
        indicator_prompt_types = self._prompt_types_for_stage("indicators")
        pairwise_prompt_types = self._prompt_types_for_stage("pairwise")

        expected_datasets = num_personas
        expected_objective = num_personas * num_models * len(objective_prompt_types)
        expected_subjective = (
            num_personas * num_models * num_judges * len(subjective_prompt_types)
        )
        expected_indicators = num_personas * num_models * len(indicator_prompt_types)
        expected_pairwise = 0
        reference_persona = (
            "novice_user"
            if "novice_user" in self.config.use_personas
            else (self.config.use_personas[0] if self.config.use_personas else None)
        )
        for judgment_type in self._get_pairwise_judgment_types():
            for prompt_type in pairwise_prompt_types:
                if (
                    judgment_type == "general_user"
                    and prompt_type in {"original", "control"}
                    and reference_persona
                ):
                    expected_pairwise += num_model_pairs * num_judges
                else:
                    expected_pairwise += num_personas * num_model_pairs * num_judges

        return {
            "datasets": {
                "completed": len(datasets),
                "expected": expected_datasets,
                "personas": list(datasets),
            },
            "objective": {
                "completed": len(objective),
                "expected": expected_objective,
                "pairs": [(p, m, pt) for p, m, pt in objective],
            },
            "subjective": {
                "completed": len(subjective),
                "expected": expected_subjective,
                "triples": [(p, m, j, pt) for p, m, j, pt in subjective],
            },
            "indicators": {
                "completed": len(indicators),
                "expected": expected_indicators,
                "pairs": [(p, m, pt) for p, m, pt in indicators],
            },
            "pairwise": {
                "completed": len(pairwise),
                "expected": expected_pairwise,
                "quads": [(p, ma, mb, j, pt, jt) for p, ma, mb, j, pt, jt in pairwise],
            },
        }

    def _prompt_types_for_stage(self, stage: str) -> List[Optional[str]]:
        """
        Return the normalized prompt types configured for a given stage.

        Args:
            stage: Stage identifier (e.g., "objective").

        Returns:
            List of prompt type tokens, or [None] when no prompt types are configured.
        """
        prompt_types = self.config.get_prompt_types(stage)
        if not prompt_types:
            return [None]

        normalized: List[Optional[str]] = []
        seen: Set[Optional[str]] = set()
        for prompt_type in prompt_types:
            try:
                token = normalize_token(prompt_type)
            except ValueError:
                token = prompt_type
            if token not in seen:
                normalized.append(token)
                seen.add(token)

        return normalized if normalized else [None]

    def _has_artifacts(self, root: Path, patterns: Iterable[str]) -> bool:
        """
        Recursively check for artifacts matching any pattern under a root path.

        Args:
            root: Directory to search.
            patterns: Glob patterns to match.

        Returns:
            True if any matching file is found; otherwise False.
        """
        if not root.exists():
            return False

        for pattern in patterns:
            hit = next(root.rglob(pattern), None)
            if hit is None:
                continue
            try:
                if hit.is_file() and hit.stat().st_size == 0:
                    continue
            except OSError:
                # If we cannot stat the file, be conservative and treat it as absent.
                continue
            return True
        return False

    def _canonicalize_model_key(
        self, disk_token: str, allowed: Sequence[str]
    ) -> Optional[str]:
        """
        Map a token found in on-disk directory names back to a canonical model key.

        This handles common cases where:
        - Older runs used a different alias key (e.g., ``gpt4-o``) while the current
          experiment config expects ``gpt4``.
        - Directory tokens are already normalized and must be matched by
          normalization.
        - Multiple keys point at the same model config path, and we prefer the
          key that is actually used in this experiment.

        Args:
            disk_token: Token extracted from directory names.
            allowed: Canonical keys allowed for this experiment (e.g., use_models or use_judges).

        Returns:
            Canonical key if resolvable, otherwise None.
        """
        if not disk_token:
            return None

        # Fast path: exact match.
        if disk_token in allowed:
            return disk_token

        disk_norm = normalize_token(disk_token)

        # Match by normalization among allowed keys.
        allowed_norm_map: Dict[str, str] = {}
        for key in allowed:
            try:
                key_norm = normalize_token(key)
            except ValueError:
                continue
            # Keep first occurrence; ambiguity is handled below.
            allowed_norm_map.setdefault(key_norm, key)
        if disk_norm in allowed_norm_map:
            return allowed_norm_map[disk_norm]

        # If the disk token is a known key (or normalizes to one), use config-path
        # matching to map alias keys back onto the experiment's allowed keys.
        candidate_disk_keys: List[str] = []
        if disk_token in self.config.models:
            candidate_disk_keys.append(disk_token)
        else:
            # Try to find a model key whose normalized form matches the disk token.
            for key in self.config.models.keys():
                try:
                    if normalize_token(key) == disk_norm:
                        candidate_disk_keys.append(key)
                except ValueError:
                    continue

        for disk_key in candidate_disk_keys:
            disk_cfg = self.config.models.get(disk_key, {})
            disk_cfg_path = disk_cfg.get("config")
            if not disk_cfg_path:
                continue
            matches = [
                key
                for key in allowed
                if self.config.models.get(key, {}).get("config") == disk_cfg_path
            ]
            if len(matches) == 1:
                return matches[0]

        return None

    def _cleanup_empty_dirs(self, root: Path) -> None:
        """
        Remove empty directories under the given root, bottom-up.

        Args:
            root: Directory to clean.
        """
        if not root.exists():
            return

        # Walk bottom-up to safely remove empty directories
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    if not any(path.iterdir()):
                        path.rmdir()
                        logger.debug("Removed empty directory: %s", path)
                except OSError:
                    # Directory not empty or cannot be removed
                    continue

        # Finally try the root itself
        try:
            if root.is_dir() and not any(root.iterdir()):
                root.rmdir()
                logger.debug("Removed empty directory: %s", root)
        except OSError:
            pass
