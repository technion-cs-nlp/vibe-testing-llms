"""
High-level orchestration utilities for evaluation strategies.
"""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.vibe_testing.evaluation.strategies.base import (
    EvaluationStrategy,
    PreparedInput,
    GenerationResult,
    ExecutionResult,
    StrategyResult,
)

import tqdm


class EvaluationOrchestrator:
    """
    Coordinates dataset iteration for a concrete EvaluationStrategy.
    """

    def __init__(
        self, strategy: EvaluationStrategy, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            strategy (EvaluationStrategy): Strategy responsible for the end-to-end
                evaluation workflow.
            logger (Optional[logging.Logger]): Optional logger for progress updates.
        """
        self.strategy = strategy
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def run(
        self,
        samples: Sequence[Dict[str, Any]],
        batch_size: int = 1,
        allowed_prompt_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the strategy for each dataset sample in sequence.

        Args:
            samples (Sequence[Dict[str, Any]]): Iterable of dataset samples already
                loaded from disk.

        Returns:
            Dict[str, Any]: A dictionary containing `results` (per-sample StrategyResult
                objects) and `summary` (strategy-specific aggregate output).
        """
        expanded_samples = self._expand_samples(samples)

        allowed_set = (
            {pt.lower() for pt in allowed_prompt_types}
            if allowed_prompt_types
            else None
        )
        if allowed_set is not None:
            filtered_samples: List[Dict[str, Any]] = []
            for sample in expanded_samples:
                prompt_type = self._infer_prompt_type(sample)
                if prompt_type in allowed_set:
                    filtered_samples.append(sample)
            expanded_samples = filtered_samples

        total_units = len(expanded_samples)
        results: List[StrategyResult] = []
        self.logger.info(f"Evaluating {total_units} samples...")
        batch_size = max(1, int(batch_size))
        pbar = tqdm.tqdm(total=total_units, desc="Evaluating samples")

        prepared_inputs: List[PreparedInput] = []
        skipped_results: List[StrategyResult] = []
        verification_failed_results: List[StrategyResult] = []
        verification_failed_units = 0
        verification_failed_units_loaded_from_disk = 0
        actual_samples_to_evaluate = 0
        for sample in expanded_samples:
            prepared = self._safe_prepare(sample)
            if prepared:
                # Check if result already exists to avoid redundant compute
                expected_path = self.strategy.get_expected_output_path(
                    prepared.sample_id
                )
                if expected_path and expected_path.exists():
                    try:
                        from src.vibe_testing.utils import load_json

                        data = load_json(str(expected_path))
                        # Basic validation that the file is complete
                        if "metrics" in data and "artifacts" in data:
                            self.logger.info(
                                "Skipping already evaluated sample: %s",
                                prepared.sample_id,
                            )
                            if bool(data.get("metrics", {}).get("verification_failed")):
                                verification_failed_units_loaded_from_disk += 1
                            skipped_results.append(
                                StrategyResult(
                                    sample_id=data["sample_id"],
                                    metrics=data["metrics"],
                                    artifacts=data["artifacts"],
                                )
                            )
                            pbar.update(1)
                            continue
                    except Exception as e:
                        self.logger.warning(
                            "Failed to load existing result for %s: %s. Re-evaluating.",
                            prepared.sample_id,
                            e,
                        )

                if self._should_skip_due_to_failed_verification(sample):
                    verification_failed_units += 1
                    verification_failed_results.append(
                        self._build_failed_verification_result(
                            prepared=prepared, sample=sample
                        )
                    )
                    pbar.update(1)
                    continue

                prepared_inputs.append(prepared)
                actual_samples_to_evaluate += 1

        self.logger.info(
            f"Actual samples to evaluate: {actual_samples_to_evaluate} / {total_units}"
        )

        for idx in range(0, len(prepared_inputs), batch_size):
            batch = prepared_inputs[idx : idx + batch_size]
            generated_batch = self._safe_generate_batch(batch)
            for generated in generated_batch:
                executed: Optional[ExecutionResult] = self._safe_execute(generated)
                if executed is None:
                    raise Exception(f"Failed to execute {generated.sample_id}")
                result = self._safe_collect(executed)
                if result is None:
                    raise Exception(
                        f"Failed to collect metrics for {generated.sample_id}"
                    )
                results.append(result)
                pbar.update(1)

        # Combine new and skipped results
        results.extend(skipped_results)
        results.extend(verification_failed_results)

        self.logger.info(
            "Completed evaluation of %d evaluation units (%d new, %d skipped, %d failed_verification). "
            "Actual samples to evaluate: %d / %d. Generating reports...",
            len(results),
            len(results) - len(skipped_results) - len(verification_failed_results),
            len(skipped_results),
            len(verification_failed_results),
            actual_samples_to_evaluate,
            total_units,
        )

        # count overall failures
        overall_failures = 0
        for result in results:
            if bool(result.metrics.get("verification_failed", False)):
                continue
            if result.metrics.get("overall_failure", False):
                overall_failures += 1
        self.logger.info(f"Overall failures: {overall_failures}")
        if verification_failed_units or verification_failed_units_loaded_from_disk:
            self.logger.info(
                "Verification-gated units: %d newly marked failed_verification, %d loaded from disk.",
                verification_failed_units,
                verification_failed_units_loaded_from_disk,
            )

        # If we were restricted to a single prompt type, propagate that label
        # so the strategy can customize summary filenames.
        report_prompt_type = None
        if allowed_prompt_types and len(allowed_prompt_types) == 1:
            report_prompt_type = allowed_prompt_types[0]

        # Backward-compatible report() invocation:
        # older strategy implementations may not accept the prompt_type keyword.
        report_sig = inspect.signature(self.strategy.report)
        if "prompt_type" in report_sig.parameters:
            summary = self.strategy.report(results, prompt_type=report_prompt_type)
        else:
            summary = self.strategy.report(results)
        if isinstance(summary, dict):
            summary.setdefault(
                "verification_failed_units",
                int(
                    verification_failed_units
                    + verification_failed_units_loaded_from_disk
                ),
            )
        return {"results": results, "summary": summary}

    @staticmethod
    def _extract_verification_payload(
        sample: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract the verification payload from a variation entry when present.

        Args:
            sample (Dict[str, Any]): Expanded sample record (may include active_variation).

        Returns:
            Optional[Dict[str, Any]]: The verification payload if present, otherwise None.
        """
        if not isinstance(sample, dict):
            return None
        active_variation = sample.get("active_variation")
        if not isinstance(active_variation, dict):
            return None
        verification = active_variation.get("verification")
        if verification is None:
            return None
        if not isinstance(verification, dict):
            raise ValueError(
                "Invalid variation verification payload: expected dict with keys "
                "same_end_goal/same_ground_truth."
            )
        return verification

    def _should_skip_due_to_failed_verification(self, sample: Dict[str, Any]) -> bool:
        """
        Decide whether to skip evaluation due to Stage-3 verification failure.

        We only skip evaluation units that correspond to personalized prompt variations
        whose verification booleans indicate the end-goal or ground-truth changed.

        Args:
            sample (Dict[str, Any]): Expanded dataset sample entry.

        Returns:
            bool: True if the unit should be marked as failed_verification and skipped.
        """
        if not self._is_variation(sample):
            return False
        if self._infer_prompt_type(sample) != "personalized":
            return False
        verification = self._extract_verification_payload(sample)
        if verification is None:
            # Backward compatibility: older datasets may omit verification.
            return False
        if (
            "same_end_goal" not in verification
            or "same_ground_truth" not in verification
        ):
            raise ValueError(
                "Variation verification payload missing required keys. "
                "Expected 'same_end_goal' and 'same_ground_truth'."
            )
        same_end_goal = bool(verification["same_end_goal"])
        same_ground_truth = bool(verification["same_ground_truth"])
        return not (same_end_goal and same_ground_truth)

    def _build_failed_verification_result(
        self, prepared: PreparedInput, sample: Dict[str, Any]
    ) -> StrategyResult:
        """
        Construct a StrategyResult that represents a verification-gated unit.

        Args:
            prepared (PreparedInput): Prepared input containing a resolved sample_id.
            sample (Dict[str, Any]): Expanded dataset sample entry.

        Returns:
            StrategyResult: Synthetic result flagged as verification_failed.
        """
        verification = self._extract_verification_payload(sample) or {}
        metrics: Dict[str, Any] = {
            "verification_failed": True,
            "skip_reason": "failed_verification",
            "verification": verification,
        }

        artifacts: Dict[str, Any] = {
            # Downstream stages expect artifacts to exist for per-sample JSON payloads.
            "records": [],
            "verification": verification,
            "sample_metadata": {},
        }

        # Best-effort extraction of metadata from strategy-prepared payload.
        try:
            sample_obj = prepared.payload.get("sample")
            sample_metadata = getattr(sample_obj, "metadata", None)
            if isinstance(sample_metadata, dict):
                artifacts["sample_metadata"] = dict(sample_metadata)
            tests_obj = getattr(sample_obj, "tests", None)
            if tests_obj is not None and hasattr(tests_obj, "model_dump"):
                artifacts["tests"] = tests_obj.model_dump()
            prompt = getattr(sample_obj, "prompt", None)
            if isinstance(prompt, str):
                artifacts["prompt"] = prompt
        except Exception:
            # Keep artifacts minimal if strategy payload shape is unexpected.
            pass

        # Ensure variant_label exists for downstream filtering.
        artifacts["sample_metadata"].setdefault("variant_label", "personalized")
        artifacts["sample_metadata"].setdefault("verification", verification)

        return StrategyResult(
            sample_id=prepared.sample_id, metrics=metrics, artifacts=artifacts
        )

    def _expand_samples(
        self, samples: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Flatten original samples plus their prompt variations into evaluation units.
        """
        expanded: List[Dict[str, Any]] = []
        for sample in samples:
            expanded.extend(self._expand_single_sample(sample))
        return expanded

    def _expand_single_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Duplicate a PersonalizedSample for each variation so strategies receive
        one logical unit per prompt.
        """
        if not isinstance(sample, dict):
            return [sample]

        original_sample = sample.get("original_sample")
        raw_variations = sample.get("variations")
        if not original_sample:
            return [sample]
        if not isinstance(raw_variations, Sequence) or not raw_variations:
            base_entry = deepcopy(sample)
            base_entry["active_variation"] = None
            return [base_entry]

        expanded: List[Dict[str, Any]] = []
        base_entry = deepcopy(sample)
        base_entry["active_variation"] = None
        expanded.append(base_entry)

        for variation in raw_variations:
            variant_entry: Dict[str, Any] = {}
            for key, value in sample.items():
                if key == "variations":
                    continue
                variant_entry[key] = deepcopy(value)
            variant_entry["active_variation"] = deepcopy(variation)
            expanded.append(variant_entry)
        return expanded

    def _describe_sample(self, sample: Dict[str, Any]) -> str:
        """
        Build a human-readable identifier for logging.
        """
        if not isinstance(sample, dict):
            return str(sample)
        if "active_variation" in sample and sample.get("active_variation"):
            base_id = (sample.get("original_sample") or {}).get("sample_id")
            variation_id = sample["active_variation"].get("variation_id")
            if base_id and variation_id:
                return f"{base_id}::{variation_id}"
            return variation_id or base_id or "variation"
        return (
            sample.get("sample_id")
            or sample.get("instance_id")
            or (sample.get("original_sample") or {}).get("sample_id")
            or "sample"
        )

    def _is_variation(self, sample: Dict[str, Any]) -> bool:
        """
        Determine whether the provided sample corresponds to a prompt variation.
        """
        return bool(isinstance(sample, dict) and sample.get("active_variation"))

    def _infer_prompt_type(self, sample: Dict[str, Any]) -> str:
        """
        Infer a coarse prompt type for an expanded sample.

        Returns one of: ``original``, ``personalized``, or ``control``.
        """
        if not isinstance(sample, dict):
            return "original"

        active_variation = sample.get("active_variation")
        if not active_variation:
            # No active variation: this is the base/original prompt.
            return "original"

        # Try explicit labels on the variation first.
        variation_meta = active_variation if isinstance(active_variation, dict) else {}
        label = (
            str(
                variation_meta.get("prompt_type")
                or variation_meta.get("variant_label")
                or ""
            )
            .strip()
            .lower()
        )
        if label in {"original", "base"}:
            return "original"
        if label in {"control"}:
            return "control"
        if label:
            # Anything else that is explicitly labeled is treated as personalized.
            return "personalized"

        # Fall back to variation_id heuristics.
        variation_id = str(variation_meta.get("variation_id") or "").lower()
        if variation_id.startswith("control") or variation_id.endswith("::control"):
            return "control"

        # Default for any other active variation.
        return "personalized"

    def _safe_prepare(self, sample: Dict[str, Any]) -> Optional[PreparedInput]:
        try:
            return self.strategy.prepare_inputs(sample)
        except Exception as exc:
            self.logger.exception("Strategy failed while preparing inputs: %s", exc)
            # return None
            raise exc

    def _safe_generate(self, prepared: PreparedInput) -> Optional[GenerationResult]:
        try:
            return self.strategy.run_generation(prepared)
        except Exception as exc:
            self.logger.exception(
                "Strategy failed while generating outputs for %s: %s",
                prepared.sample_id,
                exc,
            )
            raise exc

    def _safe_generate_batch(
        self, prepared_batch: List[PreparedInput]
    ) -> List[GenerationResult]:
        try:
            return self.strategy.run_generation_batch(prepared_batch)
        except Exception as exc:
            self.logger.exception("Strategy failed while generating batch: %s", exc)
            raise exc

    def _safe_execute(self, generated: GenerationResult) -> Optional[ExecutionResult]:
        try:
            return self.strategy.execute(generated)
        except Exception as exc:
            self.logger.exception(
                "Strategy failed while executing sample %s: %s",
                generated.sample_id,
                exc,
            )
            return None

    def _safe_collect(self, executed: ExecutionResult) -> Optional[StrategyResult]:
        try:
            return self.strategy.collect_metrics(executed)
        except Exception as exc:
            self.logger.exception(
                "Strategy failed while collecting metrics for %s: %s",
                executed.sample_id,
                exc,
            )
            return None
