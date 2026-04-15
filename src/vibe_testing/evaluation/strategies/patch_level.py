"""
Strategy for repository-level patch validation (Paradigm B).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vibe_testing.data_utils import PatchEvaluationSample
from src.vibe_testing.evaluation.strategies.base import (
    EvaluationContext,
    EvaluationStrategy,
    ExecutionResult,
    GenerationResult,
    PreparedInput,
    StrategyResult,
)
from src.vibe_testing.harness.patch_runner import PatchHarnessInvoker
from src.vibe_testing.reporting.resolution import ResolutionReporter
from src.vibe_testing.utils import save_jsonl
from src.vibe_testing.pathing import format_artifact_name, normalize_token


class PatchLevelStrategy(EvaluationStrategy):
    """
    Implements the repository-level patch evaluation workflow.
    """

    def __init__(
        self,
        context: EvaluationContext,
        harness_config: Optional[Dict[str, Any]] = None,
        harness_invoker: Optional[PatchHarnessInvoker] = None,
        reporter: Optional[ResolutionReporter] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(context, logger=logger)
        config = harness_config or context.extra.get("patch_harness")
        if config is None:
            raise ValueError("PatchLevelStrategy requires a harness configuration.")
        self.harness_invoker = harness_invoker or PatchHarnessInvoker(config, self.logger)
        self.resolution_reporter = reporter or ResolutionReporter(self.logger)
        self._predictions: List[Dict[str, Any]] = []
        self._prediction_entries_dir = Path(self.context.output_dir) / "patch_eval"
        self._prediction_entries_dir.mkdir(parents=True, exist_ok=True)

    def prepare_inputs(self, sample: Dict[str, Any]) -> PreparedInput:
        """
        Validate the SWE-Bench instance.
        """
        normalized_sample = PatchEvaluationSample(**sample)
        payload = {"sample": normalized_sample}
        return PreparedInput(sample_id=normalized_sample.instance_id, payload=payload)

    def run_generation(self, prepared: PreparedInput) -> GenerationResult:
        """
        Generate a single patch for the instance prompt.
        """
        sample: PatchEvaluationSample = prepared.payload["sample"]
        patch_text = self.model.generate(sample.prompt)
        generations = [{"patch": patch_text, "prompt": sample.prompt}]
        metadata = {"sample": sample}
        return GenerationResult(sample_id=sample.instance_id, generations=generations, metadata=metadata)

    def execute(self, generated: GenerationResult) -> ExecutionResult:
        """
        Format the model output into the harness prediction schema.
        """
        sample_obj = generated.metadata.get("sample")
        if not isinstance(sample_obj, PatchEvaluationSample):
            raise ValueError("PatchLevelStrategy expected sample metadata during execution.")
        sample: PatchEvaluationSample = sample_obj
        patch_text = generated.generations[0]["patch"]
        prediction_entry = {
            "instance_id": sample.instance_id,
            "model_patch": patch_text,
            "model_name_or_path": self.model.model_name,
        }
        artifacts = {"prediction_entry": prediction_entry, "raw_patch": patch_text}
        return ExecutionResult(sample_id=sample.instance_id, artifacts=artifacts, metrics={})

    def collect_metrics(self, executed: ExecutionResult) -> StrategyResult:
        """
        Store the prediction entry for later harness execution.
        """
        prediction_entry = executed.artifacts["prediction_entry"]
        self._predictions.append(prediction_entry)
        metrics = {"prediction_ready": True}
        return StrategyResult(sample_id=executed.sample_id, metrics=metrics, artifacts=executed.artifacts)

    def report(
        self, results: List[StrategyResult], prompt_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Write predictions JSONL, run the harness, and summarize resolution metrics.
        """
        detail = f"model-{normalize_token(self.model.model_name)}"
        if prompt_type:
            detail = f"{detail}-{prompt_type}"
        run_name = self.context.extra.get("run_name", "no_name") if self.context.extra else "no_name"
        predictions_name = format_artifact_name(
            artifact_type="patch_eval",
            evaluation_type="predictions",
            detail=detail,
            version=0,
            ext="jsonl",
            run_name=run_name,
        )
        predictions_path = self._prediction_entries_dir / predictions_name
        save_jsonl(self._predictions, str(predictions_path))
        harness_output = self.harness_invoker.run(str(predictions_path))
        resolution_summary = {}
        if harness_output.get("results_path"):
            resolution_summary = self.resolution_reporter.summarize(harness_output["results_path"])
        return {
            "predictions_path": predictions_path,
            "harness": harness_output,
            "resolution": resolution_summary,
        }


