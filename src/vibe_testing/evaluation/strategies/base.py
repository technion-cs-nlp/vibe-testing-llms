"""
Abstract strategy definitions for multi-paradigm code evaluation.

These classes allow the stage-4 evaluation script to operate agnostically
across drastically different evaluation paradigms (e.g., MBPP+ style
function testing vs. SWE-Bench repository patch validation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from src.vibe_testing.models.base import BaseModel


@dataclass
class EvaluationContext:
    """
    Runtime context shared by an evaluation strategy during a pipeline run.
    """

    run_dir: str
    output_dir: str
    temp_dir: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedInput:
    """
    Represents the normalized inputs a strategy needs before generation.
    """

    sample_id: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """
    Captures one or more generations produced for a sample.
    """

    sample_id: str
    generations: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """
    Holds the execution/evaluation artifacts derived from generations.
    """

    sample_id: str
    artifacts: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """
    Final result persisted for a single sample once reporting completes.
    """

    sample_id: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)


class EvaluationStrategy(ABC):
    """
    Defines the interface every evaluation paradigm must implement.
    """

    def __init__(
        self, context: EvaluationContext, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the strategy.

        Args:
            context (EvaluationContext): Run-specific context, including
                output directories and any auxiliary configuration.
            logger (Optional[logging.Logger]): Optional logger; defaults to a
                module-level logger named after the strategy class.
        """
        self.context = context
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._model: Optional[BaseModel] = None

    def bind_model(self, model: BaseModel) -> None:
        """
        Attach the model instance used for inference.

        Args:
            model (BaseModel): The language model responsible for generation.
        """
        self._model = model

    @property
    def model(self) -> BaseModel:
        """
        Returns the currently bound model and raises if missing.
        """
        if self._model is None:
            msg = "Strategy attempted to access model before bind_model was called."
            self.logger.error(msg)
            raise RuntimeError(msg)
        return self._model

    @abstractmethod
    def prepare_inputs(self, sample: Dict[str, Any]) -> PreparedInput:
        """
        Normalize a raw dataset sample into the structure required by the strategy.
        """

    @abstractmethod
    def run_generation(self, prepared: PreparedInput) -> GenerationResult:
        """
        Produce one or many generations for the prepared sample.
        """

    def run_generation_batch(
        self, prepared: List[PreparedInput]
    ) -> List[GenerationResult]:
        """
        Produce generations for a batch of prepared samples.

        Default implementation calls run_generation per item for compatibility.
        Strategies can override for true batching.
        """
        return [self.run_generation(item) for item in prepared]

    @abstractmethod
    def execute(self, generated: GenerationResult) -> ExecutionResult:
        """
        Execute the generations (e.g., run tests, format patches) and gather artifacts.
        """

    @abstractmethod
    def collect_metrics(self, executed: ExecutionResult) -> StrategyResult:
        """
        Transform execution artifacts into persisted per-sample metrics.
        """

    @abstractmethod
    def report(
        self, results: List[StrategyResult], prompt_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Produce aggregated metrics and write any summary artifacts.
        """

    def get_expected_output_path(self, sample_id: str) -> Optional[Path]:
        """
        Return the Path where the result for a specific sample_id would be stored.
        Returning None means the strategy doesn't support skip-if-exists check.
        """
        return None
