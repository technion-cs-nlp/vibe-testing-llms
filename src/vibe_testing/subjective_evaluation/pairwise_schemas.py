"""
Data structures for pairwise comparison evaluation.

This module defines the schemas used for storing and processing
pairwise comparison results between two model outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ComparisonWinner(Enum):
    """Outcome of a pairwise comparison."""

    MODEL_A = "A"
    MODEL_B = "B"
    TIE = "tie"


class ComparisonConfidence(Enum):
    """Confidence level of the judge's decision."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PairwiseComparisonInput:
    """
    Input structure for a single pairwise comparison.

    Attributes:
        task_id: Unique identifier for the task/question.
        input_text: The original task/question prompt.
        model_a_name: Name of the first model.
        model_a_output: Generated output from model A.
        model_b_name: Name of the second model.
        model_b_output: Generated output from model B.
        metadata: Additional context about the comparison.
    """

    task_id: str
    input_text: str
    model_a_name: str
    model_a_output: str
    model_b_name: str
    model_b_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SingleOrderResult:
    """
    Result from a single comparison order (before position swap aggregation).

    Attributes:
        position_winner: Which position won (A or B) in this specific order.
        confidence: Confidence level of the judgment.
        rationale: Explanation for the decision.
        raw_response: Full response from the judge model.
    """

    position_winner: ComparisonWinner
    confidence: ComparisonConfidence
    rationale: str
    raw_response: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "position_winner": self.position_winner.value,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "raw_response": self.raw_response,
        }


@dataclass
class DimensionComparisonResult:
    """
    Result of comparing two outputs on a single vibe dimension.

    Includes results from both position orders to detect and mitigate
    position bias. The final winner is determined by checking if both
    orders agree on the same actual model.

    Attributes:
        dimension: The vibe dimension being compared.
        winner: Final winner after position bias check (A, B, or tie).
        confidence: Confidence level of the judgment.
        rationale: Explanation for the decision.
        raw_response: Full response from the judge model (original order).
        model_a_name: Name of model A for reference.
        model_b_name: Name of model B for reference.
        original_order_result: Result with model_a at position A.
        swapped_order_result: Result with model_a at position B.
        position_bias_detected: Whether the judge changed decision on swap.
    """

    dimension: str
    winner: ComparisonWinner
    confidence: ComparisonConfidence
    rationale: str
    raw_response: str
    model_a_name: str
    model_b_name: str
    original_order_result: Optional[SingleOrderResult] = None
    swapped_order_result: Optional[SingleOrderResult] = None
    position_bias_detected: bool = False

    def get_winner_name(self) -> Optional[str]:
        """Get the name of the winning model, or None for ties."""
        if self.winner == ComparisonWinner.MODEL_A:
            return self.model_a_name
        elif self.winner == ComparisonWinner.MODEL_B:
            return self.model_b_name
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "dimension": self.dimension,
            "winner": self.winner.value,
            "winner_model": self.get_winner_name(),
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "raw_response": self.raw_response,
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "position_bias_detected": self.position_bias_detected,
        }
        if self.original_order_result:
            result["original_order_result"] = self.original_order_result.to_dict()
        if self.swapped_order_result:
            result["swapped_order_result"] = self.swapped_order_result.to_dict()
        return result


@dataclass
class PairwiseComparisonResult:
    """
    Complete result of a pairwise comparison across all vibe dimensions.

    Attributes:
        user_id: User profile identifier.
        task_id: Task/question identifier.
        model_a_name: Name of the first model.
        model_b_name: Name of the second model.
        model_a_output: Generated output from model A.
        model_b_output: Generated output from model B.
        dimension_results: Results for each dimension comparison.
        overall_winner: Aggregated winner across all dimensions.
        win_counts: Count of wins for each model per dimension.
        judge_model_name: Name of the model used for judging.
        input_text: Original task input.
        pairwise_judgment_type: Judgment mode used for the comparison.
        metadata: Additional context.
    """

    user_id: str
    task_id: str
    model_a_name: str
    model_b_name: str
    model_a_output: str
    model_b_output: str
    dimension_results: Dict[str, DimensionComparisonResult]
    overall_winner: Optional[str]
    win_counts: Dict[str, int]
    judge_model_name: str
    input_text: str
    pairwise_judgment_type: str = "persona"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "user_id": self.user_id,
            "task_id": self.task_id,
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "model_a_output": self.model_a_output,
            "model_b_output": self.model_b_output,
            "dimension_results": {
                dim: result.to_dict() for dim, result in self.dimension_results.items()
            },
            "overall_winner": self.overall_winner,
            "win_counts": self.win_counts,
            "judge_model_name": self.judge_model_name,
            "input_text": self.input_text,
            "pairwise_judgment_type": self.pairwise_judgment_type,
            "metadata": self.metadata,
        }


@dataclass
class ModelPair:
    """
    A pair of models to compare.

    Attributes:
        model_a: Name of the first model.
        model_b: Name of the second model.
        model_a_config: Path to model A's config file.
        model_b_config: Path to model B's config file.
    """

    model_a: str
    model_b: str
    model_a_config: Optional[str] = None
    model_b_config: Optional[str] = None

    def __hash__(self):
        """Order-independent hash for deduplication."""
        return hash(frozenset([self.model_a, self.model_b]))

    def __eq__(self, other):
        """Check equality independent of order."""
        if not isinstance(other, ModelPair):
            return False
        return {self.model_a, self.model_b} == {other.model_a, other.model_b}

    def reversed(self) -> "ModelPair":
        """Return a new ModelPair with A and B swapped."""
        return ModelPair(
            model_a=self.model_b,
            model_b=self.model_a,
            model_a_config=self.model_b_config,
            model_b_config=self.model_a_config,
        )
