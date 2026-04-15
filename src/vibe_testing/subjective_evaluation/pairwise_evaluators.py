"""
Pairwise comparison evaluator for vibe dimensions.

This module orchestrates pairwise comparisons between model outputs,
aggregating results across all vibe dimensions. Supports position bias
mitigation by running comparisons in both orders (A vs B and B vs A).
"""

import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .pairwise_judges import BasePairwiseJudge, PairwiseJudge
from .pairwise_schemas import (
    ComparisonConfidence,
    ComparisonWinner,
    DimensionComparisonResult,
    PairwiseComparisonInput,
    PairwiseComparisonResult,
    SingleOrderResult,
)
from .schemas import UserProfile
from .vibe_dimensions import VibeDimension
from src.vibe_testing.models.base import BaseModel

logger = logging.getLogger(__name__)


class PairwiseVibeEvaluator:
    """
    Orchestrates pairwise comparison evaluation across vibe dimensions.

    Compares two model outputs and determines which better serves the
    user persona for each dimension, then aggregates to an overall winner.

    Supports position bias mitigation by running each comparison twice
    with swapped positions. If the judge's decision changes when positions
    are swapped, the result is marked as a tie (position bias detected).
    """

    def __init__(
        self,
        judge_model: BaseModel,
        config: Dict[str, Any] | None = None,
        judge: Optional[BasePairwiseJudge] = None,
    ):
        """
        Initialize the pairwise evaluator.

        Args:
            judge_model (BaseModel): Model to use as judge.
            config (Dict[str, Any], optional): Configuration options.
                - use_position_swap (bool): Enable position bias mitigation.
                  Default: True.
                - dimensions (list): Dimensions to evaluate.
                - generation_kwargs (dict): Parameters for judge generation.
            judge (Optional[BasePairwiseJudge]): Optional pre-built judge
                implementation. When omitted, the default persona-aware
                PairwiseJudge is used.
        """
        self.config = config or {}
        generation_kwargs = self.config.get("generation_kwargs", {})
        self.judge = judge or PairwiseJudge(
            judge_model, generation_kwargs=generation_kwargs
        )
        self.dimensions_to_evaluate = self.config.get(
            "dimensions",
            list(VibeDimension),
        )
        self.judge_model_name = getattr(judge_model, "model_name", "unknown_judge")
        # Position swap is enabled by default for position bias mitigation
        self.use_position_swap = self.config.get("use_position_swap", True)
        # Tie breaker mode: "strict" (default) or "finegrained"
        # strict: requires both orders to agree (current behavior)
        # finegrained: uses confidence scores to break ties when orders disagree
        self.tie_breaker_mode = self.config.get("tie_breaker_mode", "strict")

    def evaluate(
        self,
        user_profile: UserProfile,
        comparison_input: PairwiseComparisonInput,
    ) -> PairwiseComparisonResult:
        """
        Perform pairwise comparison across all vibe dimensions.

        If position swap is enabled, each dimension is evaluated twice:
        once with model_a at position A, and once with model_a at position B.
        If the judge picks different actual models in the two runs, the result
        is a tie (position bias detected).

        Args:
            user_profile: Persona context for evaluation.
            comparison_input: The two model outputs to compare.

        Returns:
            PairwiseComparisonResult: Complete comparison results.
        """
        dimension_results: Dict[str, DimensionComparisonResult] = {}

        for dimension in self.dimensions_to_evaluate:
            try:
                if self.use_position_swap:
                    result = self._evaluate_dimension_with_swap(
                        dimension, user_profile, comparison_input
                    )
                else:
                    result = self.judge.compare_dimension(
                        dimension, user_profile, comparison_input
                    )
                dimension_results[dimension.value] = result
            except Exception as exc:
                logger.error(
                    "Failed to evaluate dimension %s: %s for comparsion model %s and %s input length %d",
                    dimension.value,
                    exc,
                    comparison_input.model_a_name,
                    comparison_input.model_b_name,
                    len(comparison_input.model_a_output)
                    + len(comparison_input.model_b_output),
                )
                # Record error as a tie with low confidence
                dimension_results[dimension.value] = DimensionComparisonResult(
                    dimension=dimension.value,
                    winner=ComparisonWinner.TIE,
                    confidence=ComparisonConfidence.LOW,
                    rationale=f"Evaluation error: {exc}",
                    raw_response="",
                    model_a_name=comparison_input.model_a_name,
                    model_b_name=comparison_input.model_b_name,
                )

        # if all dimensions failed, raise an error
        if len(dimension_results) == 0:
            raise ValueError("All dimensions failed to evaluate")

        # Aggregate results
        win_counts = self._compute_win_counts(dimension_results)
        overall_winner = self._determine_overall_winner(
            win_counts,
            comparison_input.model_a_name,
            comparison_input.model_b_name,
        )

        return PairwiseComparisonResult(
            user_id=user_profile.user_id,
            task_id=comparison_input.task_id,
            model_a_name=comparison_input.model_a_name,
            model_b_name=comparison_input.model_b_name,
            model_a_output=comparison_input.model_a_output,
            model_b_output=comparison_input.model_b_output,
            dimension_results=dimension_results,
            overall_winner=overall_winner,
            win_counts=win_counts,
            judge_model_name=self.judge_model_name,
            input_text=comparison_input.input_text,
            pairwise_judgment_type=self.config.get("pairwise_judgment_type", "persona"),
            metadata=comparison_input.metadata,
        )

    def _evaluate_dimension_with_swap(
        self,
        dimension: VibeDimension,
        user_profile: UserProfile,
        comparison_input: PairwiseComparisonInput,
    ) -> DimensionComparisonResult:
        """
        Evaluate a dimension with position swap for bias mitigation.

        Runs the comparison twice:
        1. Original order: model_a at position A, model_b at position B
        2. Swapped order: model_b at position A, model_a at position B

        If both runs agree on the same actual model, that model wins.
        If they disagree (position bias detected), the result is a tie.

        Args:
            dimension: The dimension to evaluate.
            user_profile: Persona context.
            comparison_input: Original comparison input.

        Returns:
            DimensionComparisonResult with aggregated results.
        """
        # Run 1: Original order (model_a at position A)
        original_result = self.judge.compare_dimension(
            dimension, user_profile, comparison_input
        )
        original_order = SingleOrderResult(
            position_winner=original_result.winner,
            confidence=original_result.confidence,
            rationale=original_result.rationale,
            raw_response=original_result.raw_response,
        )

        # Create swapped input (model_b at position A, model_a at position B)
        swapped_input = PairwiseComparisonInput(
            task_id=comparison_input.task_id,
            input_text=comparison_input.input_text,
            model_a_name=comparison_input.model_b_name,
            model_a_output=comparison_input.model_b_output,
            model_b_name=comparison_input.model_a_name,
            model_b_output=comparison_input.model_a_output,
            metadata=comparison_input.metadata,
        )

        # Run 2: Swapped order
        swapped_result = self.judge.compare_dimension(
            dimension, user_profile, swapped_input
        )
        swapped_order = SingleOrderResult(
            position_winner=swapped_result.winner,
            confidence=swapped_result.confidence,
            rationale=swapped_result.rationale,
            raw_response=swapped_result.raw_response,
        )

        # Determine actual model winner from each run
        # Original: position A = model_a, position B = model_b
        # Swapped: position A = model_b, position B = model_a
        original_model_winner = self._position_to_model_winner(
            original_order.position_winner, is_swapped=False
        )
        swapped_model_winner = self._position_to_model_winner(
            swapped_order.position_winner, is_swapped=True
        )

        # Check for consistency
        position_bias_detected = original_model_winner != swapped_model_winner

        if position_bias_detected:
            # Judges disagree - apply tie breaker logic
            if self.tie_breaker_mode == "finegrained":
                final_winner, final_confidence, final_rationale = (
                    self._finegrained_tie_breaker(
                        original_model_winner,
                        original_order.confidence,
                        original_result.rationale,
                        swapped_model_winner,
                        swapped_order.confidence,
                        swapped_result.rationale,
                        original_order.position_winner.value,
                        swapped_order.position_winner.value,
                    )
                )
                logger.debug(
                    "Fine-grained tie breaker for %s: original=%s (%s), swapped=%s (%s) -> %s",
                    dimension.value,
                    original_order.position_winner.value,
                    original_order.confidence.value,
                    swapped_order.position_winner.value,
                    swapped_order.confidence.value,
                    final_winner.value,
                )
            else:
                # Strict mode: mark as tie
                final_winner = ComparisonWinner.TIE
                final_rationale = (
                    f"Position bias detected: original order picked "
                    f"{original_order.position_winner.value}, swapped order picked "
                    f"{swapped_order.position_winner.value}. Marked as tie."
                )
                # Use lower confidence when bias detected
                final_confidence = ComparisonConfidence.LOW
                logger.debug(
                    "Position bias detected for %s: original=%s, swapped=%s",
                    dimension.value,
                    original_order.position_winner.value,
                    swapped_order.position_winner.value,
                )
        else:
            # Judges agree - use the consistent result
            final_winner = original_model_winner
            final_rationale = original_result.rationale
            final_confidence = original_result.confidence

        return DimensionComparisonResult(
            dimension=dimension.value,
            winner=final_winner,
            confidence=final_confidence,
            rationale=final_rationale,
            raw_response=original_result.raw_response,
            model_a_name=comparison_input.model_a_name,
            model_b_name=comparison_input.model_b_name,
            original_order_result=original_order,
            swapped_order_result=swapped_order,
            position_bias_detected=position_bias_detected,
        )

    def _finegrained_tie_breaker(
        self,
        original_model_winner: ComparisonWinner,
        original_confidence: ComparisonConfidence,
        original_rationale: str,
        swapped_model_winner: ComparisonWinner,
        swapped_confidence: ComparisonConfidence,
        swapped_rationale: str,
        original_position_winner: str,
        swapped_position_winner: str,
    ) -> tuple[ComparisonWinner, ComparisonConfidence, str]:
        """
        Break ties using confidence scores when orders disagree.

        If one order has higher confidence than the other, use that order's winner.
        If one order has a confident winner (medium or high) and the other is a tie,
        use the confident winner.
        If both have the same confidence, mark as tie.

        Args:
            original_model_winner: Winner from original order (model-based).
            original_confidence: Confidence from original order.
            original_rationale: Rationale from original order.
            swapped_model_winner: Winner from swapped order (model-based).
            swapped_confidence: Confidence from swapped order.
            swapped_rationale: Rationale from swapped order.
            original_position_winner: Position winner from original order (for logging).
            swapped_position_winner: Position winner from swapped order (for logging).

        Returns:
            Tuple of (final_winner, final_confidence, final_rationale).
        """
        # Confidence hierarchy: high > medium > low
        confidence_values = {
            ComparisonConfidence.HIGH: 3,
            ComparisonConfidence.MEDIUM: 2,
            ComparisonConfidence.LOW: 1,
        }

        original_conf_value = confidence_values.get(original_confidence, 1)
        swapped_conf_value = confidence_values.get(swapped_confidence, 1)

        # Case 1: One order has a confident winner (medium or high) and the other is a tie
        if (
            original_model_winner != ComparisonWinner.TIE
            and original_conf_value >= 2
            and swapped_model_winner == ComparisonWinner.TIE
        ):
            return (
                original_model_winner,
                original_confidence,
                f"Original order picked {original_position_winner} with {original_confidence.value} confidence, "
                f"swapped order marked as tie. Using original order result due to confident winner.",
            )
        if (
            swapped_model_winner != ComparisonWinner.TIE
            and swapped_conf_value >= 2
            and original_model_winner == ComparisonWinner.TIE
        ):
            return (
                swapped_model_winner,
                swapped_confidence,
                f"Swapped order picked {swapped_position_winner} with {swapped_confidence.value} confidence, "
                f"original order marked as tie. Using swapped order result due to confident winner.",
            )

        # Case 2: Both orders say tie
        if (
            original_model_winner == ComparisonWinner.TIE
            and swapped_model_winner == ComparisonWinner.TIE
        ):
            return (
                ComparisonWinner.TIE,
                ComparisonConfidence.LOW,
                f"Both orders marked as tie. Marked as tie.",
            )

        # Case 3: One order says tie but the other has low confidence winner
        if original_model_winner == ComparisonWinner.TIE:
            return (
                ComparisonWinner.TIE,
                ComparisonConfidence.LOW,
                f"Original order marked as tie. Swapped order picked {swapped_position_winner} with {swapped_confidence.value} confidence. Marked as tie.",
            )
        if swapped_model_winner == ComparisonWinner.TIE:
            return (
                ComparisonWinner.TIE,
                ComparisonConfidence.LOW,
                f"Swapped order marked as tie. Original order picked {original_position_winner} with {original_confidence.value} confidence. Marked as tie.",
            )

        # Case 4: Both have winners - compare confidence scores
        if original_conf_value > swapped_conf_value:
            # Original has higher confidence
            return (
                original_model_winner,
                original_confidence,
                f"Original order picked {original_position_winner} with {original_confidence.value} confidence, "
                f"swapped order picked {swapped_position_winner} with {swapped_confidence.value} confidence. "
                f"Using original order result due to higher confidence.",
            )
        elif swapped_conf_value > original_conf_value:
            # Swapped has higher confidence
            return (
                swapped_model_winner,
                swapped_confidence,
                f"Swapped order picked {swapped_position_winner} with {swapped_confidence.value} confidence, "
                f"original order picked {original_position_winner} with {original_confidence.value} confidence. "
                f"Using swapped order result due to higher confidence.",
            )
        else:
            # Same confidence - mark as tie
            return (
                ComparisonWinner.TIE,
                ComparisonConfidence.LOW,
                f"Position bias detected: original order picked {original_position_winner} with {original_confidence.value} confidence, "
                f"swapped order picked {swapped_position_winner} with {swapped_confidence.value} confidence. "
                f"Both have equal confidence, marked as tie.",
            )

    def _position_to_model_winner(
        self,
        position_winner: ComparisonWinner,
        is_swapped: bool,
    ) -> ComparisonWinner:
        """
        Convert position-based winner to model-based winner.

        In original order: position A = model_a, position B = model_b
        In swapped order: position A = model_b, position B = model_a

        Args:
            position_winner: Which position won (A, B, or tie).
            is_swapped: Whether this is from the swapped order run.

        Returns:
            Which actual model won (MODEL_A, MODEL_B, or TIE).
        """
        if position_winner == ComparisonWinner.TIE:
            return ComparisonWinner.TIE

        if is_swapped:
            # Swapped: position A = model_b, position B = model_a
            if position_winner == ComparisonWinner.MODEL_A:
                return ComparisonWinner.MODEL_B
            else:
                return ComparisonWinner.MODEL_A
        else:
            # Original: position A = model_a, position B = model_b
            return position_winner

    def evaluate_batch(
        self,
        user_profiles: List[UserProfile],
        comparison_inputs: List[PairwiseComparisonInput],
        batch_size: int = 1,
    ) -> List[PairwiseComparisonResult]:
        """
        Evaluate multiple comparisons in batch.

        Args:
            user_profiles: Profile for each comparison (or broadcast single).
            comparison_inputs: List of comparison inputs.

        Returns:
            List of PairwiseComparisonResult objects.
        """
        results: List[PairwiseComparisonResult] = []
        profile_map = {p.user_id: p for p in user_profiles}

        batch_size = max(1, int(batch_size))
        failures_count = {}
        with tqdm(total=len(comparison_inputs), desc="Pairwise comparisons") as pbar:
            for start in range(0, len(comparison_inputs), batch_size):
                batch = comparison_inputs[start : start + batch_size]
                for comp_input in batch:
                    user_id = comp_input.metadata.get("user_id")
                    profile = profile_map.get(user_id) or user_profiles[0]

                    try:
                        result = self.evaluate(profile, comp_input)
                        results.append(result)
                    except Exception as exc:
                        logger.error(
                            "Failed comparison for task %s: %s",
                            comp_input.task_id,
                            exc,
                        )
                        failures_count[comp_input.task_id] = (
                            failures_count.get(comp_input.task_id, 0) + 1
                        )
                    if failures_count.get(comp_input.task_id, 0) > 10:
                        logger.error(
                            "Stopping evaluation at task %s due to too many failures",
                            comp_input.task_id,
                        )
                        raise ValueError("Too many failures! Stopping evaluation.")
                    pbar.update(1)

        return results

    def _compute_win_counts(
        self,
        dimension_results: Dict[str, DimensionComparisonResult],
    ) -> Dict[str, int]:
        """
        Count wins for each model across dimensions.

        Args:
            dimension_results: Results from each dimension comparison.

        Returns:
            Dict with keys 'model_a', 'model_b', 'tie' and win counts.
        """
        counts = {"model_a": 0, "model_b": 0, "tie": 0}

        for result in dimension_results.values():
            if result.winner == ComparisonWinner.MODEL_A:
                counts["model_a"] += 1
            elif result.winner == ComparisonWinner.MODEL_B:
                counts["model_b"] += 1
            else:
                counts["tie"] += 1

        return counts

    def _determine_overall_winner(
        self,
        win_counts: Dict[str, int],
        model_a_name: str,
        model_b_name: str,
    ) -> Optional[str]:
        """
        Determine overall winner based on dimension win counts.

        Uses simple majority voting. Returns None if tied.

        Args:
            win_counts: Win counts per model.
            model_a_name: Name of model A.
            model_b_name: Name of model B.

        Returns:
            Name of winning model, or None for a tie.
        """
        if win_counts["model_a"] > win_counts["model_b"]:
            return model_a_name
        elif win_counts["model_b"] > win_counts["model_a"]:
            return model_b_name
        else:
            return None  # Overall tie
