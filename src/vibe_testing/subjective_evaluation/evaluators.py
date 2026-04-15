import logging
from typing import Any, Dict, List

from tqdm import tqdm

from .aggregators import ScoreAggregator
from .model_judges import ModelJudge
from .rubric_scorers import RubricScorer
from .schemas import (
    ModelGeneration,
    StaticEvaluationResult,
    SubjectiveEvaluationResult,
    UserProfile,
)
from .vibe_dimensions import VibeDimension, VibeDimensionEvaluator
from src.vibe_testing.models.base import BaseModel
from src.vibe_testing.evaluation.vibe_text_metrics import (
    compute_vibe_text_metrics,
    group_vibe_text_metrics_by_dimension,
)

logger = logging.getLogger(__name__)


class SubjectiveVibeEvaluator:
    """
    Persona-aware evaluator that scores clarity, tone/style fit, efficiency,
    cognitive load, context awareness, persona consistency, and frustration.
    """

    def __init__(self, judge_model: BaseModel, config: Dict[str, Any] = None):
        """
        Args:
            judge_model (BaseModel): Model used as an LLM judge.
            config (Dict[str, Any], optional): Optional configuration dictionary.
                May contain a ``generation_kwargs`` dict that is forwarded to
                the underlying model.generate calls (e.g., to set a seed).
        """
        self.config = config or {}
        generation_kwargs = self.config.get("generation_kwargs") or {}
        self.model_judge = ModelJudge(judge_model, generation_kwargs=generation_kwargs)
        rubric_config = self.config.get("rubric", {}) or {}
        disable_rubric = bool(self.config.get("disable_rubric", False))
        if disable_rubric:
            rubric_config = {**rubric_config, "enabled": False}
        rubric_enabled = bool(rubric_config.get("enabled", True))
        # When rubric is disabled, set to None so no rubric metadata/components are emitted.
        self.rubric_scorer = RubricScorer(rubric_config) if rubric_enabled else None
        self.dimension_evaluator = VibeDimensionEvaluator(
            self.model_judge, self.rubric_scorer
        )
        self.aggregator = ScoreAggregator(self.config.get("aggregation", {}))
        self.batch_size = max(1, int(self.config.get("batch_size", 1)))
        # Indicators are deterministic/non-LLM. Some pipelines want them split out.
        self.include_indicators = bool(self.config.get("include_indicators", True))

    def evaluate(
        self,
        user_profile: UserProfile,
        model_generation: ModelGeneration,
        static_result: StaticEvaluationResult,
    ) -> SubjectiveEvaluationResult:
        """
        Evaluate a single model generation against the provided user profile.

        Args:
            user_profile (UserProfile): Persona/context definition.
            model_generation (ModelGeneration): Input/output pair to score.
            static_result (StaticEvaluationResult): Static correctness outcome.

        Returns:
            SubjectiveEvaluationResult: Detailed vibe evaluation.
        """
        dimension_scores: Dict[str, float] = {}
        dimension_breakdowns: Dict[str, Dict[str, Any]] = {}

        for dimension in VibeDimension:
            result = self.dimension_evaluator.evaluate_dimension(
                dimension, user_profile, model_generation, static_result
            )
            score = _clamp(result["score"])
            dimension_scores[dimension.value] = score
            dimension_breakdowns[dimension.value] = result["details"]

        overall_score = self.aggregator.aggregate_subjective_scores(
            dimension_scores, user_profile=user_profile
        )
        combined_score = self.aggregator.combine_with_static_scores(
            overall_score, static_result.correctness_score
        )

        input_indicators: Dict[str, float] = {}
        output_indicators: Dict[str, float] = {}
        vibe_text_metrics: Dict[str, Any] = {}
        vibe_text_metrics_flat: Dict[str, Any] = {}
        if self.include_indicators:
            input_indicators = self.aggregator.compute_input_side_indicators(
                user_profile, model_generation
            )
            output_indicators = self.aggregator.compute_output_side_indicators(
                user_profile, model_generation, static_result
            )
            persona_payload: Dict[str, Any] = dict(user_profile.persona or {})
            persona_payload.setdefault("user_id", user_profile.user_id)
            vibe_text_metrics_flat = compute_vibe_text_metrics(
                response_text=model_generation.generated_output or "",
                prompt_text=model_generation.input_text or None,
                persona=persona_payload,
            )
            vibe_text_metrics = group_vibe_text_metrics_by_dimension(
                vibe_text_metrics_flat
            )

        judge_metadata = {
            "model_name": getattr(self.model_judge.model, "model_name", "unknown"),
            "config": getattr(self.model_judge.model, "config", {}),
        }

        # Get dimension scores with backward compatibility
        # Prefer new dimension names, fall back to old names if new ones not present
        workflow_fit_score = dimension_scores.get(
            VibeDimension.WORKFLOW_FIT.value,
            dimension_scores.get(VibeDimension.EFFICIENCY.value, 0.0),
        )
        friction_score = dimension_scores.get(
            VibeDimension.FRICTION_LOSS_OF_CONTROL.value,
            dimension_scores.get(VibeDimension.FRUSTRATION.value, 0.0),
        )

        return SubjectiveEvaluationResult(
            user_id=user_profile.user_id,
            task_id=model_generation.task_id,
            model_name=model_generation.model_name,
            clarity_score=dimension_scores.get(VibeDimension.CLARITY.value, 0.0),
            tone_style_fit_score=dimension_scores.get(
                VibeDimension.TONE_STYLE_FIT.value, 0.0
            ),
            workflow_fit_score=workflow_fit_score,
            cognitive_load_score=dimension_scores.get(
                VibeDimension.COGNITIVE_LOAD.value, 0.0
            ),
            context_awareness_score=dimension_scores.get(
                VibeDimension.CONTEXT_AWARENESS.value, 0.0
            ),
            persona_consistency_score=dimension_scores.get(
                VibeDimension.PERSONA_CONSISTENCY.value, 0.0
            ),
            friction_loss_of_control_score=friction_score,
            reliability_user_trust_score=dimension_scores.get(
                VibeDimension.RELIABILITY_USER_TRUST.value, 0.0
            ),
            anthropomorphism_score=dimension_scores.get(
                VibeDimension.ANTHROPOMORPHISM.value, 0.0
            ),
            # Backward compatibility fields (populated from new fields)
            efficiency_score=workflow_fit_score,
            frustration_indicator=friction_score,
            overall_subjective_score=overall_score,
            combined_score=combined_score,
            dimension_breakdowns=dimension_breakdowns,
            judge_metadata=judge_metadata,
            input_side_indicators=input_indicators,
            output_side_indicators=output_indicators,
            vibe_text_metrics=vibe_text_metrics,
            vibe_text_metrics_flat=vibe_text_metrics_flat,
        )

    def evaluate_batch(
        self,
        user_profiles: List[UserProfile],
        model_generations: List[ModelGeneration],
        static_results: List[StaticEvaluationResult],
    ) -> List[SubjectiveEvaluationResult]:
        """
        Evaluate multiple generations in batch.

        Args:
            user_profiles (List[UserProfile]): Profiles to reference.
            model_generations (List[ModelGeneration]): Generations to evaluate.
            static_results (List[StaticEvaluationResult]): Static outcomes.

        Returns:
            List[SubjectiveEvaluationResult]: Results aligned with generations.
        """
        results: List[SubjectiveEvaluationResult] = []
        profile_map = {p.user_id: p for p in user_profiles}
        static_map = {r.task_id: r for r in static_results}

        with tqdm(total=len(model_generations), desc="Evaluating generations") as pbar:
            for start in range(0, len(model_generations), self.batch_size):
                batch_gens = model_generations[start : start + self.batch_size]
                batch_profiles = []
                batch_static = []
                # Align profiles and static results
                for generation in batch_gens:
                    profile = profile_map.get(generation.user_id)
                    static = static_map.get(generation.task_id)
                    if not profile or not static:
                        logger.warning(
                            "Missing profile or static result for user_id=%s task_id=%s",
                            generation.user_id,
                            generation.task_id,
                        )
                        continue
                    batch_profiles.append(profile)
                    batch_static.append(static)

                if not batch_profiles:
                    pbar.update(len(batch_gens))
                    continue

                # Dimension-wise batching
                dimension_scores_batch = [dict() for _ in batch_profiles]
                dimension_breakdowns_batch = [dict() for _ in batch_profiles]
                for dimension in VibeDimension:
                    dim_results = self.dimension_evaluator.evaluate_dimension_batch(
                        dimension, batch_profiles, batch_gens, batch_static
                    )
                    for idx, dim_result in enumerate(dim_results):
                        dimension_scores_batch[idx][dimension.value] = dim_result[
                            "score"
                        ]
                        dimension_breakdowns_batch[idx][dimension.value] = dim_result[
                            "details"
                        ]

                for idx, generation in enumerate(batch_gens):
                    if idx >= len(batch_profiles):
                        pbar.update(1)
                        continue
                    profile = batch_profiles[idx]
                    static = batch_static[idx]
                    dimension_scores = dimension_scores_batch[idx]
                    breakdowns = dimension_breakdowns_batch[idx]

                    overall_score = self.aggregator.aggregate_subjective_scores(
                        dimension_scores, user_profile=profile
                    )
                    combined_score = self.aggregator.combine_with_static_scores(
                        overall_score, static.correctness_score
                    )

                    input_indicators: Dict[str, float] = {}
                    output_indicators: Dict[str, float] = {}
                    vibe_text_metrics: Dict[str, Any] = {}
                    vibe_text_metrics_flat: Dict[str, Any] = {}
                    if self.include_indicators:
                        input_indicators = (
                            self.aggregator.compute_input_side_indicators(
                                profile, generation
                            )
                        )
                        output_indicators = (
                            self.aggregator.compute_output_side_indicators(
                                profile, generation, static
                            )
                        )
                        persona_payload: Dict[str, Any] = dict(profile.persona or {})
                        persona_payload.setdefault("user_id", profile.user_id)
                        vibe_text_metrics_flat = compute_vibe_text_metrics(
                            response_text=generation.generated_output or "",
                            prompt_text=generation.input_text or None,
                            persona=persona_payload,
                        )
                        vibe_text_metrics = group_vibe_text_metrics_by_dimension(
                            vibe_text_metrics_flat
                        )

                    judge_metadata = {
                        "model_name": getattr(
                            self.model_judge.model, "model_name", "unknown"
                        ),
                        "config": getattr(self.model_judge.model, "config", {}),
                    }

                    # Get dimension scores with backward compatibility
                    # Prefer new dimension names, fall back to old names if new ones not present
                    workflow_fit_score = dimension_scores.get(
                        VibeDimension.WORKFLOW_FIT.value,
                        dimension_scores.get(VibeDimension.EFFICIENCY.value, 0.0),
                    )
                    friction_score = dimension_scores.get(
                        VibeDimension.FRICTION_LOSS_OF_CONTROL.value,
                        dimension_scores.get(VibeDimension.FRUSTRATION.value, 0.0),
                    )

                    results.append(
                        SubjectiveEvaluationResult(
                            user_id=profile.user_id,
                            task_id=generation.task_id,
                            model_name=generation.model_name,
                            clarity_score=dimension_scores.get(
                                VibeDimension.CLARITY.value, 0.0
                            ),
                            tone_style_fit_score=dimension_scores.get(
                                VibeDimension.TONE_STYLE_FIT.value, 0.0
                            ),
                            workflow_fit_score=workflow_fit_score,
                            cognitive_load_score=dimension_scores.get(
                                VibeDimension.COGNITIVE_LOAD.value, 0.0
                            ),
                            context_awareness_score=dimension_scores.get(
                                VibeDimension.CONTEXT_AWARENESS.value, 0.0
                            ),
                            persona_consistency_score=dimension_scores.get(
                                VibeDimension.PERSONA_CONSISTENCY.value, 0.0
                            ),
                            friction_loss_of_control_score=friction_score,
                            reliability_user_trust_score=dimension_scores.get(
                                VibeDimension.RELIABILITY_USER_TRUST.value, 0.0
                            ),
                            anthropomorphism_score=dimension_scores.get(
                                VibeDimension.ANTHROPOMORPHISM.value, 0.0
                            ),
                            # Backward compatibility fields (populated from new fields)
                            efficiency_score=workflow_fit_score,
                            frustration_indicator=friction_score,
                            overall_subjective_score=overall_score,
                            combined_score=combined_score,
                            dimension_breakdowns=breakdowns,
                            judge_metadata=judge_metadata,
                            input_side_indicators=input_indicators,
                            output_side_indicators=output_indicators,
                            vibe_text_metrics=vibe_text_metrics,
                            vibe_text_metrics_flat=vibe_text_metrics_flat,
                        )
                    )
                    pbar.update(1)
        return results


def _clamp(value: float) -> float:
    """Clamp helper specific to evaluator scores."""
    return max(0.0, min(1.0, value))
