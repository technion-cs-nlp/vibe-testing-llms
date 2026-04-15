import logging
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from .schemas import ModelGeneration, StaticEvaluationResult, UserProfile

if TYPE_CHECKING:  # pragma: no cover
    from .rubric_scorers import RubricScorer

logger = logging.getLogger(__name__)


class VibeDimension(Enum):
    """
    Enumerates the supported vibe dimensions. Higher scores are
    always better from the user's perspective.
    
    Current dimensions:
    - CLARITY: Measures how clear, structured, and readable the response is.
    - TONE_STYLE_FIT: Measures whether the writing style matches user preferences and task context.
    - WORKFLOW_FIT: Measures how well the response fits the user's workflow in terms of time, steps, and iteration cost.
    - COGNITIVE_LOAD: Measures how mentally taxing it is for the user to process and apply the response.
    - CONTEXT_AWARENESS: Measures how well the response tracks conversational state and constraints.
    - PERSONA_CONSISTENCY: Measures whether the response adheres to a specified role/voice/persona.
    - FRICTION_LOSS_OF_CONTROL: Measures whether the model resists the user's intent or makes the user "wrestle" the tool.
    - RELIABILITY_USER_TRUST: Measures perceived dependability and consistency, including hallucination patterns.
    - ANTHROPOMORPHISM: Measures perceived human-likeness versus roboticness in text interaction.
    """

    CLARITY = "clarity"
    """CLARITY measures how clear, structured, and readable the response is.
    Compare: organization (headings, steps), logical flow, formatting, signposting, and whether examples (if any) make the main point easier to follow.
    Also compare: precision of language (avoids ambiguity), and whether it is easy for THIS user to locate the key answer and apply it.
    Prefer: the response that this specific user would understand faster and with fewer rereads."""
    TONE_STYLE_FIT = "tone_style_fit"
    """TONE/STYLE FIT (Expressive Style) measures whether the writing style matches the user's preferences and the task context.
    Compare: tone (formal vs casual), verbosity (terse vs detailed), directness, humor/warmth, and whether the response's voice feels appropriate for the user's stated needs.
    Important: this is about stylistic choices (witty, professional, supportive, concise), not whether the response feels human-like (Anthropomorphism).
    Prefer: the response whose expressive style best matches this persona's expectations for this task."""
    WORKFLOW_FIT = "workflow_fit"
    """WORKFLOW FIT measures how well the response fits the user's workflow in terms of time, steps, and iteration cost.
    Compare: how quickly it enables progress (actionability, concrete next steps), how many back-and-forths it would require, and whether it minimizes unnecessary work for the user.
    Also compare: whether it anticipates common follow-up needs (e.g., provides needed details, commands, or checks) without adding irrelevant padding.
    Disambiguation: "slow but obedient" belongs here as poor Workflow Fit, even if it follows instructions.
    Prefer: the response that lets THIS user accomplish the task with fewer steps and less iteration."""
    COGNITIVE_LOAD = "cognitive_load"
    """COGNITIVE LOAD measures how mentally taxing it is for THIS user to process and apply the response.
    Compare: chunking (short steps vs dense paragraphs), ordering (prerequisites before advanced details), number of new concepts introduced, and how much implicit knowledge is assumed.
    Also compare: whether the response provides scaffolding (definitions, reminders, brief rationale) only where needed, and avoids overwhelming branching options.
    Prefer: the response that this user can execute correctly with the least mental effort and confusion."""
    CONTEXT_AWARENESS = "context_awareness"
    """CONTEXT AWARENESS measures how well the response tracks the conversational state and constraints from the prompt and prior turns.
    Compare: whether it uses the provided details (constraints, goals, environment, prior decisions), avoids contradicting earlier information, and maintains state across turns.
    Also compare: whether it respects explicit instructions (formatting requirements, scope limits, terminology constraints) and correctly carries forward earlier choices.
    Important: tailoring should rely on the prompt and conversation context, not on assumptions about the user beyond what is provided.
    Prefer: the response that better incorporates and remains consistent with the given context and constraints."""
    PERSONA_CONSISTENCY = "persona_consistency"
    """PERSONA CONSISTENCY measures whether the response adheres to a specified role/voice/persona across turns when a persona is part of the evaluation setup.
    Compare: consistency of voice and role (e.g., "academic NLP researcher", "supportive tutor", "terse debugger"), stable level of detail and terminology, and whether it stays within the persona's promised stance.
    Also compare: whether later parts of the response drift (e.g., starts technical then becomes generic, starts formal then becomes chatty) without a reason grounded in the prompt.
    Important: if no persona is specified as part of the setup, do not penalize; treat this dimension as not applicable or neutral.
    Prefer: the response that maintains the specified persona most consistently from start to finish."""
    FRICTION_LOSS_OF_CONTROL = "friction_loss_of_control"
    """FRICTION/LOSS OF CONTROL (Frustration) measures whether the model resists the user's intent or makes the user "wrestle" the tool.
    Compare: constraint following (does it ignore instructions, change format, derail into tangents), cooperativeness (does it force rework), and whether it creates a "gaslit" feeling (claims it did something it did not, contradicts itself, or shifts requirements).
    Also compare: whether it unnecessarily blocks progress by being evasive, overcomplicating, or repeatedly asking for avoidable clarifications.
    Disambiguation: "fast but ignores constraints" is high Friction. Refusal-driven behavior belongs primarily to Safety/Refusal Behavior, not here.
    Prefer: the response that best follows user intent and constraints without derailing or forcing extra work."""
    RELIABILITY_USER_TRUST = "reliability_user_trust"
    """RELIABILITY (USER TRUST) measures perceived dependability and consistency, including hallucination patterns and error modes.
    Compare: internal consistency (no contradictions), factual caution (signals uncertainty when needed), and whether claims are verifiable or grounded in the provided context.
    Also compare: robustness of the guidance (handles edge cases, avoids brittle steps), and whether it would require constant user verification to trust it.
    Important: this is subjective trustworthiness, not benchmark accuracy; judge from the response's consistency, calibration, and error-avoidance behavior.
    Prefer: the response that this user could rely on with less checking and fewer surprise failure modes."""
    ANTHROPOMORPHISM = "anthropomorphism"
    """ANTHROPOMORPHISM measures perceived human-likeness versus roboticness in text interaction.
    Compare: naturalness of phrasing, conversational flow, non-templated wording, and social presence (e.g., reads like a person rather than a form letter), while staying appropriate for the setting.
    Also compare: whether it avoids overly mechanical patterns (boilerplate disclaimers, repetitive structures) without becoming overly casual or performative.
    Important: do not treat this as evidence of agency or emotion; it is purely a perception of "human-like" vs "mechanical" writing.
    Keep distinct from Expressive Style: a response can be formal and still feel human-like, or casual and still feel templated.
    Prefer: the response that feels more naturally human in interaction for this user and context."""

    # Backward compatibility aliases (deprecated, use new names)
    EFFICIENCY = "workflow_fit"  # Deprecated: use WORKFLOW_FIT
    FRUSTRATION = "friction_loss_of_control"  # Deprecated: use FRICTION_LOSS_OF_CONTROL


class VibeDimensionEvaluator:
    """
    Evaluates individual vibe dimensions using judge prompts plus rubric heuristics.
    """

    def __init__(self, judge_evaluator: Any, rubric_scorer: Optional["RubricScorer"]):
        """
        Args:
            judge_evaluator (Any): Object exposing judge_vibe_dimension.
            rubric_scorer (Optional[RubricScorer]): Heuristic scorer to blend in.
        """
        self.judge_evaluator = judge_evaluator
        self.rubric_scorer = rubric_scorer

    def evaluate_dimension(
        self,
        dimension: VibeDimension,
        user_profile: UserProfile,
        model_generation: ModelGeneration,
        static_result: StaticEvaluationResult,
    ) -> Dict[str, Any]:
        """
        Evaluate a specific vibe dimension and return a blended score.

        Args:
            dimension (VibeDimension): Dimension being evaluated.
            user_profile (UserProfile): Persona context for the user.
            model_generation (ModelGeneration): Model input/output bundle.
            static_result (StaticEvaluationResult): Static correctness reference.

        Returns:
            Dict[str, Any]: Score and supporting metadata.
        """
        logger.debug(
            "Scoring %s for task %s", dimension.value, model_generation.task_id
        )
        judge_score, details = self.judge_evaluator.judge_vibe_dimension(
            dimension, user_profile, model_generation
        )
        details["static_correctness_reference"] = static_result.correctness_score

        blended_score = judge_score
        if self.rubric_scorer:
            rubric_score, rubric_details = self.rubric_scorer.score_dimension(
                dimension, model_generation, user_profile
            )
            if rubric_score is not None:
                blend_ratio = self.rubric_scorer.get_blend_ratio(dimension)
                blended_score = self._blend_scores(
                    judge_score, rubric_score, blend_ratio
                )
                details["rubric"] = rubric_details
                details["components"] = {
                    "judge": judge_score,
                    "rubric": rubric_score,
                    "blend_ratio": blend_ratio,
                }
                logger.debug(
                    "Blended judge %.3f with rubric %.3f using ratio %.2f for %s",
                    judge_score,
                    rubric_score,
                    blend_ratio,
                    dimension.value,
                )

        return {"score": blended_score, "details": details}

    def evaluate_dimension_batch(
        self,
        dimension: VibeDimension,
        user_profiles: list[UserProfile],
        model_generations: list[ModelGeneration],
        static_results: list[StaticEvaluationResult],
    ) -> list[Dict[str, Any]]:
        """
        Batch-evaluate a dimension across multiple generations.
        """
        if not model_generations:
            return []

        judge_outputs = self.judge_evaluator.judge_vibe_dimension_batch(
            dimension, user_profiles, model_generations
        )
        results: list[Dict[str, Any]] = []
        for idx, (judge_score, details) in enumerate(judge_outputs):
            details["static_correctness_reference"] = static_results[idx].correctness_score
            blended_score = judge_score
            if self.rubric_scorer:
                rubric_score, rubric_details = self.rubric_scorer.score_dimension(
                    dimension, model_generations[idx], user_profiles[idx]
                )
                if rubric_score is not None:
                    blend_ratio = self.rubric_scorer.get_blend_ratio(dimension)
                    blended_score = self._blend_scores(
                        judge_score, rubric_score, blend_ratio
                    )
                    details["rubric"] = rubric_details
                    details["components"] = {
                        "judge": judge_score,
                        "rubric": rubric_score,
                        "blend_ratio": blend_ratio,
                    }
            results.append({"score": blended_score, "details": details})
        return results

    @staticmethod
    def _blend_scores(
        judge_score: float, rubric_score: float, blend_ratio: float
    ) -> float:
        """Blend judge and rubric scores while ensuring the result stays in [0, 1]."""
        blend_ratio = max(0.0, min(1.0, blend_ratio))
        blended = judge_score * blend_ratio + rubric_score * (1.0 - blend_ratio)
        return max(0.0, min(1.0, blended))
