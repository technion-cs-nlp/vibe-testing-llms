"""
Pairwise comparison judges for vibe dimensions.

This module implements LLM-based judges that compare two model outputs and
determine which better serves either a specific user persona or a general user.
"""

import logging
from typing import Any, Dict, Mapping, Tuple

from .pairwise_schemas import (
    ComparisonConfidence,
    ComparisonWinner,
    DimensionComparisonResult,
    PairwiseComparisonInput,
)
from .schemas import UserProfile
from .vibe_dimensions import VibeDimension
from src.vibe_testing.models.base import BaseModel

logger = logging.getLogger(__name__)

_DIMENSION_GUIDANCE: Mapping[VibeDimension, str] = {
    VibeDimension.CLARITY: """CLARITY measures how clear, structured, and readable the response is.
Compare: organization (headings, steps), logical flow, formatting, signposting, and whether examples (if any) make the main point easier to follow.
Also compare: precision of language (avoids ambiguity), and whether it is easy for THIS user to locate the key answer and apply it.
Prefer: the response that this specific user would understand faster and with fewer rereads.""",
    VibeDimension.TONE_STYLE_FIT: """TONE/STYLE FIT (Expressive Style) measures whether the writing style matches the user's preferences and the task context.
Compare: tone (formal vs casual), verbosity (terse vs detailed), directness, humor/warmth, and whether the response's voice feels appropriate for the user's stated needs.
Important: this is about stylistic choices (witty, professional, supportive, concise), not whether the response feels human-like (Anthropomorphism).
Prefer: the response whose expressive style best matches this persona's expectations for this task.""",
    VibeDimension.WORKFLOW_FIT: """WORKFLOW FIT measures how well the response fits the user's workflow in terms of time, steps, and iteration cost.
Compare: how quickly it enables progress (actionability, concrete next steps), how many back-and-forths it would require, and whether it minimizes unnecessary work for the user.
Also compare: whether it anticipates common follow-up needs (e.g., provides needed details, commands, or checks) without adding irrelevant padding.
Disambiguation: "slow but obedient" belongs here as poor Workflow Fit, even if it follows instructions.
Prefer: the response that lets THIS user accomplish the task with fewer steps and less iteration.""",
    VibeDimension.COGNITIVE_LOAD: """COGNITIVE LOAD measures how mentally taxing it is for THIS user to process and apply the response.
Compare: chunking (short steps vs dense paragraphs), ordering (prerequisites before advanced details), number of new concepts introduced, and how much implicit knowledge is assumed.
Also compare: whether the response provides scaffolding (definitions, reminders, brief rationale) only where needed, and avoids overwhelming branching options.
Prefer: the response that this user can execute correctly with the least mental effort and confusion.""",
    VibeDimension.CONTEXT_AWARENESS: """CONTEXT AWARENESS measures how well the response tracks the conversational state and constraints from the prompt and prior turns.
Compare: whether it uses the provided details (constraints, goals, environment, prior decisions), avoids contradicting earlier information, and maintains state across turns.
Also compare: whether it respects explicit instructions (formatting requirements, scope limits, terminology constraints) and correctly carries forward earlier choices.
Important: tailoring should rely on the prompt and conversation context, not on assumptions about the user beyond what is provided.
Prefer: the response that better incorporates and remains consistent with the given context and constraints.""",
    VibeDimension.PERSONA_CONSISTENCY: """PERSONA CONSISTENCY measures whether the response adheres to a specified role/voice/persona across turns when a persona is part of the evaluation setup.
Compare: consistency of voice and role (e.g., "academic NLP researcher", "supportive tutor", "terse debugger"), stable level of detail and terminology, and whether it stays within the persona's promised stance.
Also compare: whether later parts of the response drift (e.g., starts technical then becomes generic, starts formal then becomes chatty) without a reason grounded in the prompt.
Important: if no persona is specified as part of the setup, do not penalize; treat this dimension as not applicable or neutral.
Prefer: the response that maintains the specified persona most consistently from start to finish.""",
    VibeDimension.FRICTION_LOSS_OF_CONTROL: """FRICTION/LOSS OF CONTROL (Frustration) measures whether the model resists the user's intent or makes the user "wrestle" the tool.
Compare: constraint following (does it ignore instructions, change format, derail into tangents), cooperativeness (does it force rework), and whether it creates a "gaslit" feeling (claims it did something it did not, contradicts itself, or shifts requirements).
Also compare: whether it unnecessarily blocks progress by being evasive, overcomplicating, or repeatedly asking for avoidable clarifications.
Disambiguation: "fast but ignores constraints" is high Friction. Refusal-driven behavior belongs primarily to Safety/Refusal Behavior, not here.
Prefer: the response that best follows user intent and constraints without derailing or forcing extra work.""",
    VibeDimension.RELIABILITY_USER_TRUST: """RELIABILITY (USER TRUST) measures perceived dependability and consistency, including hallucination patterns and error modes.
Compare: internal consistency (no contradictions), factual caution (signals uncertainty when needed), and whether claims are verifiable or grounded in the provided context.
Also compare: robustness of the guidance (handles edge cases, avoids brittle steps), and whether it would require constant user verification to trust it.
Important: this is subjective trustworthiness, not benchmark accuracy; judge from the response's consistency, calibration, and error-avoidance behavior.
Prefer: the response that this user could rely on with less checking and fewer surprise failure modes.""",
    VibeDimension.ANTHROPOMORPHISM: """ANTHROPOMORPHISM measures perceived human-likeness versus roboticness in text interaction.
Compare: naturalness of phrasing, conversational flow, non-templated wording, and social presence (e.g., reads like a person rather than a form letter), while staying appropriate for the setting.
Also compare: whether it avoids overly mechanical patterns (boilerplate disclaimers, repetitive structures) without becoming overly casual or performative.
Important: do not treat this as evidence of agency or emotion; it is purely a perception of "human-like" vs "mechanical" writing.
Keep distinct from Expressive Style: a response can be formal and still feel human-like, or casual and still feel templated.
Prefer: the response that feels more naturally human in interaction for this user and context.""",
}


def get_dimension_guidance_text(dimension: VibeDimension | str) -> str:
    """
    Return the exact dimension guidance text used by pairwise judges.

    Args:
        dimension (VibeDimension | str): Dimension enum or dimension string.

    Returns:
        str: Dimension guidance text shown to judges.
    """
    if isinstance(dimension, str):
        value = dimension.strip().lower()
        if value == VibeDimension.EFFICIENCY.value:
            dimension = VibeDimension.WORKFLOW_FIT
        elif value == VibeDimension.FRUSTRATION.value:
            dimension = VibeDimension.FRICTION_LOSS_OF_CONTROL
        else:
            dimension = VibeDimension(value)

    if dimension == VibeDimension.EFFICIENCY:
        dimension = VibeDimension.WORKFLOW_FIT
    elif dimension == VibeDimension.FRUSTRATION:
        dimension = VibeDimension.FRICTION_LOSS_OF_CONTROL

    return _DIMENSION_GUIDANCE.get(
        dimension, "Evaluate which response is better for this dimension."
    )


def get_all_dimension_guidance_texts() -> Dict[str, str]:
    """
    Return all supported judge guidance texts keyed by canonical dimension string.

    Args:
        None.

    Returns:
        Dict[str, str]: Guidance text keyed by dimension name.
    """
    return {dimension.value: text for dimension, text in _DIMENSION_GUIDANCE.items()}


class BasePairwiseJudge:
    """
    Base class for LLM-based pairwise judges.

    Subclasses customize the system framing while preserving a shared output
    contract and response parser.
    """

    BASE_PROMPT_TEMPLATE = ""

    def __init__(
        self,
        model: BaseModel,
        generation_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Initialize the pairwise judge.

        Args:
            model (BaseModel): The LLM to use as judge.
            generation_kwargs (Dict[str, Any], optional): Parameters for generation.
        """
        self.model = model
        self._generation_kwargs = generation_kwargs or {}

    def compare_dimension(
        self,
        dimension: VibeDimension,
        user_profile: UserProfile,
        comparison_input: PairwiseComparisonInput,
    ) -> DimensionComparisonResult:
        """
        Compare two model outputs on a specific vibe dimension.

        Args:
            dimension (VibeDimension): The dimension to evaluate.
            user_profile (UserProfile): User persona context.
            comparison_input (PairwiseComparisonInput): The two outputs to compare.

        Returns:
            DimensionComparisonResult: Comparison result with winner and rationale.
        """
        prompt = self._construct_prompt(dimension, user_profile, comparison_input)
        logger.debug(
            "Comparing %s vs %s on %s",
            comparison_input.model_a_name,
            comparison_input.model_b_name,
            dimension.value,
        )

        response = self.model.generate(prompt, **self._generation_kwargs)
        winner, confidence, rationale = self._parse_response(response, dimension)

        return DimensionComparisonResult(
            dimension=dimension.value,
            winner=winner,
            confidence=confidence,
            rationale=rationale,
            raw_response=response,
            model_a_name=comparison_input.model_a_name,
            model_b_name=comparison_input.model_b_name,
        )

    def _construct_prompt(
        self,
        dimension: VibeDimension,
        user_profile: UserProfile,
        comparison_input: PairwiseComparisonInput,
    ) -> str:
        """
        Construct the comparison prompt for the judge model.

        Args:
            dimension: Target dimension.
            user_profile: Persona context.
            comparison_input: The two outputs to compare.

        Returns:
            str: Formatted prompt.
        """
        persona_info = self._format_user_context(user_profile)
        dimension_guidance = self._get_dimension_guidance(dimension)

        return self.BASE_PROMPT_TEMPLATE.format(
            persona_info=persona_info,
            input_text=comparison_input.input_text,
            model_a_name=comparison_input.model_a_name,
            response_a=comparison_input.model_a_output,
            model_b_name=comparison_input.model_b_name,
            response_b=comparison_input.model_b_output,
            dimension_name=dimension.value.upper().replace("_", " "),
            dimension_guidance=dimension_guidance,
        )

    def _format_user_context(self, user_profile: UserProfile) -> str:
        """
        Format user-facing evaluation context for the prompt.

        Args:
            user_profile (UserProfile): User profile payload.

        Returns:
            str: Prompt-ready user context block.
        """
        raise NotImplementedError

    def _get_dimension_guidance(self, dimension: VibeDimension) -> str:
        """Get specific guidance for each dimension comparison."""
        return get_dimension_guidance_text(dimension)

    def _parse_response(
        self,
        response: str,
        dimension: VibeDimension,
    ) -> Tuple[ComparisonWinner, ComparisonConfidence, str]:
        """
        Parse the judge's response to extract winner, confidence, and rationale.

        Args:
            response: Raw judge response.
            dimension: Dimension being evaluated (for logging).

        Returns:
            Tuple of (winner, confidence, rationale).
        """
        try:
            lines = [
                line.strip() for line in response.strip().splitlines() if line.strip()
            ]

            # Extract winner
            winner = ComparisonWinner.TIE
            winner_line = next(
                (l for l in lines if l.lower().startswith("winner:")), ""
            )
            if winner_line:
                winner_value = winner_line.split(":", 1)[1].strip().upper()
                if winner_value == "A":
                    winner = ComparisonWinner.MODEL_A
                elif winner_value == "B":
                    winner = ComparisonWinner.MODEL_B
                elif winner_value in ("TIE", "NEITHER", "BOTH", "EQUAL"):
                    winner = ComparisonWinner.TIE

            # Extract confidence
            confidence = ComparisonConfidence.MEDIUM
            conf_line = next(
                (l for l in lines if l.lower().startswith("confidence:")), ""
            )
            if conf_line:
                conf_value = conf_line.split(":", 1)[1].strip().lower()
                if "high" in conf_value:
                    confidence = ComparisonConfidence.HIGH
                elif "low" in conf_value:
                    confidence = ComparisonConfidence.LOW
                else:
                    confidence = ComparisonConfidence.MEDIUM

            # Extract rationale
            rationale = ""
            rat_line = next(
                (l for l in lines if l.lower().startswith("rationale:")), ""
            )
            if rat_line:
                rationale = rat_line.split(":", 1)[1].strip()
            else:
                # Fallback: use any line that isn't winner or confidence
                other_lines = [
                    l
                    for l in lines
                    if not l.lower().startswith(("winner:", "confidence:"))
                ]
                if other_lines:
                    rationale = " ".join(other_lines)

            return winner, confidence, rationale

        except Exception as exc:
            logger.exception(
                "Failed to parse pairwise judge response for %s: %s",
                dimension.value,
                exc,
            )
            return ComparisonWinner.TIE, ComparisonConfidence.LOW, f"Parse error: {exc}"


class PairwiseJudge(BasePairwiseJudge):
    """
    LLM-based judge for pairwise comparison of model outputs.

    Prompts a judge model to compare two responses and determine which
    better serves the user persona for each vibe dimension.
    """

    # Base prompt template for pairwise comparison
    BASE_PROMPT_TEMPLATE = """You are an impartial evaluator comparing two AI responses for a specific user persona. You will be given a user profile, a question and two responses. You will need to compare the two responses with respect to the given aspect and determine which response better serves the user persona.

{persona_info}

Question:
{input_text}

--- Response A (Model: A) ---
{response_a}

--- Response B (Model: B) ---
{response_b}

Aspect: {dimension_name}
{dimension_guidance}

Compare the two responses above with respect to the given aspect.
Which response better serves this user persona for the given aspect?

IMPORTANT:
- Focus ONLY on the given aspect, not overall quality.
- Consider the user's background, preferences, and context.
- "tie" is acceptable only if both responses are genuinely equivalent for the given aspect.

Return your verdict STRICTLY in this format (exactly 3 lines):
Winner: [A/B/tie]
Confidence: [low/medium/high]
Rationale: [One sentence explaining your choice for this persona and the given aspect]"""

    def _format_user_context(self, user_profile: UserProfile) -> str:
        """Format user profile into a readable string."""
        lines = [f"User Persona: {user_profile.persona}"]
        if user_profile.context:
            lines.append(f"User Context: {user_profile.context}")
        if user_profile.preferred_output_dimensions:
            lines.append(
                f"User Preferences: {', '.join(user_profile.preferred_output_dimensions)}"
            )
        return "\n".join(lines)


class GeneralUserPairwiseJudge(BasePairwiseJudge):
    """
    LLM-based judge for persona-agnostic pairwise comparisons.

    This variant explicitly asks the judge to evaluate responses for a general
    coding user without relying on persona-specific preferences.
    """

    BASE_PROMPT_TEMPLATE = """You are an impartial evaluator comparing two AI responses for a general coding user. You will be given general evaluation instructions, a question, and two responses. You must compare the responses with respect to the given aspect and determine which response better serves a general user.

{persona_info}

Question:
{input_text}

--- Response A (Model: A) ---
{response_a}

--- Response B (Model: B) ---
{response_b}

Aspect: {dimension_name}
{dimension_guidance}

Compare the two responses above with respect to the given aspect.
Which response better serves a general user for the given aspect?

IMPORTANT:
- Focus ONLY on the given aspect, not overall quality.
- Do NOT rely on a specific persona, user background, or personalized preference.
- Judge what would work better for a general user who wants a useful, clear, and cooperative response.
- "tie" is acceptable only if both responses are genuinely equivalent for the given aspect.

Return your verdict STRICTLY in this format (exactly 3 lines):
Winner: [A/B/tie]
Confidence: [low/medium/high]
Rationale: [One sentence explaining your choice for a general user and the given aspect]"""

    def _format_user_context(self, user_profile: UserProfile) -> str:
        """
        Format general-user instructions into a readable string.

        Args:
            user_profile (UserProfile): User profile payload from the calling stage.

        Returns:
            str: Prompt-ready general-user instruction block.
        """
        lines = [
            "General User Setting: Evaluate these responses without assuming a specific persona."
        ]
        if user_profile.context:
            lines.append(
                "Task Context: Use the explicit task and conversation constraints only."
            )
        lines.append(
            "General User Preference: Favor the response that would help a broad coding user succeed with less confusion and less unnecessary effort."
        )
        return "\n".join(lines)
