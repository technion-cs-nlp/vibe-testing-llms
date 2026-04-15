import logging
import re
from typing import Any, Dict, List, Tuple

from .schemas import ModelGeneration, UserProfile
from .vibe_dimensions import VibeDimension
from src.vibe_testing.models.base import BaseModel, GenerationRequest

logger = logging.getLogger(__name__)


class ModelJudge:
    """
    Base class for model-based judges that prompt an LLM to grade vibe dimensions.
    """

    def __init__(self, model: BaseModel, generation_kwargs: Dict[str, Any] | None = None):
        self.model = model
        self._generation_kwargs = generation_kwargs or {}

    def judge_vibe_dimension(
        self,
        dimension: VibeDimension,
        user_profile: UserProfile,
        model_generation: ModelGeneration,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Judge a specific vibe dimension.

        Args:
            dimension (VibeDimension): Target dimension.
            user_profile (UserProfile): Persona metadata.
            model_generation (ModelGeneration): Input/output pair.

        Returns:
            Tuple[float, Dict[str, Any]]: Normalized score plus rationale details.
        """
        prompt = self._construct_prompt(dimension, user_profile, model_generation)
        logger.debug("Invoking judge model for %s", dimension.value)
        response = self.model.generate(prompt, **self._generation_kwargs)
        score, rationale = self._parse_response(response, dimension)
        return score, {"rationale": rationale, "raw_response": response}

    def judge_vibe_dimension_batch(
        self,
        dimension: VibeDimension,
        user_profiles: List[UserProfile],
        model_generations: List[ModelGeneration],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Batch version of judge_vibe_dimension leveraging model.generate_batch.
        """
        if not model_generations:
            return []

        prompts: List[str] = []
        for profile, generation in zip(user_profiles, model_generations):
            prompts.append(self._construct_prompt(dimension, profile, generation))

        requests = [
            GenerationRequest(prompt=p, generation_kwargs=self._generation_kwargs)
            for p in prompts
        ]
        responses = self.model.generate_batch(requests)

        results: List[Tuple[float, Dict[str, Any]]] = []
        for response in responses:
            score, rationale = self._parse_response(response, dimension)
            results.append((score, {"rationale": rationale, "raw_response": response}))
        return results

    def _construct_prompt(
        self,
        dimension: VibeDimension,
        user_profile: UserProfile,
        model_generation: ModelGeneration,
    ) -> str:
        """
        Construct the prompt for the judge model based on the dimension.
        """
        persona_info = f"User Persona: {user_profile.persona}\nUser Context: {user_profile.context}"
        io_block = f"""Task/Input:
{model_generation.input_text}

Model Output:
{model_generation.generated_output}
"""
        dimension_guidance = self._dimension_guidance(dimension)

        prompt = f"""
You are an impartial evaluator scoring AI responses for a specific user persona.
{persona_info}

{io_block}

Dimension under review: {dimension.value.upper()}
{dimension_guidance}

Always use the user's persona and context to calibrate expectations.
Return your verdict strictly in this format:
Score: [0-10]
Rationale: [One or two sentences that justify the score for this persona]
"""
        return prompt.strip()

    def _dimension_guidance(self, dimension: VibeDimension) -> str:
        """Map each dimension to explicit instructions."""
        guidance = {
            VibeDimension.CLARITY: """CLARITY measures how clear, structured, and readable the response is.
Compare: organization (headings, steps), logical flow, formatting, signposting, and whether examples (if any) make the main point easier to follow.
Also compare: precision of language (avoids ambiguity), and whether it is easy for THIS user to locate the key answer and apply it.
Prefer: the response that this specific user would understand faster and with fewer rereads.
10 = perfectly structured and easy to follow for the persona. 0 = extremely confusing.""",
            VibeDimension.TONE_STYLE_FIT: """TONE/STYLE FIT (Expressive Style) measures whether the writing style matches the user's preferences and the task context.
Compare: tone (formal vs casual), verbosity (terse vs detailed), directness, humor/warmth, and whether the response's voice feels appropriate for the user's stated needs.
Important: this is about stylistic choices (witty, professional, supportive, concise), not whether the response feels human-like (Anthropomorphism).
Prefer: the response whose expressive style best matches this persona's expectations for this task.
10 = perfectly aligned with persona preferences. 0 = completely mismatched style.""",
            VibeDimension.WORKFLOW_FIT: """WORKFLOW FIT measures how well the response fits the user's workflow in terms of time, steps, and iteration cost.
Compare: how quickly it enables progress (actionability, concrete next steps), how many back-and-forths it would require, and whether it minimizes unnecessary work for the user.
Also compare: whether it anticipates common follow-up needs (e.g., provides needed details, commands, or checks) without adding irrelevant padding.
Disambiguation: "slow but obedient" belongs here as poor Workflow Fit, even if it follows instructions.
Prefer: the response that lets THIS user accomplish the task with fewer steps and less iteration.
10 = maximally efficient workflow fit for this task. 0 = requires excessive iteration or rework.""",
            VibeDimension.COGNITIVE_LOAD: """COGNITIVE LOAD measures how mentally taxing it is for THIS user to process and apply the response.
Compare: chunking (short steps vs dense paragraphs), ordering (prerequisites before advanced details), number of new concepts introduced, and how much implicit knowledge is assumed.
Also compare: whether the response provides scaffolding (definitions, reminders, brief rationale) only where needed, and avoids overwhelming branching options.
Prefer: the response that this user can execute correctly with the least mental effort and confusion.
10 = very easy to digest (low cognitive load). 0 = extremely taxing to understand.""",
            VibeDimension.CONTEXT_AWARENESS: """CONTEXT AWARENESS measures how well the response tracks the conversational state and constraints from the prompt and prior turns.
Compare: whether it uses the provided details (constraints, goals, environment, prior decisions), avoids contradicting earlier information, and maintains state across turns.
Also compare: whether it respects explicit instructions (formatting requirements, scope limits, terminology constraints) and correctly carries forward earlier choices.
Important: tailoring should rely on the prompt and conversation context, not on assumptions about the user beyond what is provided.
Prefer: the response that better incorporates and remains consistent with the given context and constraints.
10 = deeply contextualized and consistent. 0 = ignores context or contradicts earlier information.""",
            VibeDimension.PERSONA_CONSISTENCY: """PERSONA CONSISTENCY measures whether the response adheres to a specified role/voice/persona across turns when a persona is part of the evaluation setup.
Compare: consistency of voice and role (e.g., "academic NLP researcher", "supportive tutor", "terse debugger"), stable level of detail and terminology, and whether it stays within the persona's promised stance.
Also compare: whether later parts of the response drift (e.g., starts technical then becomes generic, starts formal then becomes chatty) without a reason grounded in the prompt.
Important: if no persona is specified as part of the setup, do not penalize; treat this dimension as not applicable or neutral.
Prefer: the response that maintains the specified persona most consistently from start to finish.
10 = perfectly consistent persona adherence. 0 = significant persona drift.""",
            VibeDimension.FRICTION_LOSS_OF_CONTROL: """FRICTION/LOSS OF CONTROL (Frustration) measures whether the model resists the user's intent or makes the user "wrestle" the tool.
Compare: constraint following (does it ignore instructions, change format, derail into tangents), cooperativeness (does it force rework), and whether it creates a "gaslit" feeling (claims it did something it did not, contradicts itself, or shifts requirements).
Also compare: whether it unnecessarily blocks progress by being evasive, overcomplicating, or repeatedly asking for avoidable clarifications.
Disambiguation: "fast but ignores constraints" is high Friction. Refusal-driven behavior belongs primarily to Safety/Refusal Behavior, not here.
Prefer: the response that best follows user intent and constraints without derailing or forcing extra work.
10 = no friction (clear, honest, helpful, follows constraints). 0 = highly frustrating (misleading, contradictory, ignores instructions).""",
            VibeDimension.RELIABILITY_USER_TRUST: """RELIABILITY (USER TRUST) measures perceived dependability and consistency, including hallucination patterns and error modes.
Compare: internal consistency (no contradictions), factual caution (signals uncertainty when needed), and whether claims are verifiable or grounded in the provided context.
Also compare: robustness of the guidance (handles edge cases, avoids brittle steps), and whether it would require constant user verification to trust it.
Important: this is subjective trustworthiness, not benchmark accuracy; judge from the response's consistency, calibration, and error-avoidance behavior.
Prefer: the response that this user could rely on with less checking and fewer surprise failure modes.
10 = highly reliable and trustworthy. 0 = unreliable, contradictory, or requires constant verification.""",
            VibeDimension.ANTHROPOMORPHISM: """ANTHROPOMORPHISM measures perceived human-likeness versus roboticness in text interaction.
Compare: naturalness of phrasing, conversational flow, non-templated wording, and social presence (e.g., reads like a person rather than a form letter), while staying appropriate for the setting.
Also compare: whether it avoids overly mechanical patterns (boilerplate disclaimers, repetitive structures) without becoming overly casual or performative.
Important: do not treat this as evidence of agency or emotion; it is purely a perception of "human-like" vs "mechanical" writing.
Keep distinct from Expressive Style: a response can be formal and still feel human-like, or casual and still feel templated.
Prefer: the response that feels more naturally human in interaction for this user and context.
10 = feels naturally human-like. 0 = feels mechanical or templated.""",
        }
        # Handle backward compatibility aliases
        if dimension not in guidance:
            if dimension == VibeDimension.EFFICIENCY:
                return guidance[VibeDimension.WORKFLOW_FIT]
            elif dimension == VibeDimension.FRUSTRATION:
                return guidance[VibeDimension.FRICTION_LOSS_OF_CONTROL]
        return guidance[dimension]

    def _parse_response(self, response: str, dimension: VibeDimension) -> Tuple[float, str]:
        """
        Parse the judge model's response to extract score and rationale.
        """
        try:
            lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
            score_line = next((l for l in lines if l.lower().startswith("score:")), "")
            rationale_line = next((l for l in lines if l.lower().startswith("rationale:")), "")

            score = self._extract_score_value(score_line, response)
            rationale = rationale_line.split(":", 1)[1].strip() if ":" in rationale_line else response
            return score, rationale
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to parse judge response for %s: %s", dimension.value, exc)
            return 0.0, f"Error parsing response: {exc}"

    @staticmethod
    def _extract_score_value(score_line: str, response: str) -> float:
        """Normalize the score line to [0, 1]."""
        try:
            if score_line:
                raw_val = score_line.split(":", 1)[1].strip()
            else:
                match = re.search(r"score:\s*([\d\.]+)", response, re.IGNORECASE)
                raw_val = match.group(1) if match else "5"
            if "/" in raw_val:
                raw_val = raw_val.split("/", 1)[0]
            numeric = float(raw_val)
            if numeric > 1:
                numeric /= 10.0
            return max(0.0, min(1.0, numeric))
        except ValueError:
            return 0.5
