import logging
import re
from typing import Any, Dict, Tuple, Union

from .schemas import ModelGeneration, UserProfile
from .vibe_dimensions import VibeDimension

logger = logging.getLogger(__name__)


def _clamp(value: float) -> float:
    """Clamp helper that keeps scores in the [0, 1] range."""
    return max(0.0, min(1.0, value))


class RubricScorer:
    """
    Rule-based and rubric-based scoring for vibe dimensions.
    Uses lightweight heuristics to provide deterministic anchors
    that can be blended with judge-model outputs.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the rubric scorer.

        Args:
            config (Dict[str, Any], optional): Optional configuration dictionary
                that controls heuristic thresholds and blending weights.
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.default_blend_alpha = self.config.get("blend_alpha", 0.7)
        self.dimension_blend = self.config.get("dimension_blend", {})
        self.max_word_count = self.config.get("max_word_count", 450)
        self.context_weight = self.config.get("context_weight", 0.6)

    def score_dimension(
        self,
        dimension: VibeDimension,
        model_generation: ModelGeneration,
        user_profile: UserProfile,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score a vibe dimension using heuristics.

        Args:
            dimension (VibeDimension): Target dimension.
            model_generation (ModelGeneration): Generation under review.
            user_profile (UserProfile): Persona context.

        Returns:
            Tuple[float, Dict[str, Any]]: Score in [0, 1] plus rubric details.
        """
        if not self.enabled:
            return None, {}

        text = model_generation.generated_output or ""
        features = self._extract_features(text)
        persona_type = str(user_profile.persona.get("type", "")).lower()
        context_terms = self._extract_context_terms(user_profile)

        if dimension == VibeDimension.CLARITY:
            score = self._score_clarity(features)
            details = {
                "avg_sentence_length": features["avg_sentence_length"],
                "bullet_count": features["bullet_count"],
            }
        elif dimension == VibeDimension.TONE_STYLE_FIT:
            score = self._score_tone_style(features, persona_type)
            details = {
                "persona_type": persona_type,
                "friendly_markers": features["friendly_markers"],
                "directive_markers": features["directive_markers"],
            }
        elif dimension == VibeDimension.WORKFLOW_FIT:
            # Reuse efficiency heuristic but focus on workflow aspects
            score = self._score_workflow_fit(features)
            details = {
                "word_count": features["word_count"],
                "code_blocks": features["code_block_count"],
                "step_count": features["step_count"],
            }
        elif dimension == VibeDimension.COGNITIVE_LOAD:
            score = self._score_cognitive_load(features)
            details = {
                "steps": features["step_count"],
                "avg_sentence_length": features["avg_sentence_length"],
            }
        elif dimension == VibeDimension.CONTEXT_AWARENESS:
            hit_count = self._count_keyword_hits(text, context_terms)
            score = self._score_context_awareness(text, context_terms)
            details = {"context_hits": hit_count}
        elif dimension == VibeDimension.PERSONA_CONSISTENCY:
            score = self._score_persona_consistency(features, persona_type)
            details = {
                "persona_type": persona_type,
                "step_count": features["step_count"],
            }
        elif dimension == VibeDimension.FRICTION_LOSS_OF_CONTROL:
            score = self._score_friction_loss_of_control(features)
            details = {
                "apology_terms": features["apology_terms"],
                "hedging_terms": features["hedging_terms"],
            }
        elif dimension == VibeDimension.RELIABILITY_USER_TRUST:
            score = self._score_reliability_user_trust(features)
            details = {
                "overconfidence_terms": features["overconfidence_terms"],
                "hedging_terms": features["hedging_terms"],
                "refusal_terms": features["refusal_terms"],
                "disclaimer_terms": features["disclaimer_terms"],
                "code_block_count": features["code_block_count"],
                "step_count": features["step_count"],
                "url_count": features["url_count"],
            }
        elif dimension == VibeDimension.ANTHROPOMORPHISM:
            score = self._score_anthropomorphism(features)
            details = {
                "first_person_pronouns": features["first_person_pronouns"],
                "contraction_count": features["contraction_count"],
                "disclaimer_terms": features["disclaimer_terms"],
                "type_token_ratio": features["type_token_ratio"],
                "friendly_markers": features["friendly_markers"],
                "directive_markers": features["directive_markers"],
            }
        # Backward compatibility aliases
        elif dimension == VibeDimension.EFFICIENCY:
            return self.score_dimension(VibeDimension.WORKFLOW_FIT, model_generation, user_profile)
        elif dimension == VibeDimension.FRUSTRATION:
            return self.score_dimension(VibeDimension.FRICTION_LOSS_OF_CONTROL, model_generation, user_profile)
        else:
            return None, {}

        score = _clamp(score)
        return score, details

    def get_blend_ratio(self, dimension: VibeDimension) -> float:
        """
        Retrieve the blend ratio for a dimension.

        Args:
            dimension (VibeDimension): Target dimension.

        Returns:
            float: Blend ratio between judge and rubric contributions.
        """
        return self.dimension_blend.get(dimension.value, self.default_blend_alpha)

    def score_clarity(
        self,
        generation: Union[str, ModelGeneration],
        user_profile: UserProfile,
    ) -> float:
        """
        Score clarity based on readability and structure.

        Args:
            generation (Union[str, ModelGeneration]): Generation text or object.
            user_profile (UserProfile): Persona context (unused but kept for API parity).

        Returns:
            float: Clarity score in [0, 1].
        """
        text = (
            generation if isinstance(generation, str) else generation.generated_output
        )
        return self._score_clarity(self._extract_features(text or ""))

    def score_tone_style_fit(
        self,
        generation: Union[str, ModelGeneration],
        user_profile: UserProfile,
    ) -> float:
        """
        Score tone/style alignment with persona.

        Args:
            generation (Union[str, ModelGeneration]): Generation text or object.
            user_profile (UserProfile): Persona context.

        Returns:
            float: Tone/style fit score in [0, 1].
        """
        text = (
            generation if isinstance(generation, str) else generation.generated_output
        )
        persona_type = str(user_profile.persona.get("type", "")).lower()
        return self._score_tone_style(self._extract_features(text or ""), persona_type)

    def _extract_features(self, text: str) -> Dict[str, float]:
        """Extract reusable textual features."""
        tokens = text.split()
        word_count = len(tokens)
        sentence_count = max(len(re.findall(r"[.!?]", text)) or 1, 1)
        paragraph_count = len([p for p in text.split("\n\n") if p.strip()]) or 1
        bullet_count = len(
            [
                line
                for line in text.splitlines()
                if line.strip().startswith(("-", "*", "•"))
            ]
        )
        step_pattern = re.compile(r"^\s*\d+[\).\]]")
        step_count = len(
            [line for line in text.splitlines() if step_pattern.match(line)]
        )
        code_block_count = text.count("```")
        url_count = len(re.findall(r"https?://\S+", text))
        inline_code_count = len(re.findall(r"`[^`\n]+`", text))
        heading_count = len([line for line in text.splitlines() if line.lstrip().startswith("#")])
        contraction_count = len(re.findall(r"\b\w+'\w+\b", text))
        # Tokenize for type/token ratio (lexical diversity proxy)
        word_tokens = re.findall(r"[a-z0-9]+", text.lower())
        type_token_ratio = (len(set(word_tokens)) / float(len(word_tokens))) if word_tokens else 0.0
        friendly_markers = sum(
            1
            for token in tokens
            if token.lower() in {"let's", "awesome", "great", "friendly"}
        )
        directive_markers = sum(
            1 for token in tokens if token.lower() in {"must", "ensure", "required"}
        )
        apology_terms = sum(
            1 for token in tokens if token.lower() in {"sorry", "apologies"}
        )
        hedging_terms = sum(
            1 for token in tokens if token.lower() in {"maybe", "might", "perhaps"}
        )
        overconfidence_terms = sum(
            1
            for token in tokens
            if token.lower() in {"definitely", "certainly", "guaranteed", "always", "never", "obviously"}
        )
        refusal_terms = sum(
            1 for token in tokens if token.lower() in {"can't", "cannot", "won't", "unable", "refuse"}
        )
        disclaimer_terms = 1.0 if "as an ai language model" in text.lower() else 0.0
        first_person_pronouns = sum(
            1 for token in tokens if token.lower() in {"i", "i'm", "ive", "i've", "me", "my", "we", "our", "us"}
        )
        avg_sentence_length = (
            word_count / sentence_count if sentence_count else word_count
        )

        return {
            "word_count": float(word_count),
            "sentence_count": float(sentence_count),
            "avg_sentence_length": float(avg_sentence_length),
            "paragraph_count": float(paragraph_count),
            "bullet_count": float(bullet_count),
            "step_count": float(step_count),
            "code_block_count": float(code_block_count),
            "url_count": float(url_count),
            "inline_code_count": float(inline_code_count),
            "heading_count": float(heading_count),
            "contraction_count": float(contraction_count),
            "type_token_ratio": float(type_token_ratio),
            "friendly_markers": float(friendly_markers),
            "directive_markers": float(directive_markers),
            "apology_terms": float(apology_terms),
            "hedging_terms": float(hedging_terms),
            "overconfidence_terms": float(overconfidence_terms),
            "refusal_terms": float(refusal_terms),
            "disclaimer_terms": float(disclaimer_terms),
            "first_person_pronouns": float(first_person_pronouns),
        }

    def _extract_context_terms(self, user_profile: UserProfile) -> Dict[str, str]:
        """Collect lowercase context values for keyword matching."""
        terms = {}
        for key, value in (user_profile.context or {}).items():
            if isinstance(value, str):
                terms[key.lower()] = value.lower()
        return terms

    def _score_clarity(self, features: Dict[str, float]) -> float:
        """Clarity heuristic."""
        length_penalty = min(features["avg_sentence_length"] / 30.0, 1.2)
        structure_bonus = (
            min(features["bullet_count"], 5) * 0.04
            + min(features["paragraph_count"], 6) * 0.02
        )
        return _clamp(1.0 - length_penalty + structure_bonus)

    def _score_tone_style(self, features: Dict[str, float], persona_type: str) -> float:
        """Tone/style heuristic."""
        score = 0.5
        if "novice" in persona_type:
            score += (
                min(features["step_count"], 6) * 0.05
                + min(features["friendly_markers"], 4) * 0.04
            )
            score -= min(features["directive_markers"], 3) * 0.05
        elif "expert" in persona_type or "researcher" in persona_type:
            conciseness = 1.0 - min(features["word_count"] / 600.0, 1.0)
            score += conciseness * 0.4 + min(features["code_block_count"], 4) * 0.05
            score -= min(features["friendly_markers"], 3) * 0.03
        else:
            score += min(features["bullet_count"], 5) * 0.03
        return _clamp(score)

    def _score_efficiency(self, features: Dict[str, float]) -> float:
        """Deprecated: use _score_workflow_fit instead."""
        return self._score_workflow_fit(features)

    def _score_cognitive_load(self, features: Dict[str, float]) -> float:
        """Cognitive load heuristic (higher = easier)."""
        clarity_component = self._score_clarity(features)
        steps_bonus = min(features["step_count"], 6) * 0.05
        return _clamp((clarity_component + steps_bonus) / 1.1)

    def _score_context_awareness(
        self, text: str, context_terms: Dict[str, str]
    ) -> float:
        """Context awareness heuristic."""
        if not context_terms:
            return 0.5
        hits = self._count_keyword_hits(text, context_terms)
        coverage = hits / max(len(context_terms), 1)
        return _clamp(coverage * self.context_weight + 0.2)

    def _count_keyword_hits(self, text: str, context_terms: Dict[str, str]) -> int:
        """Count context keyword hits."""
        lowered = text.lower()
        return sum(1 for term in context_terms.values() if term and term in lowered)

    def _score_persona_consistency(
        self, features: Dict[str, float], persona_type: str
    ) -> float:
        """Persona consistency heuristic."""
        tone_score = self._score_tone_style(features, persona_type)
        detail_bonus = (
            min(features["step_count"], 5) * 0.03
            if "novice" in persona_type
            else min(features["code_block_count"], 4) * 0.04
        )
        return _clamp(tone_score + detail_bonus)

    def _score_friction_loss_of_control(self, features: Dict[str, float]) -> float:
        """Friction/loss of control heuristic (higher = less friction)."""
        # Similar to frustration but emphasizes constraint following and cooperativeness
        penalty = (
            min(features["apology_terms"], 3) * 0.1
            + min(features["hedging_terms"], 5) * 0.05
        )
        structure_bonus = min(features["paragraph_count"], 5) * 0.02
        return _clamp(1.0 - penalty + structure_bonus)
    
    def _score_workflow_fit(self, features: Dict[str, float]) -> float:
        """Workflow fit heuristic (focuses on actionability and iteration cost)."""
        # Similar to efficiency but emphasizes workflow aspects
        verbosity_penalty = min(features["word_count"] / self.max_word_count, 1.2)
        code_bonus = min(features["code_block_count"], 4) * 0.05
        step_bonus = min(features["step_count"], 6) * 0.03  # Actionable steps are good
        return _clamp(1.0 - verbosity_penalty + code_bonus + step_bonus)
    
    def _score_frustration(self, features: Dict[str, float]) -> float:
        """Deprecated: use _score_friction_loss_of_control instead."""
        return self._score_friction_loss_of_control(features)

    def _score_reliability_user_trust(self, features: Dict[str, float]) -> float:
        """
        Reliability/user-trust heuristic.

        This is intentionally conservative and uses only deterministic surface signals:
        - Penalize strong overconfidence and boilerplate disclaimers/refusals.
        - Reward actionable/evidence-like structure (steps, code blocks, links).
        """
        base = 0.55
        evidence_bonus = (
            min(features.get("code_block_count", 0.0), 4.0) * 0.05
            + min(features.get("step_count", 0.0), 6.0) * 0.02
            + min(features.get("url_count", 0.0), 3.0) * 0.03
        )
        overconfidence_penalty = min(features.get("overconfidence_terms", 0.0), 4.0) * 0.06
        refusal_penalty = 0.15 if features.get("refusal_terms", 0.0) > 0 else 0.0
        disclaimer_penalty = min(features.get("disclaimer_terms", 0.0), 2.0) * 0.10
        excessive_hedge_penalty = 0.05 if features.get("hedging_terms", 0.0) > 8 else 0.0
        score = base + evidence_bonus - overconfidence_penalty - refusal_penalty - disclaimer_penalty - excessive_hedge_penalty
        return _clamp(score)

    def _score_anthropomorphism(self, features: Dict[str, float]) -> float:
        """
        Anthropomorphism heuristic (human-like vs robotic).

        Deterministic approximation:
        - Reward light conversational markers (contractions, first-person, friendly style).
        - Penalize boilerplate disclaimers and overly templated markers.
        """
        base = 0.45
        conversational_bonus = (
            min(features.get("contraction_count", 0.0), 6.0) * 0.03
            + min(features.get("first_person_pronouns", 0.0), 6.0) * 0.02
            + min(features.get("friendly_markers", 0.0), 4.0) * 0.05
        )
        diversity_bonus = min(features.get("type_token_ratio", 0.0), 0.65) / 0.65 * 0.10 if features.get("type_token_ratio", 0.0) > 0 else 0.0
        boilerplate_penalty = min(features.get("disclaimer_terms", 0.0), 2.0) * 0.15
        score = base + conversational_bonus + diversity_bonus - boilerplate_penalty
        return _clamp(score)
