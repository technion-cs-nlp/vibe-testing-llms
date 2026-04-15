import logging
import re
from typing import Any, Dict, Optional

from .schemas import ModelGeneration, StaticEvaluationResult, UserProfile
from .vibe_dimensions import VibeDimension

logger = logging.getLogger(__name__)


def _clamp(value: float) -> float:
    """Clamp helper."""
    return max(0.0, min(1.0, value))


class ScoreAggregator:
    """
    Combines subjective scores, derives indicators, and integrates with static metrics.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config (Dict[str, Any], optional): Aggregation configuration.
        """
        self.config = config or {}
        self.default_dimension_weights = self.config.get("dimension_weights", {})
        self.preferred_multiplier = self.config.get("preferred_multiplier", 2.0)
        self.combined_weights = self.config.get(
            "combined_weights", {"subjective": 0.5, "static": 0.5}
        )

    def aggregate_subjective_scores(
        self,
        dimension_scores: Dict[str, float],
        user_profile: Optional[UserProfile] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Aggregate individual dimension scores into an overall subjective score.

        Args:
            dimension_scores (Dict[str, float]): Scores per dimension in [0, 1].
            user_profile (Optional[UserProfile]): Used to upweight preferred dimensions.
            weights (Optional[Dict[str, float]]): Explicit overrides for dimension weights.

        Returns:
            float: Weighted average subjective score in [0, 1].
        """
        if not dimension_scores:
            logger.warning("No dimension scores provided, returning 0.0.")
            return 0.0

        final_weights = {
            dim: self.default_dimension_weights.get(dim, 1.0)
            for dim in dimension_scores
        }

        if user_profile and user_profile.preferred_output_dimensions:
            preferred = {
                pref.lower() for pref in user_profile.preferred_output_dimensions
            }
            for dim in final_weights:
                if dim.lower() in preferred:
                    final_weights[dim] *= self.preferred_multiplier

        if weights:
            final_weights.update(weights)

        total_weight = sum(final_weights.values())
        if total_weight <= 0:
            logger.warning("Total dimension weight is zero; returning 0.")
            return 0.0

        weighted_sum = 0.0
        for dim, score in dimension_scores.items():
            clamped = _clamp(score)
            weighted_sum += clamped * final_weights.get(dim, 1.0)

        overall = weighted_sum / total_weight
        return _clamp(overall)

    def combine_with_static_scores(
        self,
        subjective_score: float,
        static_score: float,
        combination_method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Combine subjective vibe scores with static correctness scores.

        Args:
            subjective_score (float): Subjective score in [0, 1].
            static_score (float): Static correctness score in [0, 1].
            combination_method (str, optional): Combination strategy (currently weighted average).
            weights (Optional[Dict[str, float]]): Overrides for subjective/static weights.

        Returns:
            float: Combined score in [0, 1].
        """
        if combination_method != "weighted_average":
            logger.warning(
                "Unsupported combination method %s; defaulting to weighted_average.",
                combination_method,
            )

        weight_config = weights or self.combined_weights
        w_sub = weight_config.get("subjective", 0.5)
        w_stat = weight_config.get("static", 0.5)

        denom = w_sub + w_stat
        if denom <= 0:
            logger.warning("Subjective/static weights sum to zero. Returning 0.")
            return 0.0

        combined = (
            _clamp(subjective_score) * w_sub + _clamp(static_score) * w_stat
        ) / denom
        return _clamp(combined)

    def compute_input_side_indicators(
        self,
        user_profile: UserProfile,
        model_generation: ModelGeneration,
    ) -> Dict[str, float]:
        """
        Compute interpretable input-side indicators (prompt and user context).

        Args:
            user_profile (UserProfile): Persona and task context.
            model_generation (ModelGeneration): Generation under review.

        Returns:
            Dict[str, float]: Indicator name to value.
        """
        indicators: Dict[str, float] = {}
        prompt = model_generation.input_text or ""
        tasks = user_profile.tasks or []

        indicators["input_length_chars"] = float(len(prompt))
        indicators["input_length_tokens"] = float(len(prompt.split()))
        indicators["is_long_context"] = 1.0 if len(prompt) > 1000 else 0.0
        indicators["input_sentence_count"] = float(self._sentence_count(prompt))
        indicators["input_paragraph_count"] = float(self._paragraph_count(prompt))
        indicators["input_bullet_count"] = float(self._bullet_count(prompt))
        indicators["input_heading_count"] = float(self._heading_count(prompt))
        indicators["input_code_fence_count"] = float(prompt.count("```"))
        indicators["input_url_count"] = float(self._url_count(prompt))
        indicators["input_inline_code_span_count"] = float(
            self._inline_code_span_count(prompt)
        )
        indicators["input_unique_token_ratio"] = float(self._unique_token_ratio(prompt))
        indicators["task_count"] = float(len(tasks))
        avg_task_len = sum(len(t) for t in tasks) / len(tasks) if tasks else 0.0
        indicators["avg_task_complexity_proxy"] = _clamp(avg_task_len / 800.0)
        # Use dynamic dimension count for coverage calculation
        # Filter out aliases (EFFICIENCY, FRUSTRATION) to get actual unique dimension count
        unique_values = set(d.value for d in VibeDimension)
        total_dimensions = len(unique_values)
        indicators["preferred_dimension_coverage"] = _clamp(
            len(user_profile.preferred_output_dimensions or []) / float(total_dimensions)
        )

        context_constraints = len(user_profile.context or {})
        indicators["context_constraints_count"] = float(context_constraints)
        indicators["is_high_complexity_task"] = (
            1.0 if indicators["input_length_tokens"] > 250 else 0.0
        )

        return indicators

    def compute_output_side_indicators(
        self,
        user_profile: UserProfile,
        model_generation: ModelGeneration,
        static_result: StaticEvaluationResult,
    ) -> Dict[str, float]:
        """
        Compute interpretable output-side indicators (answer characteristics).

        Args:
            user_profile (UserProfile): Persona (unused but keeps parity).
            model_generation (ModelGeneration): Model output bundle.
            static_result (StaticEvaluationResult): Static correctness reference.

        Returns:
            Dict[str, float]: Indicator name to value.
        """
        indicators: Dict[str, float] = {}
        output = model_generation.generated_output or ""
        tokens = output.split()
        step_pattern = re.compile(r"^\s*\d+[\).\]]")
        steps = len([line for line in output.splitlines() if step_pattern.match(line)])

        indicators["output_length_chars"] = float(len(output))
        indicators["output_length_tokens"] = float(len(tokens))
        indicators["code_block_count"] = float(output.count("```"))
        indicators["code_block_pair_count"] = float(output.count("```") // 2)
        indicators["has_code_block"] = (
            1.0 if indicators["code_block_count"] > 0 else 0.0
        )
        indicators["num_steps_in_explanation"] = float(steps)
        indicators["output_sentence_count"] = float(self._sentence_count(output))
        indicators["output_paragraph_count"] = float(self._paragraph_count(output))
        indicators["output_bullet_count"] = float(self._bullet_count(output))
        indicators["output_heading_count"] = float(self._heading_count(output))
        indicators["output_url_count"] = float(self._url_count(output))
        indicators["output_markdown_link_count"] = float(self._markdown_link_count(output))
        indicators["output_inline_code_span_count"] = float(
            self._inline_code_span_count(output)
        )
        indicators["output_hedging_term_count"] = float(self._hedging_term_count(output))
        indicators["output_overconfidence_term_count"] = float(
            self._overconfidence_term_count(output)
        )
        indicators["output_refusal_term_count"] = float(self._refusal_term_count(output))
        indicators["output_disclaimer_term_count"] = float(
            self._disclaimer_term_count(output)
        )
        indicators["output_first_person_pronoun_count"] = float(
            self._first_person_pronoun_count(output)
        )
        indicators["output_second_person_pronoun_count"] = float(
            self._second_person_pronoun_count(output)
        )
        indicators["output_unique_token_ratio"] = float(self._unique_token_ratio(output))

        prompt = model_generation.input_text or ""
        indicators["prompt_output_token_overlap_jaccard"] = float(
            self._token_overlap_jaccard(prompt, output)
        )
        context_hit_count, context_coverage = self._context_term_stats(
            user_profile, output
        )
        indicators["context_term_hit_count"] = float(context_hit_count)
        indicators["context_term_coverage"] = float(context_coverage)
        indicators["static_correctness"] = _clamp(static_result.correctness_score)
        indicators["static_correctness_bucket"] = self._bucketize_correctness(
            static_result.correctness_score
        )

        avg_sentence_length = len(tokens) / max(
            len(re.findall(r"[.!?]", output)) or 1, 1
        )
        indicators["avg_sentence_length_output"] = float(avg_sentence_length)

        return indicators

    @staticmethod
    def _bucketize_correctness(score: float) -> float:
        """Convert correctness into coarse buckets for plotting."""
        if score >= 0.9:
            return 1.0
        if score >= 0.5:
            return 0.5
        return 0.0

    @staticmethod
    def _sentence_count(text: str) -> int:
        """Count sentences using light punctuation heuristics."""
        if not text:
            return 0
        return int(max(len(re.findall(r"[.!?]", text)), 1))

    @staticmethod
    def _paragraph_count(text: str) -> int:
        """Count non-empty paragraphs separated by blank lines."""
        if not text:
            return 0
        return int(len([p for p in text.split("\n\n") if p.strip()]))

    @staticmethod
    def _bullet_count(text: str) -> int:
        """Count markdown-ish bullet lines."""
        if not text:
            return 0
        return int(
            len(
                [
                    line
                    for line in text.splitlines()
                    if line.strip().startswith(("-", "*", "•"))
                ]
            )
        )

    @staticmethod
    def _heading_count(text: str) -> int:
        """Count markdown headings (#, ##, ###...)."""
        if not text:
            return 0
        return int(
            len([line for line in text.splitlines() if line.lstrip().startswith("#")])
        )

    @staticmethod
    def _url_count(text: str) -> int:
        """Count URLs (http/https)."""
        if not text:
            return 0
        return int(len(re.findall(r"https?://\S+", text)))

    @staticmethod
    def _markdown_link_count(text: str) -> int:
        """Count markdown links: [label](url)."""
        if not text:
            return 0
        return int(len(re.findall(r"\[[^\]]+\]\([^)]+\)", text)))

    @staticmethod
    def _inline_code_span_count(text: str) -> int:
        """Count inline code spans delimited by single backticks."""
        if not text:
            return 0
        return int(len(re.findall(r"`[^`\n]+`", text)))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase word tokens."""
        if not text:
            return []
        return re.findall(r"[a-z0-9]+", text.lower())

    @classmethod
    def _unique_token_ratio(cls, text: str) -> float:
        """Compute unique-token ratio (type/token) using normalized word tokens."""
        tokens = cls._tokenize(text)
        if not tokens:
            return 0.0
        return float(len(set(tokens)) / float(len(tokens)))

    @classmethod
    def _token_overlap_jaccard(cls, a: str, b: str) -> float:
        """Compute Jaccard overlap between two token sets."""
        a_tokens = set(cls._tokenize(a))
        b_tokens = set(cls._tokenize(b))
        if not a_tokens and not b_tokens:
            return 0.0
        denom = len(a_tokens | b_tokens)
        return float(len(a_tokens & b_tokens) / float(denom)) if denom else 0.0

    @staticmethod
    def _count_terms(text: str, terms: set[str]) -> int:
        """Count occurrences of any term tokens."""
        if not text or not terms:
            return 0
        tokens = re.findall(r"[a-z']+", text.lower())
        return int(sum(1 for t in tokens if t in terms))

    @classmethod
    def _hedging_term_count(cls, text: str) -> int:
        """Count hedging/calibration markers."""
        terms = {"maybe", "might", "perhaps", "likely", "probably", "unsure", "unclear"}
        return cls._count_terms(text, terms)

    @classmethod
    def _overconfidence_term_count(cls, text: str) -> int:
        """Count overconfident markers."""
        terms = {"definitely", "certainly", "guaranteed", "always", "never", "obviously"}
        return cls._count_terms(text, terms)

    @classmethod
    def _refusal_term_count(cls, text: str) -> int:
        """Count refusal-style markers (very rough)."""
        terms = {"can't", "cannot", "won't", "unable", "refuse"}
        return cls._count_terms(text, terms)

    @classmethod
    def _disclaimer_term_count(cls, text: str) -> int:
        """Count common boilerplate disclaimer markers."""
        if not text:
            return 0
        lowered = text.lower()
        patterns = [
            "as an ai language model",
            "i cannot",
            "i can't",
            "i am unable",
        ]
        return int(sum(1 for p in patterns if p in lowered))

    @classmethod
    def _first_person_pronoun_count(cls, text: str) -> int:
        """Count first-person pronouns."""
        terms = {"i", "i'm", "ive", "i've", "me", "my", "we", "our", "us"}
        return cls._count_terms(text, terms)

    @classmethod
    def _second_person_pronoun_count(cls, text: str) -> int:
        """Count second-person pronouns."""
        terms = {"you", "your", "yours"}
        return cls._count_terms(text, terms)

    @staticmethod
    def _context_term_stats(user_profile: UserProfile, text: str) -> tuple[int, float]:
        """
        Count context-term hits and a simple coverage ratio.

        This treats each string value in user_profile.context as a "term" and checks
        substring matches. It is intentionally simple and deterministic.
        """
        context = user_profile.context or {}
        values = [v for v in context.values() if isinstance(v, str) and v.strip()]
        if not values:
            return 0, 0.0
        lowered = (text or "").lower()
        hits = sum(1 for v in values if v.lower() in lowered)
        return int(hits), float(hits / float(len(values)))
