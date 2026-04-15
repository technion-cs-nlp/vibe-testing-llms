"""Core data structures for the vibe-testing pipeline."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# --- Data structures for Stage 3: Vibe Dataset Construction ---


class ChangeOption(BaseModel):
    """Represents a single, specific modification option for a prompt."""

    name: str = Field(
        description="A short, unique, snake_case identifier for the change."
    )
    change: str = Field(description="A brief description of the modification.")
    example: str = Field(description="A concrete example illustrating the change.")
    rationale: str = Field(
        description="An explanation of why this change aligns with the user profile field."
    )


class FieldChanges(BaseModel):
    """A list of change options related to a single field in the user profile."""

    field: str = Field(description="The UserProfile field this set of changes targets.")
    options: List[ChangeOption]


class ChangeIdentificationOutput(BaseModel):
    """The structured output from the change identification step (Subtask 1)."""

    changes_by_field: List[FieldChanges]


class VerificationOutput(BaseModel):
    """The structured output from the verification step (Subtask 3)."""

    same_end_goal: bool
    same_ground_truth: bool
    reason_if_failed: Optional[str] = None


class PromptVariation(BaseModel):
    """Represents a single personalized and verified prompt variation."""

    variation_id: str
    applied_changes: List[str] = Field(
        description="A list of 'name' identifiers for the ChangeOptions applied."
    )
    modified_prompt: str
    verification: VerificationOutput


class UserProfilePersona(BaseModel):
    """
    A structured representation of a user's preferences and requirements
    for LLM evaluation (used during profiling).
    """

    persona_description: str
    input_dimensions: Dict[str, Any]
    output_dimensions: Dict[str, Any]


class UserProfile(BaseModel):
    """
    A unified structured representation of a user's preferences and requirements
    across all pipeline stages.
    """

    user_id: str
    description: str = ""
    persona_description: str = ""
    tasks: List[str] = Field(default_factory=list)
    input_dimensions: Dict[str, Any] = Field(default_factory=dict)
    output_dimensions: Dict[str, Any] = Field(default_factory=dict)

    # Subjective evaluation fields (populated automatically if missing)
    persona: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    preferred_output_dimensions: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def unify_profile(cls, data: Any) -> Any:
        """Robustly unifies profile data from different pipeline stages."""
        if not isinstance(data, dict):
            return data

        # Map Stage 1 (Profiling) outputs to unified fields
        if "persona_description" in data and not data.get("persona"):
            data["persona"] = {"description": data["persona_description"]}

        if "input_dimensions" in data and not data.get("context"):
            data["context"] = data["input_dimensions"]

        if "output_dimensions" in data and not data.get("preferred_output_dimensions"):
            vibe_prefs = []
            output_dims = data.get("output_dimensions", {})
            # Keys that are intentionally not mapped to evaluation dimensions
            # (user preference only, no corresponding VibeDimension).
            _PREFERENCE_ONLY_KEYS = {"safety", "refusal", "creativity", "originality"}
            for k in output_dims:
                k_lower = k.lower()
                matched = False
                if "clarity" in k_lower:
                    vibe_prefs.append("clarity")
                    matched = True
                if (
                    "workflow" in k_lower
                    or "efficiency" in k_lower
                    or "time" in k_lower
                    or "iteration" in k_lower
                ):
                    vibe_prefs.append("workflow_fit")
                    matched = True
                if "style" in k_lower or "tone" in k_lower or "expressive" in k_lower:
                    vibe_prefs.append("tone_style_fit")
                    matched = True
                if "cognitive" in k_lower or "load" in k_lower or "mental" in k_lower:
                    vibe_prefs.append("cognitive_load")
                    matched = True
                if "context" in k_lower or "awareness" in k_lower:
                    vibe_prefs.append("context_awareness")
                    matched = True
                if "persona" in k_lower or "consistency" in k_lower:
                    vibe_prefs.append("persona_consistency")
                    matched = True
                if (
                    "friction" in k_lower
                    or "frustration" in k_lower
                    or "control" in k_lower
                ):
                    vibe_prefs.append("friction_loss_of_control")
                    matched = True
                if (
                    "reliability" in k_lower
                    or "trust" in k_lower
                    or "dependability" in k_lower
                ):
                    vibe_prefs.append("reliability_user_trust")
                    matched = True
                if (
                    "anthropomorphism" in k_lower
                    or "human" in k_lower
                    or "robotic" in k_lower
                ):
                    vibe_prefs.append("anthropomorphism")
                    matched = True
                if not matched:
                    is_preference_only = any(
                        tok in k_lower for tok in _PREFERENCE_ONLY_KEYS
                    )
                    if is_preference_only:
                        logger.debug(
                            "output_dimensions key '%s' is a preference-only field "
                            "with no corresponding evaluation dimension; skipped "
                            "in preferred_output_dimensions.",
                            k,
                        )
                    else:
                        logger.warning(
                            "output_dimensions key '%s' did not match any known "
                            "vibe dimension pattern and was not added to "
                            "preferred_output_dimensions. If this is a new dimension, "
                            "update the mapping in UserProfile.unify_profile.",
                            k,
                        )
            # Remove duplicates while preserving order
            seen = set()
            unique_prefs = []
            for pref in vibe_prefs:
                if pref not in seen:
                    seen.add(pref)
                    unique_prefs.append(pref)
            data["preferred_output_dimensions"] = unique_prefs

        # Map back from Stage 5 style if persona_description or dimensions are missing
        if not data.get("persona_description") and "persona" in data:
            data["persona_description"] = data["persona"].get("description", "")

        if not data.get("input_dimensions") and "context" in data:
            data["input_dimensions"] = data["context"]

        return data


class BenchmarkSample(BaseModel):
    """Represents a single, raw task from a benchmark dataset."""

    sample_id: str
    source_benchmark: str
    prompt: str
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = {}


class VibeTask(BaseModel):
    """
    Represents a personalized evaluation task, combining a benchmark sample
    with user-specific adaptations.
    """

    task_id: str
    base_sample: BenchmarkSample
    user_profile: UserProfile
    final_prompt: str
    evaluation_metrics: List[str]


class PersonalizedSample(BaseModel):
    """
    Contains the original benchmark sample plus all personalized variations
    derived from it. This is the core data object for the final dataset.
    """

    original_sample: BenchmarkSample
    variations: List[PromptVariation]


class VibeDataset(BaseModel):
    """Represents the entire generated dataset for a given user profile."""

    user_profile: UserProfile
    change_options: ChangeIdentificationOutput
    personalized_samples: List[PersonalizedSample]


class EvaluationResult(BaseModel):
    """
    Stores the outcome of a single model evaluation on a VibeTask.
    """

    task_id: str
    model_name: str
    raw_output: str
    quantitative_scores: Dict[str, float] = {}
    qualitative_scores: Dict[str, Any] = {}
    error: Optional[str] = None


class FunctionEvaluationTests(BaseModel):
    """
    Defines the entry point and tests required for function-level evaluation.
    """

    entry_point: str
    base_tests: List[str]
    plus_tests: List[str] = Field(default_factory=list)


class FunctionEvaluationSample(BaseModel):
    """
    Represents a single MBPP+-style task for function-level evaluation.
    """

    sample_id: str
    prompt: str
    tests: FunctionEvaluationTests
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PatchEvaluationSample(BaseModel):
    """
    Represents a SWE-Bench-style patch evaluation instance.
    """

    instance_id: str
    prompt: str
    repo_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
