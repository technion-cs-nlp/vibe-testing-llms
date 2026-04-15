from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

# Import the unified UserProfile from data_utils
from src.vibe_testing.data_utils import UserProfile


@dataclass
class ModelGeneration:
    user_id: str
    task_id: str
    input_text: str
    generated_output: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticEvaluationResult:
    user_id: str
    task_id: str
    correctness_score: float
    accuracy_metrics: Dict[str, float]
    error_types: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubjectiveEvaluationResult:
    user_id: str
    task_id: str
    model_name: str

    # Vibe dimension scores (0-1 scale)
    clarity_score: float
    tone_style_fit_score: float
    workflow_fit_score: (
        float  # New: measures workflow fit (time, steps, iteration cost)
    )
    cognitive_load_score: float  # Higher is better after normalization (low load).
    context_awareness_score: float
    persona_consistency_score: float
    friction_loss_of_control_score: float  # New: measures friction/loss of control
    reliability_user_trust_score: float  # New: measures reliability and user trust
    anthropomorphism_score: float  # New: measures human-likeness vs roboticness

    # Backward compatibility fields (deprecated, populated from new fields)
    efficiency_score: float  # Deprecated: use workflow_fit_score
    frustration_indicator: float  # Deprecated: use friction_loss_of_control_score

    # Aggregated scores
    overall_subjective_score: float
    combined_score: float  # Combined with static scores

    # Detailed breakdowns
    dimension_breakdowns: Dict[str, Dict[str, Any]]
    judge_metadata: Dict[str, Any]

    # Input-side and output-side indicators
    input_side_indicators: Dict[str, float]
    output_side_indicators: Dict[str, float]
    # Grouped view: dimension -> metrics dict
    vibe_text_metrics: Dict[str, Any] = field(default_factory=dict)
    # Flat view (metric_name -> value) for backward compatibility / easy joins
    vibe_text_metrics_flat: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
        """Helper to serialize to dict."""
        return self.__dict__
