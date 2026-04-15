"""Public exports for the Stage 6 analysis package."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List

from src.vibe_testing.data_utils import EvaluationResult

from .aggregations import (  # noqa: F401
    AggregationBundle,
    AnalysisDataError,
    run_full_aggregation,
)
from .exporters import (  # noqa: F401
    write_joint_preference_long_table,
    write_joint_preference_matrix,
    write_joint_preference_overall_latex,
    write_joint_preference_streamlit_agreement_latex,
    write_joint_preference_streamlit_dimension_agreement_latex,
    write_model_overall_summary,
    write_pairwise_dimension_summary,
    write_pairwise_pair_summary,
    write_pairwise_preference_matrix,
    write_pairwise_sample_level,
    write_pairwise_statistical_tests,
    write_pairwise_user_summary,
    write_sample_level_flat,
    write_user_model_deltas,
    write_user_model_variant_summary,
)
from .figures import (  # noqa: F401
    plot_joint_preference_matrix_heatmap,
    plot_joint_preference_overall_grid,
    plot_joint_preference_persona_panels,
    plot_model_ranking,
    plot_objective_vs_subjective_scatter,
    plot_pairwise_by_user,
    plot_pairwise_dimension_comparison,
    plot_pairwise_dimension_heatmap,
    plot_pairwise_forest,
    plot_pairwise_win_rates,
    plot_persona_metric_bars,
    plot_personalization_deltas,
    plot_position_bias_rates,
    plot_preference_matrix_heatmap,
    plot_vibe_dimension_bars,
)
from .io import AnalysisInputLoader  # noqa: F401
from .judge_utils import (  # noqa: F401
    filter_human_judges_from_df,
    is_human_judge_token,
    split_judges_by_group,
)
from .dimension_omits import (  # noqa: F401
    apply_pairwise_dimension_omits,
    apply_subjective_dimension_omits,
    normalize_omit_dimensions,
)
from .pairwise import (  # noqa: F401
    PairwiseAggregationBundle,
    PairwiseAnalysisError,
    build_preference_matrix,
    compute_dimension_win_rates,
    compute_model_rankings,
    compute_pair_summary,
    compute_statistical_significance,
    compute_user_pair_summary,
    run_pairwise_aggregation,
)
from .joint_preference import (  # noqa: F401
    JointPreferenceError,
    JointPreferenceMatrix,
    compute_joint_preference_matrices,
)

__all__ = [
    "AggregationBundle",
    "AnalysisDataError",
    "AnalysisInputLoader",
    "apply_pairwise_dimension_omits",
    "apply_subjective_dimension_omits",
    "filter_human_judges_from_df",
    "is_human_judge_token",
    "normalize_omit_dimensions",
    "JointPreferenceError",
    "JointPreferenceMatrix",
    "PairwiseAggregationBundle",
    "PairwiseAnalysisError",
    "ResultAnalyzer",
    "build_preference_matrix",
    "compute_dimension_win_rates",
    "compute_joint_preference_matrices",
    "compute_model_rankings",
    "compute_pair_summary",
    "compute_statistical_significance",
    "compute_user_pair_summary",
    "plot_joint_preference_matrix_heatmap",
    "plot_joint_preference_overall_grid",
    "plot_joint_preference_persona_panels",
    "plot_model_ranking",
    "plot_objective_vs_subjective_scatter",
    "plot_pairwise_by_user",
    "plot_pairwise_dimension_comparison",
    "plot_pairwise_dimension_heatmap",
    "plot_pairwise_forest",
    "plot_pairwise_win_rates",
    "plot_persona_metric_bars",
    "plot_personalization_deltas",
    "plot_position_bias_rates",
    "plot_preference_matrix_heatmap",
    "plot_vibe_dimension_bars",
    "run_full_aggregation",
    "run_pairwise_aggregation",
    "split_judges_by_group",
    "write_joint_preference_long_table",
    "write_joint_preference_matrix",
    "write_joint_preference_overall_latex",
    "write_joint_preference_streamlit_agreement_latex",
    "write_joint_preference_streamlit_dimension_agreement_latex",
    "write_model_overall_summary",
    "write_pairwise_dimension_summary",
    "write_pairwise_pair_summary",
    "write_pairwise_preference_matrix",
    "write_pairwise_sample_level",
    "write_pairwise_statistical_tests",
    "write_pairwise_user_summary",
    "write_sample_level_flat",
    "write_user_model_deltas",
    "write_user_model_variant_summary",
]


class ResultAnalyzer:
    """
    Backwards-compatible analyzer for legacy scripts that still operate on
    ``EvaluationResult`` objects instead of tabular data.
    """

    def analyze(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
        Compute simple aggregate statistics from evaluation results.

        Args:
            results (List[EvaluationResult]): Detailed evaluation records.

        Returns:
            Dict[str, float]: Summary metrics (sample count + averages).
        """
        if not results:
            return {"total_samples": 0, "avg_pass_fail": 0.0, "avg_clarity": 0.0}

        quantitative = [r.quantitative_scores.get("pass_fail") for r in results]
        qualitative = [
            r.qualitative_scores.get("clarity_and_comprehensibility")
            for r in results
        ]

        return {
            "total_samples": len(results),
            "avg_pass_fail": _safe_mean(quantitative),
            "avg_clarity": _safe_mean(qualitative),
        }


def _safe_mean(values: List[float]) -> float:
    """Compute the mean while ignoring ``None`` entries."""
    filtered = [v for v in values if v is not None]
    return mean(filtered) if filtered else 0.0
