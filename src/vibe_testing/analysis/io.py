"""Input loading and normalization utilities for Stage 6 analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.vibe_testing.model_names import canonicalize_model_name
from src.vibe_testing.pairwise_artifact_diagnostics import (
    PairwiseArtifactLoadContext,
    PairwiseArtifactLoadError,
    wrap_pairwise_artifact_load_error,
)
from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
)
from src.vibe_testing.pathing import infer_prompt_type, normalize_token
from src.vibe_testing.utils import load_json, load_jsonl

VARIATION_TOKEN = "::variation::"
VARIANT_FRIENDLY_MAP = {
    "variation": "personalized",
    "personalized": "personalized",
    "personalization": "personalized",
    "original": "original",
    "base": "original",
    "control": "control",
}
SUBJECTIVE_DIMENSIONS = {
    "clarity_score": "subj_clarity",
    "tone_style_fit_score": "subj_tone_style_fit",
    "workflow_fit_score": "subj_workflow_fit",
    "cognitive_load_score": "subj_cognitive_load",
    "context_awareness_score": "subj_context_awareness",
    "persona_consistency_score": "subj_persona_consistency",
    "friction_loss_of_control_score": "subj_friction_loss_of_control",
    "reliability_user_trust_score": "subj_reliability_user_trust",
    "anthropomorphism_score": "subj_anthropomorphism",
    # Backward compatibility mappings (deprecated)
    "efficiency_score": "subj_workflow_fit",  # Maps to workflow_fit for compatibility
    "frustration_indicator": "subj_friction_loss_of_control",  # Maps to friction_loss_of_control
}
# Map raw dimension keys (as stored in dimension_breakdowns) to analysis columns
DIMENSION_KEY_TO_COLUMN = {
    "clarity": "subj_clarity",
    "tone_style_fit": "subj_tone_style_fit",
    "workflow_fit": "subj_workflow_fit",
    "cognitive_load": "subj_cognitive_load",
    "context_awareness": "subj_context_awareness",
    "persona_consistency": "subj_persona_consistency",
    "friction_loss_of_control": "subj_friction_loss_of_control",
    "reliability_user_trust": "subj_reliability_user_trust",
    "anthropomorphism": "subj_anthropomorphism",
    # Backward compatibility mappings (deprecated)
    "efficiency": "subj_workflow_fit",  # Maps to workflow_fit for compatibility
    "frustration": "subj_friction_loss_of_control",  # Maps to friction_loss_of_control
}

# Pairwise dimension names (matching VibeDimension enum values)
PAIRWISE_DIMENSIONS = [
    "clarity",
    "tone_style_fit",
    "workflow_fit",
    "cognitive_load",
    "context_awareness",
    "persona_consistency",
    "friction_loss_of_control",
    "reliability_user_trust",
    "anthropomorphism",
]


def _build_pairwise_load_context(
    *,
    artifact_path: Path,
    failure_stage: str,
    entry: Optional[Dict[str, Any]] = None,
    record_index: Optional[int] = None,
    record_count: Optional[int] = None,
    payload_type: Optional[str] = None,
) -> PairwiseArtifactLoadContext:
    """Build structured context for a pairwise artifact load attempt."""

    safe_entry = entry if isinstance(entry, dict) else {}
    return PairwiseArtifactLoadContext(
        artifact_path=str(artifact_path),
        failure_stage=failure_stage,
        record_index=record_index,
        record_count=record_count,
        payload_type=payload_type,
        task_id=safe_entry.get("task_id"),
        raw_task_id=safe_entry.get("task_id"),
        model_a_name=safe_entry.get("model_a_name"),
        model_b_name=safe_entry.get("model_b_name"),
        judge_model_name=safe_entry.get("judge_model_name"),
    )


def count_failed_verification_units(function_eval_dir: Path) -> int:
    """
    Count objective evaluation units that were skipped due to failed Stage-3 verification.

    This inspects per-sample Stage-4 JSON artifacts written under the
    ``function_eval/`` directory and counts entries with ``metrics.verification_failed``.

    Args:
        function_eval_dir (Path): Directory containing Stage-4 function-level artifacts.

    Returns:
        int: Number of verification-gated units found.

    Raises:
        ValueError: If the provided path exists but is not a directory.
    """
    if function_eval_dir.exists() and not function_eval_dir.is_dir():
        raise ValueError(f"Expected a directory path, got file: {function_eval_dir}")
    if not function_eval_dir.exists():
        return 0

    failed = 0
    for path in function_eval_dir.glob("*.json"):
        payload = load_json(str(path))
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {}) or {}
        if bool(metrics.get("verification_failed", False)):
            failed += 1
    return failed


class AnalysisInputLoader:
    """
    Loads and normalizes all artifacts required for Stage 6 analysis.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initializes the loader.

        Args:
            logger (Optional[logging.Logger]): Logger instance used for status
                updates. Defaults to ``logging.getLogger(__name__)``.
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_user_profiles(self, profile_path: str) -> pd.DataFrame:
        """
        Load user profile metadata and flatten key persona attributes.

        Args:
            profile_path (str): Absolute path to the Stage 1 profile JSON/JSONL.

        Returns:
            pandas.DataFrame: One row per ``user_id`` with persona descriptors.
        """
        path = self._validate_path(profile_path)
        raw_profiles = self._load_json_payload(path)
        if isinstance(raw_profiles, dict):
            raw_profiles = raw_profiles.get("profiles", [raw_profiles])
        records: List[Dict[str, Any]] = []
        for profile in raw_profiles:
            if "user_id" not in profile:
                continue
            records.append(self._normalize_profile_record(profile))

        profile_df = pd.DataFrame(records)
        self._log_shape("user profiles", profile_df)
        return profile_df

    def load_subjective_results(
        self, subjective_path: str, default_model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load subjective evaluation outputs (Stage 5) and annotate variants.

        Args:
            subjective_path (str): Path to the subjective evaluation JSON/JSONL.
            default_model_name (Optional[str]): Model name used when records omit
                ``model_name``.

        Returns:
            pandas.DataFrame: Rows keyed by ``(user_id, task_id, model_name,
            variant_label)`` containing vibe dimensions and combined scores.
        """
        path = self._validate_path(subjective_path)
        raw_subjective = self._load_json_payload(path)
        if isinstance(raw_subjective, dict):
            raw_subjective = raw_subjective.get("results", [raw_subjective])

        records: List[Dict[str, Any]] = []
        for entry in raw_subjective:
            task_identifier = entry.get("task_id")
            if not task_identifier:
                continue
            (
                task_id,
                variant_label,
                variant_id,
            ) = self._resolve_variant_fields(
                identifier=task_identifier,
                provided_label=entry.get("variant_label"),
                provided_variant_id=entry.get("variant_id"),
            )
            record: Dict[str, Any] = {
                "user_id": entry.get("user_id"),
                "task_id": task_id,
                "raw_model_name": entry.get("model_name") or default_model_name,
                "model_name": canonicalize_model_name(
                    entry.get("model_name") or default_model_name
                ),
                "variant_label": variant_label,
                "variant_id": variant_id,
                "subj_overall": entry.get("overall_subjective_score"),
                "combined_score": entry.get("combined_score"),
                "subjective_metadata": entry.get("dimension_breakdowns"),
                "input_indicators": entry.get("input_side_indicators"),
                "output_indicators": entry.get("output_side_indicators"),
            }
            # Map dimension scores with backward compatibility
            # Process new field names first, then fall back to old names if needed
            for source_key, dest_key in SUBJECTIVE_DIMENSIONS.items():
                value = entry.get(source_key)
                if value is not None:
                    # For backward compatibility fields, only use if dest_key not already set
                    if source_key in ("efficiency_score", "frustration_indicator"):
                        if dest_key not in record:
                            record[dest_key] = value
                    else:
                        # New field names take precedence
                        record[dest_key] = value
            records.append(record)

        subjective_df = pd.DataFrame(records)
        self._log_shape("subjective results", subjective_df)
        return subjective_df

    def build_component_subjective_frame(
        self, subjective_df: pd.DataFrame, component_key: str
    ) -> pd.DataFrame:
        """
        Derive a judge-only or rubric-only subjective frame from blended results.

        Args:
            subjective_df (pd.DataFrame): Output of :meth:`load_subjective_results`.
            component_key (str): Component to extract (``"judge"`` or ``"rubric"``).

        Returns:
            pd.DataFrame: Subjective frame with per-dimension component scores and
            recomputed overall/combined scores.
        """
        if subjective_df is None or subjective_df.empty:
            return pd.DataFrame()

        key = component_key.lower()
        if key not in {"judge", "rubric"}:
            raise ValueError("component_key must be 'judge' or 'rubric'.")

        records: List[Dict[str, Any]] = []
        for _, row in subjective_df.iterrows():
            breakdowns = row.get("subjective_metadata") or {}
            record: Dict[str, Any] = {
                "user_id": row.get("user_id"),
                "task_id": row.get("task_id"),
                "model_name": row.get("model_name"),
                "variant_label": row.get("variant_label") or "original",
                "variant_id": row.get("variant_id") or row.get("task_id"),
                "subjective_component": key,
                "subjective_metadata": breakdowns,
                "input_indicators": row.get("input_indicators"),
                "output_indicators": row.get("output_indicators"),
            }

            # Preserve upstream configuration metadata when present
            for meta_field in (
                "generator_model",
                "filter_model",
                "judge_model",
                "dataset_type",
            ):
                if meta_field in row:
                    record[meta_field] = row.get(meta_field)

            # Extract per-dimension component scores
            dimension_values: List[float] = []
            static_reference: Optional[float] = None
            for dim_key, dest_col in DIMENSION_KEY_TO_COLUMN.items():
                dim_details = breakdowns.get(dim_key, {}) or {}
                components = dim_details.get("components", {}) or {}
                component_value = components.get(key)
                if component_value is None:
                    continue
                record[dest_col] = component_value
                dimension_values.append(component_value)
                if static_reference is None:
                    static_reference = dim_details.get("static_correctness_reference")

            # Fallback static correctness from output indicators if available
            if static_reference is None:
                output_indicators = row.get("output_indicators") or {}
                static_reference = output_indicators.get("static_correctness")

            overall = (
                sum(dimension_values) / len(dimension_values)
                if dimension_values
                else None
            )

            # Simple weighted average (0.5/0.5) mirroring default subjective/static blend
            combined = None
            if overall is not None and static_reference is not None:
                combined = (
                    self._clamp(overall) * 0.5 + self._clamp(static_reference) * 0.5
                )

            record["subj_overall"] = (
                self._clamp(overall) if overall is not None else None
            )
            record["combined_score"] = (
                self._clamp(combined) if combined is not None else None
            )

            records.append(record)

        component_df = pd.DataFrame(records)
        self._log_shape(f"{key}-only subjective results", component_df)
        return component_df

    def load_objective_results(
        self, objective_path: str, default_model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load objective evaluation metrics and compute derived statistics.

        Args:
            objective_path (str): Path to the Stage 4 outputs (JSON or CSV).
            default_model_name (Optional[str]): Model name fallback.

        Returns:
            pandas.DataFrame: Rows keyed by ``task_id`` and ``variant_label``
            with Pass@k, attempt counts, and other static metrics.
        """
        path = self._validate_path(objective_path)
        if path.suffix.lower() == ".csv":
            objective_df = self._normalize_objective_csv(path, default_model_name)
            self._log_shape("objective results", objective_df)
            return objective_df

        raw_objective = self._load_json_payload(path)
        if isinstance(raw_objective, dict) and "samples" in raw_objective:
            sample_iterable = raw_objective["samples"]
            default_model_name = default_model_name or raw_objective.get("model_name")
        elif isinstance(raw_objective, list):
            sample_iterable = raw_objective
        else:
            sample_iterable = [raw_objective]

        records: List[Dict[str, Any]] = []
        for sample in sample_iterable:
            normalized = self._normalize_objective_sample(
                sample=sample, default_model_name=default_model_name
            )
            if normalized:
                records.append(normalized)

        objective_df = pd.DataFrame(records)
        self._log_shape("objective results", objective_df)
        return objective_df

    def load_indicator_scores(
        self,
        indicator_path: str,
        default_model_name: Optional[str] = None,
        default_user_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load Stage 5c indicator score outputs.

        Args:
            indicator_path (str): Path to the Stage 5c `indicator_scores_*.json` artifact.
            default_model_name (Optional[str]): Model name fallback when records omit it.
            default_user_id (Optional[str]): User id fallback when records omit it.

        Returns:
            pandas.DataFrame: One row per (user_id, task_id, model_name) containing
            indicator blocks and optional rubric scores.

        Raises:
            ValueError: If the file payload is not a list of records.
        """
        path = self._validate_path(indicator_path)
        raw_payload = self._load_json_payload(path)

        if isinstance(raw_payload, dict):
            # Legacy/proto shapes: allow wrapping under "results".
            raw_payload = raw_payload.get("results", [raw_payload])

        if not isinstance(raw_payload, list):
            raise ValueError(
                "Stage 5c indicator scores must be a list of records. "
                f"Got: {type(raw_payload).__name__} from {indicator_path}"
            )

        attempt_token = "::attempt::"
        records: List[Dict[str, Any]] = []
        for entry in raw_payload:
            if not isinstance(entry, dict):
                continue
            raw_task_id = entry.get("task_id")
            if not raw_task_id:
                continue

            # Stage 5c can emit per-attempt task_ids; strip attempt suffix for stable grouping.
            stable_identifier = str(raw_task_id)
            if attempt_token in stable_identifier:
                stable_identifier = stable_identifier.split(attempt_token, 1)[0]

            task_id, variant_label, variant_id = self._resolve_variant_fields(
                identifier=stable_identifier,
                provided_label=entry.get("variant_label"),
                provided_variant_id=entry.get("variant_id"),
            )

            user_id = entry.get("user_id") or default_user_id
            model_name = entry.get("model_name") or default_model_name
            if not user_id or not model_name:
                continue

            metadata = entry.get("_model_metadata", {}) or {}

            record: Dict[str, Any] = {
                "user_id": user_id,
                "task_id": task_id,
                "raw_task_id": raw_task_id,
                "variant_label": variant_label,
                "variant_id": variant_id,
                "raw_model_name": model_name,
                "model_name": canonicalize_model_name(model_name),
                "input_indicators": entry.get("input_side_indicators"),
                "output_indicators": entry.get("output_side_indicators"),
                "rubric_dimension_scores": entry.get("rubric_dimension_scores"),
                "rubric_dimension_details": entry.get("rubric_dimension_details"),
                "vibe_text_metrics": entry.get("vibe_text_metrics"),
                "vibe_text_metrics_flat": entry.get("vibe_text_metrics_flat"),
                "static_correctness_score": entry.get("static_correctness_score"),
                "static_accuracy_metrics": entry.get("static_accuracy_metrics"),
                "static_error_types": entry.get("static_error_types"),
                # Stage-6 configuration columns (best effort, non-breaking)
                "generator_model": canonicalize_model_name(
                    metadata.get("generator_model_name")
                ),
                "filter_model": canonicalize_model_name(
                    metadata.get("filter_model_name")
                ),
                "evaluated_model": canonicalize_model_name(
                    metadata.get("evaluated_model_name")
                ),
                "prompt_type": metadata.get("prompt_type"),
            }
            records.append(record)

        indicator_df = pd.DataFrame(records)
        self._log_shape("indicator scores", indicator_df)
        return indicator_df

    def load_pairwise_results(
        self, pairwise_path: str, tie_breaker_mode: str = "strict"
    ) -> pd.DataFrame:
        """
        Load Stage 5B pairwise comparison results.

        Args:
            pairwise_path (str): Path to the pairwise comparison JSON/JSONL.
            tie_breaker_mode (str): Tie breaker mode - "strict" (default) or "finegrained".
                If "finegrained", re-computes winners from original_order_result and
                swapped_order_result using confidence scores.

        Returns:
            pandas.DataFrame: Rows representing pairwise comparisons with
            flattened dimension results and aggregated metrics.
        """
        path = self._validate_path(pairwise_path)
        try:
            raw_pairwise = self._load_json_payload(path)
        except Exception as exc:
            raise wrap_pairwise_artifact_load_error(
                exc,
                context=_build_pairwise_load_context(
                    artifact_path=path,
                    failure_stage="load_json_payload",
                ).merged(tie_breaker_mode=tie_breaker_mode),
                message="Failed to read pairwise artifact payload.",
            ) from exc

        if isinstance(raw_pairwise, dict):
            raw_pairwise = raw_pairwise.get("results", [raw_pairwise])
        payload_type = type(raw_pairwise).__name__
        if not isinstance(raw_pairwise, list):
            raise PairwiseArtifactLoadError(
                context=_build_pairwise_load_context(
                    artifact_path=path,
                    failure_stage="payload_shape",
                    payload_type=payload_type,
                ).merged(tie_breaker_mode=tie_breaker_mode),
                message="Pairwise artifact payload must be a list or a dict with 'results'.",
            )

        records: List[Dict[str, Any]] = []
        for record_index, entry in enumerate(raw_pairwise):
            if not isinstance(entry, dict):
                raise PairwiseArtifactLoadError(
                    context=_build_pairwise_load_context(
                        artifact_path=path,
                        failure_stage="record_shape",
                        record_index=record_index,
                        record_count=len(raw_pairwise),
                        payload_type=payload_type,
                    ).merged(tie_breaker_mode=tie_breaker_mode),
                    message="Pairwise artifact record must be a JSON object.",
                    cause_type=type(entry).__name__,
                    cause_message=repr(entry),
                )
            task_id = entry.get("task_id")
            if not task_id:
                raise PairwiseArtifactLoadError(
                    context=_build_pairwise_load_context(
                        artifact_path=path,
                        failure_stage="missing_task_id",
                        entry=entry,
                        record_index=record_index,
                        record_count=len(raw_pairwise),
                        payload_type=payload_type,
                    ).merged(tie_breaker_mode=tie_breaker_mode),
                    message="Pairwise artifact record is missing required task_id.",
                )

            # Resolve variant info from task_id
            try:
                base_task_id, variant_label, variant_id = self._resolve_variant_fields(
                    identifier=task_id
                )
            except Exception as exc:
                raise wrap_pairwise_artifact_load_error(
                    exc,
                    context=_build_pairwise_load_context(
                        artifact_path=path,
                        failure_stage="resolve_variant_fields",
                        entry=entry,
                        record_index=record_index,
                        record_count=len(raw_pairwise),
                        payload_type=payload_type,
                    ).merged(tie_breaker_mode=tie_breaker_mode),
                    message="Failed while resolving pairwise task variant fields.",
                ) from exc

            try:
                record = self._normalize_pairwise_record(
                    entry,
                    base_task_id,
                    variant_label,
                    variant_id=variant_id,
                    tie_breaker_mode=tie_breaker_mode,
                )
            except Exception as exc:
                raise wrap_pairwise_artifact_load_error(
                    exc,
                    context=_build_pairwise_load_context(
                        artifact_path=path,
                        failure_stage="normalize_pairwise_record",
                        entry=entry,
                        record_index=record_index,
                        record_count=len(raw_pairwise),
                        payload_type=payload_type,
                    ).merged(tie_breaker_mode=tie_breaker_mode),
                    message="Failed while normalizing pairwise artifact record.",
                ) from exc
            records.append(record)

        pairwise_df = pd.DataFrame(records)
        self._log_shape("pairwise results", pairwise_df)
        return pairwise_df

    def _normalize_pairwise_record(
        self,
        entry: Dict[str, Any],
        task_id: str,
        variant_label: str,
        variant_id: str,
        tie_breaker_mode: str = "strict",
    ) -> Dict[str, Any]:
        """
        Flatten a single pairwise comparison record into analysis-ready columns.

        Args:
            entry (Dict[str, Any]): Raw pairwise comparison record.
            task_id (str): Normalized task identifier.
            variant_label (str): Variant label (original/personalized).
            tie_breaker_mode (str): Tie breaker mode - "strict" or "finegrained".

        Returns:
            Dict[str, Any]: Flattened record with dimension-level columns.
        """
        model_a = entry.get("model_a_name", "unknown_a")
        model_b = entry.get("model_b_name", "unknown_b")
        win_counts = entry.get("win_counts", {})

        record: Dict[str, Any] = {
            # Preserve raw ordering for debugging and for canonicalization fallbacks.
            "raw_model_a_name": model_a,
            "raw_model_b_name": model_b,
            "raw_model_pair": f"{model_a}_vs_{model_b}",
            "raw_judge_model_name": entry.get("judge_model_name"),
            "user_id": entry.get("user_id"),
            "task_id": task_id,
            "raw_task_id": entry.get("task_id"),
            "variant_label": variant_label,
            # Preserve the unique per-sample identifier so multiple dataset variations
            # of the same base task don't collapse downstream.
            "variant_id": variant_id,
            "model_a_name": model_a,
            "model_b_name": model_b,
            "model_pair": f"{model_a}_vs_{model_b}",
            "overall_winner": entry.get("overall_winner"),
            "overall_winner_label": self._winner_to_label(
                entry.get("overall_winner"), model_a, model_b
            ),
            "win_count_a": win_counts.get("model_a", 0),
            "win_count_b": win_counts.get("model_b", 0),
            "win_count_tie": win_counts.get("tie", 0),
            "judge_model_name": canonicalize_model_name(entry.get("judge_model_name")),
            "input_text": entry.get("input_text"),
        }

        # Extract metadata if available
        metadata = entry.get("_model_metadata", {})
        pairwise_judgment_type = metadata.get("pairwise_judgment_type") or entry.get(
            "pairwise_judgment_type"
        )
        dimension_results = entry.get("dimension_results", {})
        public_metadata = entry.get("metadata", {})
        record["overall_choice"] = public_metadata.get("overall_choice")
        record["human_overall_choice"] = public_metadata.get("human_overall_choice")
        record["human_overall_confidence"] = public_metadata.get(
            "human_overall_confidence"
        )
        annotated_dimensions = metadata.get(
            "annotated_dimensions"
        ) or public_metadata.get("annotated_dimensions")
        if not annotated_dimensions:
            annotated_dimensions = sorted(dimension_results.keys())
        annotated_dimensions = [
            dim for dim in annotated_dimensions if dim in PAIRWISE_DIMENSIONS
        ]
        record["position_swap_enabled"] = metadata.get("position_swap_enabled", True)
        record["bias_measured"] = metadata.get(
            "bias_measured", public_metadata.get("bias_measured", True)
        )
        record["pairwise_judgment_type"] = normalize_pairwise_judgment_type(
            pairwise_judgment_type or PAIRWISE_JUDGMENT_TYPE_PERSONA
        )
        record["pairwise_source_persona"] = metadata.get("source_persona")
        record["annotated_dimensions"] = annotated_dimensions
        record["measured_dimension_count"] = len(annotated_dimensions)

        # Flatten dimension results
        total_bias_detected = 0
        annotated_dimension_set = set(annotated_dimensions)

        for dim_name in PAIRWISE_DIMENSIONS:
            dim_result = dimension_results.get(dim_name, {})
            if dim_name not in annotated_dimension_set:
                winner_value = None
                winner_model = None
                confidence = None
                bias_detected = None
            elif tie_breaker_mode == "finegrained":
                recomputed = self._recompute_winner_with_finegrained_tie_breaker(
                    dim_result, model_a, model_b
                )
                if recomputed is not None:
                    winner_value, winner_model, confidence, bias_detected = recomputed
                else:
                    # Fall back to stored values if recomputation not possible
                    winner_value = dim_result.get("winner", "tie")
                    winner_model = dim_result.get("winner_model")
                    confidence = dim_result.get("confidence")
                    bias_detected = dim_result.get("position_bias_detected")
            else:
                # Use stored values (strict mode)
                winner_value = dim_result.get("winner", "tie")
                winner_model = dim_result.get("winner_model")
                confidence = dim_result.get("confidence")
                bias_detected = dim_result.get("position_bias_detected")

            record[f"dim_{dim_name}_winner"] = winner_value
            record[f"dim_{dim_name}_winner_model"] = winner_model
            record[f"dim_{dim_name}_winner_label"] = self._winner_to_label(
                winner_model, model_a, model_b
            )
            record[f"dim_{dim_name}_confidence"] = confidence
            record[f"dim_{dim_name}_bias_detected"] = bias_detected
            record[f"dim_{dim_name}_rationale"] = dim_result.get("rationale", "")

            if bias_detected:
                total_bias_detected += 1

        record["total_bias_detected"] = total_bias_detected
        measured_dim_count = max(len(annotated_dimension_set), 1)
        record["bias_rate"] = (
            total_bias_detected / measured_dim_count if annotated_dimension_set else 0.0
        )

        # Recompute overall winner and win counts if fine-grained mode was used
        if tie_breaker_mode == "finegrained":
            recomputed_win_counts = {"model_a": 0, "model_b": 0, "tie": 0}
            for dim_name in annotated_dimensions:
                winner_label = record.get(f"dim_{dim_name}_winner_label", "tie")
                if winner_label == "model_a":
                    recomputed_win_counts["model_a"] += 1
                elif winner_label == "model_b":
                    recomputed_win_counts["model_b"] += 1
                else:
                    recomputed_win_counts["tie"] += 1

            # Determine overall winner
            if recomputed_win_counts["model_a"] > recomputed_win_counts["model_b"]:
                record["overall_winner"] = model_a
            elif recomputed_win_counts["model_b"] > recomputed_win_counts["model_a"]:
                record["overall_winner"] = model_b
            else:
                record["overall_winner"] = None

            record["overall_winner_label"] = self._winner_to_label(
                record["overall_winner"], model_a, model_b
            )
            record["win_count_a"] = recomputed_win_counts["model_a"]
            record["win_count_b"] = recomputed_win_counts["model_b"]
            record["win_count_tie"] = recomputed_win_counts["tie"]

        return self._canonicalize_pairwise_record(record)

    def _canonicalize_pairwise_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canonicalize model ordering for a Stage-5b pairwise record.

        Stage 5b artifacts encode wins/losses relative to the record's A/B order, but
        the on-disk directory structure canonicalizes pairs by sorting model tokens.
        To prevent downstream misattribution, we canonicalize A/B ordering here and
        recompute all dependent fields.

        Args:
            record (Dict[str, Any]): Flattened pairwise record from `_normalize_pairwise_record`.

        Returns:
            Dict[str, Any]: Record with canonical `model_a_name/model_b_name` ordering and
            recomputed winner labels/counts.

        Raises:
            ValueError: If model identifiers collide after normalization, or winner fields
                reference models outside the compared pair.
        """
        raw_a = str(record.get("raw_model_a_name") or record.get("model_a_name") or "")
        raw_b = str(record.get("raw_model_b_name") or record.get("model_b_name") or "")
        if not raw_a or not raw_b:
            raise ValueError(
                "Pairwise record is missing model names required for canonicalization. "
                f"raw_model_a_name={raw_a!r} raw_model_b_name={raw_b!r}"
            )

        canonical_raw_a = canonicalize_model_name(raw_a)
        canonical_raw_b = canonicalize_model_name(raw_b)
        norm_a = normalize_token(canonical_raw_a)
        norm_b = normalize_token(canonical_raw_b)
        if norm_a == norm_b and canonical_raw_a == canonical_raw_b:
            raise ValueError(
                "Pairwise record compares two aliases of the same canonical model. "
                f"raw_model_a_name={raw_a!r} raw_model_b_name={raw_b!r} "
                f"canonical_name={canonical_raw_a!r}. "
                "Canonical-name grouping would collapse this comparison."
            )

        key_a = (norm_a, canonical_raw_a, raw_a)
        key_b = (norm_b, canonical_raw_b, raw_b)
        swap = bool(key_a > key_b)

        canonical_a = canonical_raw_b if swap else canonical_raw_a
        canonical_b = canonical_raw_a if swap else canonical_raw_b

        record["pairwise_swapped_to_canonical"] = swap
        record["model_a_name"] = canonical_a
        record["model_b_name"] = canonical_b
        record["model_pair"] = f"{canonical_a}_vs_{canonical_b}"

        def _winner_value_to_model_name(
            winner_value: object, *, raw_model_a: str, raw_model_b: str
        ) -> Optional[str]:
            if winner_value is None:
                return None
            val = str(winner_value).strip()
            if not val or val.lower() == "tie":
                return None
            if val == "A":
                return canonicalize_model_name(raw_model_a)
            if val == "B":
                return canonicalize_model_name(raw_model_b)
            # Some artifacts store the model name directly in the winner field.
            return canonicalize_model_name(val)

        def _validate_winner_model(winner_model: Optional[str]) -> None:
            if winner_model is None:
                return
            if winner_model not in {canonical_raw_a, canonical_raw_b}:
                raise ValueError(
                    "Pairwise record contains winner_model that is not one of the compared "
                    "models after canonicalization. "
                    f"winner_model={winner_model!r} "
                    f"canonical_models=({canonical_raw_a!r},{canonical_raw_b!r}) "
                    f"raw_models=({raw_a!r},{raw_b!r}). "
                    f"raw_task_id={record.get('raw_task_id')!r} user_id={record.get('user_id')!r}"
                )

        # Canonicalize overall winner.
        overall_raw = record.get("overall_winner")
        overall_model = _winner_value_to_model_name(
            overall_raw, raw_model_a=raw_a, raw_model_b=raw_b
        )
        if overall_model is not None and overall_model not in {
            canonical_raw_a,
            canonical_raw_b,
        }:
            raise ValueError(
                "Pairwise record has overall_winner that does not match compared models "
                "after canonicalization. "
                f"overall_winner={overall_raw!r} resolved={overall_model!r} "
                f"canonical_models=({canonical_raw_a!r},{canonical_raw_b!r}) "
                f"raw_models=({raw_a!r},{raw_b!r})."
            )
        record["overall_winner"] = overall_model
        record["overall_winner_label"] = self._winner_to_label(
            overall_model, canonical_a, canonical_b
        )

        annotated_dimension_set = set(record.get("annotated_dimensions") or [])

        # Canonicalize per-dimension outcomes.
        for dim_name in PAIRWISE_DIMENSIONS:
            if dim_name not in annotated_dimension_set:
                record[f"dim_{dim_name}_winner_model"] = None
                record[f"dim_{dim_name}_winner_label"] = None
                record[f"dim_{dim_name}_winner"] = None
                continue
            winner_model = record.get(f"dim_{dim_name}_winner_model")
            if winner_model is None or (
                isinstance(winner_model, float) and pd.isna(winner_model)
            ):
                winner_value = record.get(f"dim_{dim_name}_winner")
                winner_model = _winner_value_to_model_name(
                    winner_value, raw_model_a=raw_a, raw_model_b=raw_b
                )
            else:
                winner_model = canonicalize_model_name(str(winner_model))

            _validate_winner_model(winner_model)

            if winner_model is None:
                record[f"dim_{dim_name}_winner_model"] = None
                record[f"dim_{dim_name}_winner_label"] = "tie"
                record[f"dim_{dim_name}_winner"] = "tie"
            elif winner_model == canonical_a:
                record[f"dim_{dim_name}_winner_model"] = winner_model
                record[f"dim_{dim_name}_winner_label"] = "model_a"
                record[f"dim_{dim_name}_winner"] = "A"
            elif winner_model == canonical_b:
                record[f"dim_{dim_name}_winner_model"] = winner_model
                record[f"dim_{dim_name}_winner_label"] = "model_b"
                record[f"dim_{dim_name}_winner"] = "B"
            else:
                # Should be impossible after validation, but keep a loud guard.
                raise ValueError(
                    "Unexpected canonical winner mapping state for dim="
                    f"{dim_name!r}: winner_model={winner_model!r} "
                    f"canonical=({canonical_a!r},{canonical_b!r})."
                )

        # Recompute win counts from canonical dimension labels for consistency.
        a_wins = 0
        b_wins = 0
        ties = 0
        for dim_name in annotated_dimension_set:
            label = record.get(f"dim_{dim_name}_winner_label")
            if label == "model_a":
                a_wins += 1
            elif label == "model_b":
                b_wins += 1
            else:
                ties += 1
        record["win_count_a"] = int(a_wins)
        record["win_count_b"] = int(b_wins)
        record["win_count_tie"] = int(ties)

        return record

    def _recompute_winner_with_finegrained_tie_breaker(
        self,
        dim_result: Dict[str, Any],
        model_a: str,
        model_b: str,
    ) -> Optional[tuple[str, Optional[str], str, bool]]:
        """
        Re-compute winner using fine-grained tie breaker from original/swapped order results.

        Args:
            dim_result: Dimension result dictionary with original_order_result and
                swapped_order_result.
            model_a: Name of model A.
            model_b: Name of model B.

        Returns:
            Tuple of (winner_value, winner_model, confidence, bias_detected) if
            recomputation is possible, None otherwise.
        """
        original_order = dim_result.get("original_order_result")
        swapped_order = dim_result.get("swapped_order_result")

        if not original_order or not swapped_order:
            return None

        # Extract position winners and confidence
        orig_pos_winner = original_order.get("position_winner", "tie")
        orig_confidence = original_order.get("confidence", "low")
        swapped_pos_winner = swapped_order.get("position_winner", "tie")
        swapped_confidence = swapped_order.get("confidence", "low")

        # Convert position winners to model winners
        # Original: position A = model_a, position B = model_b
        if orig_pos_winner == "A":
            orig_model_winner = model_a
        elif orig_pos_winner == "B":
            orig_model_winner = model_b
        else:
            orig_model_winner = None

        # Swapped: position A = model_b, position B = model_a
        if swapped_pos_winner == "A":
            swapped_model_winner = model_b
        elif swapped_pos_winner == "B":
            swapped_model_winner = model_a
        else:
            swapped_model_winner = None

        # Check if they agree
        if orig_model_winner == swapped_model_winner:
            # They agree - use the agreed winner
            if orig_model_winner is None:
                return ("tie", None, orig_confidence, False)
            winner_value = "A" if orig_model_winner == model_a else "B"
            return (winner_value, orig_model_winner, orig_confidence, False)

        # They disagree - apply fine-grained tie breaker
        # Confidence hierarchy: high > medium > low
        confidence_values = {"high": 3, "medium": 2, "low": 1}
        orig_conf_value = confidence_values.get(orig_confidence.lower(), 1)
        swapped_conf_value = confidence_values.get(swapped_confidence.lower(), 1)

        # Case 1: One order has a confident winner (medium or high) and the other is a tie
        if (
            orig_model_winner is not None
            and orig_conf_value >= 2
            and swapped_model_winner is None
        ):
            winner_value = "A" if orig_model_winner == model_a else "B"
            return (winner_value, orig_model_winner, orig_confidence, True)
        if (
            swapped_model_winner is not None
            and swapped_conf_value >= 2
            and orig_model_winner is None
        ):
            winner_value = "A" if swapped_model_winner == model_a else "B"
            return (winner_value, swapped_model_winner, swapped_confidence, True)

        # Case 2: Both say tie
        if orig_model_winner is None and swapped_model_winner is None:
            return ("tie", None, "low", True)

        # Case 3: One says tie but the other has low confidence winner
        if orig_model_winner is None or swapped_model_winner is None:
            return ("tie", None, "low", True)

        # Case 4: Both have winners - compare confidence scores
        if orig_conf_value > swapped_conf_value:
            # Original has higher confidence
            winner_value = "A" if orig_model_winner == model_a else "B"
            return (winner_value, orig_model_winner, orig_confidence, True)
        elif swapped_conf_value > orig_conf_value:
            # Swapped has higher confidence
            winner_value = "A" if swapped_model_winner == model_a else "B"
            return (winner_value, swapped_model_winner, swapped_confidence, True)
        else:
            # Same confidence - mark as tie
            return ("tie", None, "low", True)

    def _winner_to_label(
        self,
        winner: Optional[str],
        model_a: str,
        model_b: str,
    ) -> str:
        """
        Convert winner model name to a standardized label.

        Args:
            winner (Optional[str]): Winner model name or None for tie.
            model_a (str): Name of model A.
            model_b (str): Name of model B.

        Returns:
            str: 'model_a', 'model_b', or 'tie'.
        """
        if winner is None:
            return "tie"
        if winner == model_a:
            return "model_a"
        if winner == model_b:
            return "model_b"
        # Handle 'A' or 'B' values from raw winner field
        if winner == "A":
            return "model_a"
        if winner == "B":
            return "model_b"
        return "tie"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_profile_record(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested profile fields into analysis-ready columns.

        Args:
            profile (Dict[str, Any]): Raw profile dictionary.

        Returns:
            Dict[str, Any]: Normalized record.
        """
        input_dimensions = profile.get("input_dimensions", {})
        output_dimensions = profile.get("output_dimensions", {})
        task_complexity = input_dimensions.get("task_complexity_gradient", [None, None])
        persona_label = input_dimensions.get("persona_based_task_framing")
        underspec = input_dimensions.get("preference_and_underspecification_handling")
        return {
            "user_id": profile.get("user_id"),
            "user_profile_type": persona_label or profile.get("user_id"),
            "persona_description": profile.get("persona_description"),
            "profile_description": profile.get("description"),
            "input_task_complexity_min": (
                task_complexity[0] if len(task_complexity) > 0 else None
            ),
            "input_task_complexity_max": (
                task_complexity[1] if len(task_complexity) > 1 else None
            ),
            "input_real_world_context": input_dimensions.get(
                "real_world_context_embedding"
            ),
            "input_persona_framing": persona_label,
            "input_underspec_handling": underspec,
            "output_clarity_preference": output_dimensions.get(
                "clarity_and_comprehensibility"
            ),
            "output_workflow_fit_preference": output_dimensions.get("workflow_fit"),
            "output_efficiency_preference": output_dimensions.get(
                "efficiency", output_dimensions.get("workflow_fit")
            ),
            "output_friction_preference": output_dimensions.get(
                "friction_and_frustration"
            ),
            "output_reliability_preference": output_dimensions.get("reliability"),
            "output_persona_context_preference": output_dimensions.get(
                "persona_consistency_and_context_awareness"
            ),
            "output_safety_preference": output_dimensions.get(
                "safety_and_refusal_behavior"
            ),
            "output_anthropomorphism_preference": output_dimensions.get(
                "anthropomorphism"
            ),
            "output_creativity_preference": output_dimensions.get(
                "creativity_and_originality"
            ),
            "output_expressive_style": output_dimensions.get("expressive_style"),
            "raw_input_dimensions": input_dimensions,
            "raw_output_dimensions": output_dimensions,
        }

    def _normalize_objective_csv(
        self, csv_path: Path, default_model_name: Optional[str]
    ) -> pd.DataFrame:
        """
        Normalize a CSV export from Stage 4.

        Args:
            csv_path (Path): CSV path.
            default_model_name (Optional[str]): Model fallback.

        Returns:
            pandas.DataFrame: Normalized rows.
        """
        df = pd.read_csv(csv_path)
        if "task_id" not in df.columns and "sample_id" in df.columns:
            df = df.rename(columns={"sample_id": "task_id"})

        df["raw_model_name"] = df.get("model_name", default_model_name)
        df["model_name"] = df["raw_model_name"].apply(canonicalize_model_name)
        df["variant_label"] = df.apply(
            lambda row: self._normalize_variant_label(
                row.get("variant_label"),
                row.get("variant_id"),
                row.get("task_id"),
            ),
            axis=1,
        )

        df["variant_id"] = df.apply(
            lambda row: self._resolve_variant_fields(
                identifier=row.get("task_id", ""),
                provided_label=row.get("variant_label"),
                provided_variant_id=row.get("variant_id"),
            )[2],
            axis=1,
        )
        df["task_id"] = df["task_id"].apply(
            lambda identifier: self._resolve_variant_fields(identifier)[0]
        )
        return df

    def _normalize_objective_sample(
        self, sample: Dict[str, Any], default_model_name: Optional[str]
    ) -> Dict[str, Any]:
        """
        Normalize a single objective sample dictionary.

        Args:
            sample (Dict[str, Any]): Raw sample metrics.
            default_model_name (Optional[str]): Fallback model identifier.

        Returns:
            Dict[str, Any]: Normalized metrics ready for aggregation.
        """
        # Some legacy per-sample JSONs wrap the actual metrics under a
        # ``metrics`` key. In that case, lift the nested fields to the
        # top level so downstream normalization logic can treat both
        # schemas uniformly.
        if "metrics" in sample and isinstance(sample["metrics"], dict):
            metrics_block = sample["metrics"]
            # Only copy known metric-bearing fields if they are not already set
            for key in ("base", "plus", "pass_at_k", "model_name", "source"):
                if key in metrics_block and key not in sample:
                    sample[key] = metrics_block[key]

        sample_identifier = sample.get("sample_id") or sample.get("task_id")
        if not sample_identifier:
            return {}

        task_id, variant_label, variant_id = self._resolve_variant_fields(
            identifier=sample_identifier,
            provided_label=sample.get("variant_label"),
            provided_variant_id=sample.get("variant_id"),
        )
        record: Dict[str, Any] = {
            "task_id": task_id,
            "variant_label": variant_label,
            "variant_id": variant_id,
            "raw_model_name": sample.get("model_name") or default_model_name,
            "model_name": canonicalize_model_name(
                sample.get("model_name") or default_model_name
            ),
            "objective_source": sample.get("source") or "function_eval",
        }

        for prefix in ("base", "plus"):
            record.update(
                self._extract_pass_metrics(
                    sample.get(prefix, {}), prefix=f"obj_{prefix}"
                )
            )

        record.update(
            self._extract_pass_metrics(
                sample.get("pass_at_k", {}), prefix="obj_overall"
            )
        )
        return record

    def _extract_pass_metrics(
        self, block: Any, prefix: str
    ) -> Dict[str, Optional[float]]:
        """
        Extract pass@k metrics and attempt statistics from a block.

        Args:
            block (Any): Section of the objective JSON that may include
                ``pass_at_k`` and ``per_attempt`` data.
            prefix (str): Column prefix.

        Returns:
            Dict[str, Optional[float]]: Flattened metrics.
        """
        if block is None:
            return {}

        metrics: Dict[str, Optional[float]] = {}
        if isinstance(block, dict):
            pass_metrics = block.get("pass_at_k", {}) if "pass_at_k" in block else block
            if isinstance(pass_metrics, dict):
                for k, value in pass_metrics.items():
                    metrics[f"{prefix}_pass_at_{k}"] = value
            per_attempt = block.get("per_attempt")
            if isinstance(per_attempt, list) and per_attempt:
                attempt_count = len(per_attempt)
                pass_count = sum(1 for attempt in per_attempt if bool(attempt))
                metrics[f"{prefix}_attempt_count"] = attempt_count
                metrics[f"{prefix}_pass_count"] = pass_count
                metrics[f"{prefix}_pass_rate"] = pass_count / attempt_count
        return metrics

    def _resolve_variant_fields(
        self,
        identifier: str,
        provided_label: Optional[str] = None,
        provided_variant_id: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Derive canonical task + variant identifiers from any naming scheme.

        Args:
            identifier (str): Raw task or sample identifier.
            provided_label (Optional[str]): Explicit label when supplied by the
                upstream stage.
            provided_variant_id (Optional[str]): Explicit variant identifier.

        Returns:
            Tuple[str, str, str]: ``(task_id, variant_label, variant_id)``.
        """
        base_identifier = identifier
        variant_segment = None
        if VARIATION_TOKEN in identifier:
            base_identifier, variant_segment = identifier.split(VARIATION_TOKEN, 1)
        variant_id = provided_variant_id or variant_segment or base_identifier
        variant_label = self._normalize_variant_label(
            provided_label, variant_id, identifier
        )
        return base_identifier, variant_label, variant_id

    def _normalize_variant_label(
        self,
        provided_label: Optional[str],
        variant_identifier: Optional[str],
        raw_identifier: Optional[str],
    ) -> str:
        """
        Normalize variant labels to ``original`` or ``personalized``.

        Args:
            provided_label (Optional[str]): Upstream label.
            variant_identifier (Optional[str]): Variant-specific identifier.
            raw_identifier (Optional[str]): Complete raw identifier string.

        Returns:
            str: Canonical variant label.
        """
        label = (provided_label or "").lower()
        if label in VARIANT_FRIENDLY_MAP:
            return VARIANT_FRIENDLY_MAP[label]

        # Heuristics when no explicit label is provided.
        identifier = (raw_identifier or "") + " " + (variant_identifier or "")
        return infer_prompt_type(identifier)

    def _validate_path(self, file_path: str) -> Path:
        """
        Validate that the provided path exists.

        Args:
            file_path (str): Input path.

        Returns:
            Path: Resolved ``Path`` object.

        Raises:
            FileNotFoundError: When the file is missing.
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path

    def _load_json_payload(self, path: Path) -> Any:
        """
        Load JSON or JSON lines content based on file suffix.

        Args:
            path (Path): File path.

        Returns:
            Any: Parsed payload.
        """
        if path.suffix.lower() == ".jsonl":
            return load_jsonl(str(path))
        return load_json(str(path))

    def _log_shape(self, label: str, frame: pd.DataFrame) -> None:
        """
        Emit a debug log describing the resulting DataFrame shape.

        Args:
            label (str): Friendly dataset label.
            frame (pandas.DataFrame): DataFrame being logged.
        """
        self.logger.info("%s loaded with shape %s", label, frame.shape)

    @staticmethod
    def _clamp(value: Optional[float]) -> float:
        """
        Clamp a numeric value into the [0, 1] range.

        Args:
            value (Optional[float]): Value to clamp.

        Returns:
            float: Clamped value or 0.0 when ``value`` is ``None``.
        """
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))
