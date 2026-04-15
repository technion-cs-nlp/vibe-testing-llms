"""
Stage 6: Analysis

Aggregates objective and subjective evaluation outputs, computes user-level
summaries, exports tidy CSVs, and renders publication-friendly figures.

Supports two modes:
1. Single-run analysis (legacy): specific input files provided via arguments.
2. Batch analysis (new): scans directories for results across multiple models/personas.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

# Add src to path to allow direct script execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.vibe_testing.analysis.aggregations import (  # noqa: E402
    AggregationBundle,
    AnalysisDataError,
    run_full_aggregation,
)
from src.vibe_testing.analysis.exporters import (  # noqa: E402
    write_model_overall_summary,
    write_pairwise_dimension_summary,
    write_pairwise_pair_summary,
    write_pairwise_preference_matrix,
    write_pairwise_sample_level,
    write_pairwise_statistical_tests,
    write_pairwise_user_summary,
    write_joint_preference_long_table,
    write_joint_preference_matrix,
    write_joint_preference_judge_agreement_latex,
    write_joint_preference_overall_latex,
    write_joint_preference_streamlit_agreement_latex,
    write_joint_preference_streamlit_dimension_agreement_latex,
    write_joint_preference_rubric_detail_summary,
    write_sample_level_flat,
    write_user_model_deltas,
    write_user_model_variant_summary,
)
from src.vibe_testing.analysis.figures import (  # noqa: E402
    plot_model_ranking,
    plot_objective_vs_subjective_scatter,
    plot_pairwise_by_user,
    plot_pairwise_dimension_comparison,
    plot_pairwise_dimension_heatmap,
    plot_pairwise_forest,
    plot_pairwise_objective_passk,
    plot_pairwise_win_rates,
    PAIRWISE_DIMENSION_LABELS,
    plot_persona_metric_bars,
    plot_personalization_deltas,
    plot_position_bias_rates,
    plot_preference_matrix_heatmap,
    apply_figure_config_overrides,
    apply_figure_dimension_omits,
    plot_joint_preference_matrix_heatmap,
    plot_joint_preference_persona_panels,
    plot_joint_preference_overall_grid,
    plot_vibe_dimension_bars,
    plot_vibe_dimension_by_variant,
)
from src.vibe_testing.analysis.io import (  # noqa: E402
    AnalysisInputLoader,
    count_failed_verification_units,
    PAIRWISE_DIMENSIONS,
)
from src.vibe_testing.analysis.judge_utils import (  # noqa: E402
    filter_human_judges_from_df,
    is_human_judge_token,
    split_judges_by_group,
)
from src.vibe_testing.analysis.joint_preference import (  # noqa: E402
    PROMPT_TYPES as JOINT_PROMPT_TYPES,
    ObjectiveConsistencyError,
    compute_cluster_aware_paired_tests_for_joint_preference,
    compute_joint_preference_matrices,
    compute_joint_preference_long_by_judge,
    compute_objective_accuracy_consistency_metrics,
    format_paired_delta_mark,
)
from src.vibe_testing.analysis.dimension_omits import (  # noqa: E402
    CORRECTNESS_MODES,
    apply_pairwise_dimension_omits,
    apply_subjective_dimension_omits,
    normalize_omit_dimensions,
    recompute_pairwise_overall_winner,
    recompute_pairwise_overall_winner_dimension_weighted,
)
from src.vibe_testing.analysis.pairwise import (  # noqa: E402
    PairwiseAggregationBundle,
    PairwiseAnalysisError,
    _build_objective_lookup,  # noqa: SLF001
    build_pairwise_df_from_indicator_scores,
    compute_pairwise_rubric_detail_summary,
    compute_model_rankings,
    run_pairwise_aggregation,
)
from src.vibe_testing.analysis.judge_agreement import (  # noqa: E402
    JudgeAgreementError,
    compute_judge_agreement_for_joint_preference,
)
from src.vibe_testing.model_names import canonicalize_model_name  # noqa: E402
from src.vibe_testing.pairwise_artifact_diagnostics import (  # noqa: E402
    PairwiseArtifactLoadContext,
    wrap_pairwise_artifact_load_error,
)
from src.vibe_testing.pairwise_judgment_types import (  # noqa: E402
    PAIRWISE_JUDGMENT_TYPE_GENERAL_USER,
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
)
from src.vibe_testing.pathing import (  # noqa: E402
    analysis_stage_dir,
    normalize_token,
)
from src.vibe_testing.ui.pairwise_explorer_stats import (  # noqa: E402
    align_pairwise_judgment_types_to_human_scope,
)
from src.vibe_testing.utils import (  # noqa: E402
    add_common_args,
    get_stage_logger_name,
    load_config,
    load_json,
    resolve_run_context,
    save_json,
    seed_everything,
    setup_logger,
)


@dataclass
class ResultContext:
    """Metadata associated with a specific result file."""

    path: Path
    # Persona directory name (may differ from canonical user_id in user profiles)
    persona: str
    evaluated_model: str
    generator_model: str
    filter_model: str
    judge_model: Optional[str] = None
    dataset_type: Optional[str] = None


def _canonicalize_model_columns(
    frame: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Canonicalize model-like columns before any grouping or deduplication.

    Args:
        frame (pd.DataFrame): Input dataframe.
        columns (List[str]): Columns containing model or judge identifiers.

    Returns:
        pd.DataFrame: Copy of ``frame`` with canonicalized identifiers. When a raw
            companion column is missing, this helper preserves the original values
            under ``raw_<column>`` for debugging and provenance.
    """
    if frame is None or frame.empty:
        return frame

    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        raw_column = f"raw_{column}"
        if raw_column not in out.columns:
            out[raw_column] = out[column]
        out[column] = out[column].apply(canonicalize_model_name)
    return out


def _build_persona_dir_to_user_id_map(
    profile_paths: List[str],
    logger: logging.Logger,
) -> Dict[str, str]:
    """
    Build mapping from persona directory token -> canonical user_id.

    Stage outputs are organized under persona directory tokens (e.g. "novice_user"),
    but the stored artifacts (especially Stage 5b pairwise JSON) often use the
    canonical `user_id` from the user profile (e.g. "python_novice_01"). For
    analyses that join objective/subjective results to pairwise results, we must
    align on the canonical user_id.

    This function attempts to infer the directory token from the profile JSON path
    and reads the contained user_id.
    """

    out: Dict[str, str] = {}
    for raw_path in profile_paths or []:
        p = Path(raw_path)
        if p.suffix.lower() != ".json":
            continue

        persona_dir: Optional[str] = None
        try:
            parts = list(p.parts)
            if "1_profile" in parts:
                idx = parts.index("1_profile")
                if idx > 0:
                    persona_dir = parts[idx - 1]
        except Exception:
            persona_dir = None

        if not persona_dir:
            match = re.search(r"persona-([A-Za-z0-9_-]+)_v\d+\.json$", p.name)
            if match:
                persona_dir = match.group(1)

        if not persona_dir:
            continue

        try:
            payload = load_json(str(p))
        except Exception as exc:
            logger.debug("Skipping user profile path %s (not JSON?): %s", str(p), exc)
            continue

        user_id: Optional[str] = None
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            user_id = payload[0].get("user_id")
        elif isinstance(payload, dict):
            user_id = payload.get("user_id")

        if user_id:
            out[str(persona_dir)] = str(user_id)

    return out


@dataclass
class PairwiseResultContext:
    """Metadata associated with a pairwise comparison result file."""

    path: Path
    model_a: str
    model_b: str
    judge_model: str
    generator_model: str
    filter_model: str
    persona: str
    prompt_type: Optional[str] = None
    pairwise_judgment_type: str = PAIRWISE_JUDGMENT_TYPE_PERSONA
    source_persona: Optional[str] = None


def _pairwise_load_context_from_scan_result(
    ctx: PairwiseResultContext,
    *,
    failure_stage: str,
    tie_breaker_mode: str,
) -> PairwiseArtifactLoadContext:
    """Build structured load context from a discovered Stage-6 pairwise artifact."""

    return PairwiseArtifactLoadContext(
        artifact_path=str(ctx.path),
        failure_stage=failure_stage,
        model_a_name=ctx.model_a,
        model_b_name=ctx.model_b,
        judge_model_name=ctx.judge_model,
        persona=ctx.persona,
        source_persona=ctx.source_persona,
        prompt_type=ctx.prompt_type,
        pairwise_judgment_type=ctx.pairwise_judgment_type,
        generator_model=ctx.generator_model,
        filter_model=ctx.filter_model,
        tie_breaker_mode=tie_breaker_mode,
    )


def _alignment_item_key_for_pairwise_row(row: pd.Series) -> str:
    """Build a judgment-type-agnostic item key for Stage 6 pairwise alignment."""
    return "||".join(
        [
            str(row.get("user_id", "") or "").strip(),
            str(row.get("variant_label", "") or "").strip(),
            str(row.get("model_a_name", "") or "").strip(),
            str(row.get("model_b_name", "") or "").strip(),
            str(row.get("task_id", "") or "").strip(),
            str(row.get("variant_id", "") or "").strip(),
        ]
    )


def _align_pairwise_judgment_types_for_agreement(
    pairwise_df: pd.DataFrame, *, logger: logging.Logger
) -> pd.DataFrame:
    """
    Align judgment types so human-vs-LLM agreement rows share one bucket.

    Human rows default to ``persona`` unless they were explicitly loaded as
    ``general_user`` by the scanner. Matching LLM rows on the same conceptual
    pairwise item are then aligned to that human-derived judgment type.
    """
    if pairwise_df is None or pairwise_df.empty:
        return pairwise_df
    if "judge_model_name" not in pairwise_df.columns:
        return pairwise_df

    frame = pairwise_df.copy()
    frame["__judgment_alignment_item_key__"] = frame.apply(
        _alignment_item_key_for_pairwise_row,
        axis=1,
    )
    before = (
        frame["pairwise_judgment_type"].astype(str).tolist()
        if "pairwise_judgment_type" in frame.columns
        else [PAIRWISE_JUDGMENT_TYPE_PERSONA] * len(frame)
    )
    frame = align_pairwise_judgment_types_to_human_scope(
        frame,
        item_key_column="__judgment_alignment_item_key__",
        judge_column="judge_model_name",
        judgment_type_column="pairwise_judgment_type",
    )
    changed = sum(
        1
        for old, new in zip(
            before,
            frame["pairwise_judgment_type"].astype(str).tolist(),
        )
        if str(old) != str(new)
    )
    if changed > 0:
        logger.info(
            "Aligned pairwise judgment types for agreement analysis on %d row(s).",
            changed,
        )
    return frame.drop(columns=["__judgment_alignment_item_key__"], errors="ignore")


def _validate_joint_preference_agreement_input(
    pairwise_df: pd.DataFrame, *, logger: logging.Logger
) -> None:
    """
    Validate that the pairwise slice is safe for judge-agreement computation.

    This check is intentionally stricter than the downstream pivot logic so Stage 6
    can fail with provenance-rich diagnostics before the lower-level agreement code
    reports a generic duplicate-item collision.

    Raises:
        ValueError: If the input contains conflicting duplicate rows for the same
            conceptual (item, judge, model-pair) key.
    """
    if pairwise_df is None or pairwise_df.empty:
        return

    required = [
        "user_id",
        "variant_label",
        "task_id",
        "variant_id",
        "model_a_name",
        "model_b_name",
        "judge_model_name",
    ]
    missing = [column for column in required if column not in pairwise_df.columns]
    if missing:
        raise ValueError(
            "Pairwise agreement input is missing required columns: "
            + ", ".join(missing)
        )

    frame = pairwise_df.copy()
    duplicate_subset = [
        "user_id",
        "variant_label",
        "task_id",
        "variant_id",
        "model_a_name",
        "model_b_name",
        "judge_model_name",
    ]
    duplicate_mask = frame.duplicated(subset=duplicate_subset, keep=False)
    if not duplicate_mask.any():
        return

    conflict_examples: List[str] = []
    diagnostic_columns = [
        "raw_task_id",
        "task_id",
        "variant_id",
        "variant_label",
        "judge_model_name",
        "overall_winner_label",
        "pairwise_judgment_type",
        "pairwise_source_persona",
        "pairwise_artifact_path",
    ]
    diagnostic_columns = [
        column for column in diagnostic_columns if column in frame.columns
    ]

    for key_values, group in frame.loc[duplicate_mask].groupby(
        duplicate_subset, dropna=False
    ):
        overall_labels = sorted(
            {
                str(value)
                for value in group["overall_winner_label"].dropna().tolist()
                if str(value).strip()
            }
        )
        if len(overall_labels) <= 1:
            continue
        context = dict(zip(duplicate_subset, key_values))
        sample_rows = (
            group[diagnostic_columns].drop_duplicates().head(6).to_dict("records")
            if diagnostic_columns
            else []
        )
        conflict_examples.append(
            "key="
            + repr(context)
            + f" labels={overall_labels!r} sample_rows={sample_rows!r}"
        )

    if conflict_examples:
        logger.error(
            "Detected conflicting duplicate pairwise rows before judge agreement. "
            "Examples: %s",
            " | ".join(conflict_examples[:5]),
        )
        raise ValueError(
            "Conflicting duplicate pairwise rows detected before judge agreement. "
            "Examples: "
            + " | ".join(conflict_examples[:5])
            + (" (truncated)" if len(conflict_examples) > 5 else "")
        )


class ResultScanner:
    """Scans directories to identify available evaluation results."""

    def __init__(self, base_dir: Path, logger: logging.Logger):
        self.base_dir = base_dir
        self.logger = logger

    def scan_objective(self, personas: List[str]) -> Iterator[ResultContext]:
        """Find all objective result files for the given personas."""
        for persona in personas:
            # Handle both raw persona name and normalized directory name
            candidate_dirs = [
                self.base_dir / normalize_token(persona),
                self.base_dir / persona,
            ]
            persona_dir: Optional[Path] = None
            for cand in candidate_dirs:
                if cand.exists():
                    persona_dir = cand
                    break

            if persona_dir is None:
                # Preserve previous logging behaviour for missing personas
                self.logger.warning(
                    "Persona directory not found for '%s' (tried: %s)",
                    persona,
                    ", ".join(str(c) for c in candidate_dirs),
                )
                continue

            obj_root = persona_dir / "4_objective_evaluation"
            if not obj_root.exists():
                continue

            # Pattern: evaluated_model_X/gen_model_Y/filter_model_Z/[prompt_type_W/]type_id/results...
            for eval_model_dir in obj_root.iterdir():
                if not eval_model_dir.is_dir() or not eval_model_dir.name.startswith(
                    "evaluated_model_"
                ):
                    continue

                evaluated_model = eval_model_dir.name.replace("evaluated_model_", "")

                for gen_model_dir in eval_model_dir.iterdir():
                    if not gen_model_dir.is_dir() or not gen_model_dir.name.startswith(
                        "gen_model_"
                    ):
                        continue

                    generator_model = gen_model_dir.name.replace("gen_model_", "")

                    for filter_model_dir in gen_model_dir.iterdir():
                        if (
                            not filter_model_dir.is_dir()
                            or not filter_model_dir.name.startswith("filter_model_")
                        ):
                            continue

                        filter_model = filter_model_dir.name.replace(
                            "filter_model_", ""
                        )

                        # Check if subdirectories are prompt types or run dirs
                        for sub_dir in filter_model_dir.iterdir():
                            if not sub_dir.is_dir():
                                continue

                            # Case 1: Prompt Type subdirectory
                            if sub_dir.name.startswith("prompt_type_"):
                                # Recurse into run dirs inside prompt type dir
                                for run_dir in sub_dir.iterdir():
                                    if not run_dir.is_dir():
                                        continue
                                    ctx = self._find_objective_in_run_dir(
                                        run_dir,
                                        persona,
                                        evaluated_model,
                                        generator_model,
                                        filter_model,
                                    )
                                    if ctx:
                                        yield ctx
                            else:
                                # Case 2: Legacy/Direct run directory
                                ctx = self._find_objective_in_run_dir(
                                    sub_dir,
                                    persona,
                                    evaluated_model,
                                    generator_model,
                                    filter_model,
                                )
                                if ctx:
                                    yield ctx

    def _find_objective_in_run_dir(
        self,
        run_dir: Path,
        persona: str,
        evaluated_model: str,
        generator_model: str,
        filter_model: str,
    ) -> Optional[ResultContext]:
        """Check a directory for objective results."""
        # Try to identify dataset type from run dir name (e.g. function_run_id_...)
        dataset_type = run_dir.name.split("_")[0] if "_" in run_dir.name else "unknown"

        # Look for result files
        found_file = self._find_result_file(run_dir, "function_eval", ["json", "csv"])
        if found_file:
            return ResultContext(
                path=found_file,
                persona=persona,
                evaluated_model=evaluated_model,
                generator_model=generator_model,
                filter_model=filter_model,
                dataset_type=dataset_type,
            )
        return None

    def scan_subjective(self, personas: List[str]) -> Iterator[ResultContext]:
        """Find all subjective result files for the given personas."""
        for persona in personas:
            candidate_dirs = [
                self.base_dir / normalize_token(persona),
                self.base_dir / persona,
            ]
            persona_dir: Optional[Path] = None
            for cand in candidate_dirs:
                if cand.exists():
                    persona_dir = cand
                    break

            if persona_dir is None:
                self.logger.warning(
                    "Persona directory not found for '%s' (tried: %s)",
                    persona,
                    ", ".join(str(c) for c in candidate_dirs),
                )
                continue

            subj_root = persona_dir / "5_subjective_evaluation"
            if not subj_root.exists():
                continue

            # Pattern: evaluated_model_X/sub_judge_model_J/gen_model_Y/filter_model_Z/results...
            for eval_model_dir in subj_root.iterdir():
                if not eval_model_dir.is_dir() or not eval_model_dir.name.startswith(
                    "evaluated_model_"
                ):
                    continue

                evaluated_model = eval_model_dir.name.replace("evaluated_model_", "")

                for judge_dir in eval_model_dir.iterdir():
                    if not judge_dir.is_dir() or not judge_dir.name.startswith(
                        "sub_judge_model_"
                    ):
                        continue

                    judge_model = judge_dir.name.replace("sub_judge_model_", "")

                    for gen_model_dir in judge_dir.iterdir():
                        if (
                            not gen_model_dir.is_dir()
                            or not gen_model_dir.name.startswith("gen_model_")
                        ):
                            continue

                        generator_model = gen_model_dir.name.replace("gen_model_", "")

                        for filter_model_dir in gen_model_dir.iterdir():
                            if (
                                not filter_model_dir.is_dir()
                                or not filter_model_dir.name.startswith("filter_model_")
                            ):
                                continue

                            filter_model = filter_model_dir.name.replace(
                                "filter_model_", ""
                            )

                            # 1. Scan prompt_type_* subdirectories
                            for sub_dir in filter_model_dir.iterdir():
                                if sub_dir.is_dir() and sub_dir.name.startswith(
                                    "prompt_type_"
                                ):
                                    found_file = self._find_result_file(
                                        sub_dir,
                                        "subjective_scores",
                                        ["json", "jsonl"],
                                    )
                                    if found_file:
                                        yield ResultContext(
                                            path=found_file,
                                            persona=persona,
                                            evaluated_model=evaluated_model,
                                            generator_model=generator_model,
                                            filter_model=filter_model,
                                            judge_model=judge_model,
                                        )

                            # 2. Check for legacy/root results (non-recursive)
                            legacy_file = self._find_result_file_non_recursive(
                                filter_model_dir,
                                "subjective_scores",
                                ["json", "jsonl"],
                            )
                            if legacy_file:
                                yield ResultContext(
                                    path=legacy_file,
                                    persona=persona,
                                    evaluated_model=evaluated_model,
                                    generator_model=generator_model,
                                    filter_model=filter_model,
                                    judge_model=judge_model,
                                )

    def scan_indicator_scores(self, personas: List[str]) -> Iterator[ResultContext]:
        """
        Find all Stage 5c indicator score result files for the given personas.

        Directory structure:
        {persona}/5c_indicator_scores/evaluated_model_{m}/gen_model_{g}/filter_model_{f}/
            [prompt_type_{t}/]indicator_scores_*.json

        Args:
            personas: List of persona names to scan for.

        Yields:
            ResultContext: Metadata for each found indicator score artifact.
        """
        for persona in personas:
            candidate_dirs = [
                self.base_dir / normalize_token(persona),
                self.base_dir / persona,
            ]
            persona_dir: Optional[Path] = None
            for cand in candidate_dirs:
                if cand.exists():
                    persona_dir = cand
                    break

            if persona_dir is None:
                self.logger.warning(
                    "Persona directory not found for '%s' (tried: %s)",
                    persona,
                    ", ".join(str(c) for c in candidate_dirs),
                )
                continue

            indicator_root = persona_dir / "5c_indicator_scores"
            if not indicator_root.exists():
                continue

            for eval_model_dir in indicator_root.iterdir():
                if not eval_model_dir.is_dir() or not eval_model_dir.name.startswith(
                    "evaluated_model_"
                ):
                    continue
                evaluated_model = eval_model_dir.name.replace("evaluated_model_", "")

                for gen_model_dir in eval_model_dir.iterdir():
                    if not gen_model_dir.is_dir() or not gen_model_dir.name.startswith(
                        "gen_model_"
                    ):
                        continue
                    generator_model = gen_model_dir.name.replace("gen_model_", "")

                    for filter_model_dir in gen_model_dir.iterdir():
                        if (
                            not filter_model_dir.is_dir()
                            or not filter_model_dir.name.startswith("filter_model_")
                        ):
                            continue
                        filter_model = filter_model_dir.name.replace(
                            "filter_model_", ""
                        )

                        # 1. Scan prompt_type_* subdirectories
                        for sub_dir in filter_model_dir.iterdir():
                            if sub_dir.is_dir() and sub_dir.name.startswith(
                                "prompt_type_"
                            ):
                                found_file = self._find_result_file_non_recursive(
                                    sub_dir, "indicator_scores", ["json"]
                                )
                                if found_file:
                                    yield ResultContext(
                                        path=found_file,
                                        persona=persona,
                                        evaluated_model=evaluated_model,
                                        generator_model=generator_model,
                                        filter_model=filter_model,
                                    )

                        # 2. Check for legacy/root results (non-recursive)
                        legacy_file = self._find_result_file_non_recursive(
                            filter_model_dir, "indicator_scores", ["json"]
                        )
                        if legacy_file:
                            yield ResultContext(
                                path=legacy_file,
                                persona=persona,
                                evaluated_model=evaluated_model,
                                generator_model=generator_model,
                                filter_model=filter_model,
                            )

    def scan_pairwise(
        self,
        personas: List[str],
        reference_persona: Optional[str] = None,
    ) -> Iterator[PairwiseResultContext]:
        """
        Find all pairwise comparison result files for the given personas.

        Directory structure:
        {persona}/5b_pairwise_evaluation/models_{a}_vs_{b}/judge_model_{j}/
            gen_model_{g}/filter_model_{f}/pairwise-comparison_*.json

        Args:
            personas: List of persona names to scan for.

        Yields:
            PairwiseResultContext: Metadata for each found pairwise result file.
        """

        def _scan_filter_root(
            *,
            target_persona: str,
            source_persona: str,
            model_a: str,
            model_b: str,
            judge_model: str,
            generator_model: str,
            filter_model: str,
            filter_root: Path,
            judgment_type: str,
            allow_shared_only: bool,
        ) -> Iterator[PairwiseResultContext]:
            produced_prompt_roots: List[str] = []
            for sub_dir in filter_root.iterdir():
                if not sub_dir.is_dir() or not sub_dir.name.startswith("prompt_type_"):
                    continue
                prompt_type = sub_dir.name.replace("prompt_type_", "", 1)
                if allow_shared_only and (
                    judgment_type != PAIRWISE_JUDGMENT_TYPE_GENERAL_USER
                    or prompt_type not in {"original", "control"}
                ):
                    continue
                found_file = self._find_result_file(
                    sub_dir, "pairwise-comparison", ["json"]
                )
                if found_file:
                    produced_prompt_roots.append(sub_dir.name)
                    yield PairwiseResultContext(
                        path=found_file,
                        model_a=model_a,
                        model_b=model_b,
                        judge_model=judge_model,
                        generator_model=generator_model,
                        filter_model=filter_model,
                        persona=target_persona,
                        prompt_type=prompt_type,
                        pairwise_judgment_type=judgment_type,
                        source_persona=source_persona,
                    )

            if produced_prompt_roots:
                self.logger.debug(
                    "Pairwise prompt roots yielded artifacts: filter_root=%s judgment_type=%s prompt_roots=%s",
                    filter_root,
                    judgment_type,
                    produced_prompt_roots,
                )

            if allow_shared_only:
                return

            legacy_file = self._find_result_file_non_recursive(
                filter_root, "pairwise-comparison", ["json"]
            )
            if legacy_file:
                self.logger.debug(
                    "Pairwise legacy root yielded artifact: filter_root=%s judgment_type=%s artifact=%s",
                    filter_root,
                    judgment_type,
                    legacy_file,
                )
                yield PairwiseResultContext(
                    path=legacy_file,
                    model_a=model_a,
                    model_b=model_b,
                    judge_model=judge_model,
                    generator_model=generator_model,
                    filter_model=filter_model,
                    persona=target_persona,
                    prompt_type=None,
                    pairwise_judgment_type=judgment_type,
                    source_persona=source_persona,
                )

        def _scan_filter_model_layout(
            *,
            target_persona: str,
            source_persona: str,
            model_a: str,
            model_b: str,
            judge_model: str,
            generator_model: str,
            filter_model: str,
            filter_model_dir: Path,
        ) -> Iterator[PairwiseResultContext]:
            judgment_dirs = sorted(
                [
                    entry
                    for entry in filter_model_dir.iterdir()
                    if entry.is_dir() and entry.name.startswith("judgment_type_")
                ]
            )
            judgment_dir_names = [entry.name for entry in judgment_dirs]
            explicit_persona_dir = filter_model_dir / (
                f"judgment_type_{PAIRWISE_JUDGMENT_TYPE_PERSONA}"
            )
            use_root_persona_fallback = not explicit_persona_dir.exists()
            root_prompt_dirs = sorted(
                [
                    entry.name
                    for entry in filter_model_dir.iterdir()
                    if entry.is_dir() and entry.name.startswith("prompt_type_")
                ]
            )
            if explicit_persona_dir.exists() and root_prompt_dirs:
                self.logger.debug(
                    "Ignoring root-level persona fallback because explicit persona branch exists: filter_root=%s explicit_persona_dir=%s root_prompt_dirs=%s",
                    filter_model_dir,
                    explicit_persona_dir,
                    root_prompt_dirs,
                )
            self.logger.debug(
                "Pairwise layout scan: persona=%s source_persona=%s model_pair=%s_vs_%s judge_model=%s generator_model=%s filter_model=%s filter_root=%s judgment_dirs=%s root_prompt_dirs=%s root_persona_fallback=%s",
                target_persona,
                source_persona,
                model_a,
                model_b,
                judge_model,
                generator_model,
                filter_model,
                filter_model_dir,
                judgment_dir_names,
                root_prompt_dirs,
                use_root_persona_fallback,
            )

            for judgment_dir in judgment_dirs:
                judgment_type = judgment_dir.name.replace("judgment_type_", "", 1)
                yield from _scan_filter_root(
                    target_persona=target_persona,
                    source_persona=source_persona,
                    model_a=model_a,
                    model_b=model_b,
                    judge_model=judge_model,
                    generator_model=generator_model,
                    filter_model=filter_model,
                    filter_root=judgment_dir,
                    judgment_type=judgment_type,
                    allow_shared_only=False,
                )

            if use_root_persona_fallback:
                yield from _scan_filter_root(
                    target_persona=target_persona,
                    source_persona=source_persona,
                    model_a=model_a,
                    model_b=model_b,
                    judge_model=judge_model,
                    generator_model=generator_model,
                    filter_model=filter_model,
                    filter_root=filter_model_dir,
                    judgment_type=PAIRWISE_JUDGMENT_TYPE_PERSONA,
                    allow_shared_only=False,
                )

        for persona in personas:
            candidate_dirs = [
                self.base_dir / normalize_token(persona),
                self.base_dir / persona,
            ]
            persona_dir: Optional[Path] = None
            for cand in candidate_dirs:
                if cand.exists():
                    persona_dir = cand
                    break

            if persona_dir is None:
                self.logger.warning(
                    "Persona directory not found for '%s' (tried: %s)",
                    persona,
                    ", ".join(str(c) for c in candidate_dirs),
                )
                continue

            pairwise_root = persona_dir / "5b_pairwise_evaluation"
            if not pairwise_root.exists():
                continue

            # Pattern: models_{a}_vs_{b}/judge_model_{j}/gen_model_{g}/filter_model_{f}/
            for model_pair_dir in pairwise_root.iterdir():
                if not model_pair_dir.is_dir():
                    continue
                if not model_pair_dir.name.startswith("models_"):
                    continue

                # Parse model pair from directory name: models_{a}_vs_{b}
                pair_name = model_pair_dir.name.replace("models_", "")
                if "_vs_" not in pair_name:
                    continue
                model_a, model_b = pair_name.split("_vs_", 1)

                for judge_dir in model_pair_dir.iterdir():
                    if not judge_dir.is_dir():
                        continue
                    if not judge_dir.name.startswith("judge_model_"):
                        continue

                    judge_model = judge_dir.name.replace("judge_model_", "")

                    for gen_model_dir in judge_dir.iterdir():
                        if not gen_model_dir.is_dir():
                            continue
                        if not gen_model_dir.name.startswith("gen_model_"):
                            continue

                        generator_model = gen_model_dir.name.replace("gen_model_", "")

                        for filter_model_dir in gen_model_dir.iterdir():
                            if not filter_model_dir.is_dir():
                                continue
                            if not filter_model_dir.name.startswith("filter_model_"):
                                continue

                            filter_model = filter_model_dir.name.replace(
                                "filter_model_", ""
                            )

                            yield from _scan_filter_model_layout(
                                target_persona=persona,
                                source_persona=persona,
                                model_a=model_a,
                                model_b=model_b,
                                judge_model=judge_model,
                                generator_model=generator_model,
                                filter_model=filter_model,
                                filter_model_dir=filter_model_dir,
                            )

            if not reference_persona or reference_persona == persona:
                continue

            reference_dir = self.base_dir / reference_persona
            if not reference_dir.exists():
                reference_dir = self.base_dir / normalize_token(reference_persona)
            pairwise_root = reference_dir / "5b_pairwise_evaluation"
            if not pairwise_root.exists():
                continue

            for model_pair_dir in pairwise_root.iterdir():
                if not model_pair_dir.is_dir() or not model_pair_dir.name.startswith(
                    "models_"
                ):
                    continue
                pair_name = model_pair_dir.name.replace("models_", "")
                if "_vs_" not in pair_name:
                    continue
                model_a, model_b = pair_name.split("_vs_", 1)

                for judge_dir in model_pair_dir.iterdir():
                    if not judge_dir.is_dir() or not judge_dir.name.startswith(
                        "judge_model_"
                    ):
                        continue
                    judge_model = judge_dir.name.replace("judge_model_", "")

                    for gen_model_dir in judge_dir.iterdir():
                        if (
                            not gen_model_dir.is_dir()
                            or not gen_model_dir.name.startswith("gen_model_")
                        ):
                            continue
                        generator_model = gen_model_dir.name.replace("gen_model_", "")

                        for filter_model_dir in gen_model_dir.iterdir():
                            if (
                                not filter_model_dir.is_dir()
                                or not filter_model_dir.name.startswith("filter_model_")
                            ):
                                continue
                            filter_model = filter_model_dir.name.replace(
                                "filter_model_", ""
                            )
                            shared_root = (
                                filter_model_dir
                                / f"judgment_type_{PAIRWISE_JUDGMENT_TYPE_GENERAL_USER}"
                            )
                            if not shared_root.exists():
                                continue
                            yield from _scan_filter_root(
                                target_persona=persona,
                                source_persona=reference_persona,
                                model_a=model_a,
                                model_b=model_b,
                                judge_model=judge_model,
                                generator_model=generator_model,
                                filter_model=filter_model,
                                filter_root=shared_root,
                                judgment_type=PAIRWISE_JUDGMENT_TYPE_GENERAL_USER,
                                allow_shared_only=True,
                            )

    def _find_result_file(
        self, directory: Path, pattern_hint: str, extensions: List[str]
    ) -> Optional[Path]:
        """Helper to find the most relevant result file in a directory."""
        # recursive search for files matching the hint
        for ext in extensions:
            # Try exact match patterns first if needed, but general search is often robust enough
            matches = list(directory.rglob(f"*{pattern_hint}*.{ext}"))
            if matches:
                # Return the most recently modified file if multiple exist
                selected = sorted(
                    matches, key=lambda p: p.stat().st_mtime, reverse=True
                )[0]
                self.logger.debug(
                    "Selected result artifact: directory=%s pattern_hint=%s extension=%s recursive=%s candidate_count=%d selected=%s",
                    directory,
                    pattern_hint,
                    ext,
                    True,
                    len(matches),
                    selected,
                )
                return selected
        self.logger.debug(
            "No result artifact matched: directory=%s pattern_hint=%s extensions=%s recursive=%s",
            directory,
            pattern_hint,
            extensions,
            True,
        )
        return None

    def _find_result_file_non_recursive(
        self, directory: Path, pattern_hint: str, extensions: List[str]
    ) -> Optional[Path]:
        """Helper to find the most relevant result file in a directory (non-recursive)."""
        for ext in extensions:
            matches = list(directory.glob(f"*{pattern_hint}*.{ext}"))
            if matches:
                # Return the most recently modified file if multiple exist
                selected = sorted(
                    matches, key=lambda p: p.stat().st_mtime, reverse=True
                )[0]
                self.logger.debug(
                    "Selected result artifact: directory=%s pattern_hint=%s extension=%s recursive=%s candidate_count=%d selected=%s",
                    directory,
                    pattern_hint,
                    ext,
                    False,
                    len(matches),
                    selected,
                )
                return selected
        self.logger.debug(
            "No result artifact matched: directory=%s pattern_hint=%s extensions=%s recursive=%s",
            directory,
            pattern_hint,
            extensions,
            False,
        )
        return None


def _load_persona_pairwise_dimension_weights(
    logger: logging.Logger,
    config_paths: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Load per-persona pairwise dimension weights from persona YAML configs.

    This is used for Stage-6-only recomputation of per-sample pairwise winners
    using a dimension-weighted vote.

    Weight source:
    - Numeric fields under persona YAML `output_dimensions`.

    Mapping:
    - clarity_and_comprehensibility -> clarity
    - workflow_fit -> workflow_fit
    - friction_and_frustration -> friction_loss_of_control
    - reliability -> reliability_user_trust
    - anthropomorphism -> anthropomorphism
    - persona_consistency_and_context_awareness -> persona_consistency + context_awareness

    For pairwise dimensions that are not mapped from the YAML, we use a neutral
    default weight of 1.0 to avoid silently dropping dimensions from the vote.

    Args:
        logger: Logger for reporting loaded weights.
        config_paths: Paths to YAML persona configs.

    Returns:
        Dict[str, Dict[str, float]]: Mapping user_id -> {pairwise_dim -> weight}.

    Raises:
        ValueError: If configs are missing required fields or produce invalid weights.
    """
    # Import here to avoid import cycles at module import time.
    from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS  # noqa: WPS433

    mapping = {
        "clarity_and_comprehensibility": ["clarity"],
        "workflow_fit": ["workflow_fit"],
        "friction_and_frustration": ["friction_loss_of_control"],
        "reliability": ["reliability_user_trust"],
        "anthropomorphism": ["anthropomorphism"],
        "persona_consistency_and_context_awareness": [
            "persona_consistency",
            "context_awareness",
        ],
    }

    out: Dict[str, Dict[str, float]] = {}
    for path in config_paths:
        cfg = load_config(path)
        user_id = cfg.get("user_id")
        if not user_id or not str(user_id).strip():
            raise ValueError(f"Persona config is missing 'user_id': {path}")

        output_dimensions = cfg.get("output_dimensions", {})
        if not isinstance(output_dimensions, dict) or not output_dimensions:
            raise ValueError(
                f"Persona config is missing a non-empty 'output_dimensions' dict: {path}"
            )

        weights: Dict[str, float] = {dim: 1.0 for dim in PAIRWISE_DIMENSIONS}
        for yaml_key, dims in mapping.items():
            value = output_dimensions.get(yaml_key)
            if isinstance(value, (int, float)):
                for dim in dims:
                    weights[dim] = float(value)

        # Validate weights are all positive.
        bad = {k: v for k, v in weights.items() if float(v) <= 0}
        if bad:
            raise ValueError(
                f"Non-positive dimension weights for user_id='{user_id}' from '{path}': {bad}"
            )

        user_key = str(user_id)
        if user_key in out:
            raise ValueError(
                f"Duplicate user_id '{user_key}' encountered in persona dimension weight configs."
            )
        out[user_key] = weights

    logger.info(
        "Loaded pairwise dimension weights for %d persona(s): %s",
        len(out),
        ", ".join(sorted(out.keys())),
    )
    return out


def main(args: Optional[List[str]] = None) -> None:
    """Entry point for Stage 6 analysis."""
    parser = argparse.ArgumentParser(
        description="Stage 6: Aggregate objective + subjective results."
    )

    # Input sources (Legacy Mode)
    parser.add_argument(
        "--objective-results",
        help="Path to specific Stage 4 objective metrics file (Legacy single-run mode).",
    )
    parser.add_argument(
        "--subjective-results",
        help="Path to specific Stage 5 subjective evaluation file (Legacy single-run mode).",
    )
    parser.add_argument(
        "--raw-results",
        help=(
            "Legacy alias for Stage 4 objective results directory. "
            "Kept for compatibility with older orchestrators; ignored by batch scanner."
        ),
    )
    parser.add_argument(
        "--vibe-dataset-dir",
        help=(
            "Legacy Stage 3 dataset directory argument. "
            "Accepted for compatibility but not required by Stage 6 analysis."
        ),
    )

    # Input sources (Batch Mode)
    parser.add_argument(
        "--personas",
        nargs="+",
        help="List of persona names to scan for.",
    )
    parser.add_argument(
        "--results-dir",
        help="Base directory to scan for results (defaults to run-base-dir).",
    )
    parser.add_argument(
        "--reference-persona",
        default="novice_user",
        help="Optional reference persona to include results from (defaults to novice_user). Useful when original/control were only run for one persona.",
    )

    # Common inputs
    parser.add_argument(
        "--user-profiles",
        required=True,
        nargs="+",
        help="Path(s) to the structured user profile JSON/JSONL (or list of profiles).",
    )

    # Outputs
    parser.add_argument(
        "--analysis-output-dir",
        type=str,
        default=None,
        help="Optional override for the Stage 6 analysis directory.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=None,
        help="Optional directory for rendered figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="If set, skip generating figures (CSV exports still run).",
    )
    parser.add_argument(
        "--skip-subjective-figures",
        action="store_true",
        help="If set, skip generating standard objective/subjective figures (pairwise figures still run).",
    )
    parser.add_argument(
        "--only-joint-preference-long",
        action="store_true",
        help=(
            "If set, run ONLY the computations required to write "
            "`tables/pairwise_joint/joint_preference_long.csv` and "
            "`tables/pairwise_joint/joint_preference_long_by_judge.csv`, plus the "
            "three LaTeX table exports under `tables/pairwise_joint/`, and "
            "skip all other Stage 6 analysis outputs (tables, figures, summaries). "
            "This mode requires pairwise results to be available and will error "
            "if prerequisites are missing."
        ),
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=None,
        help="Override for the objective metric column to use in plots.",
    )

    # Legacy/compatibility arguments (accepted but not required by Stage 6)
    parser.add_argument(
        "--output-scores-file",
        type=str,
        default=None,
        help=(
            "Optional legacy path for detailed scores JSON. "
            "Stage 6 currently writes standardized CSV/JSON artifacts under analysis-output-dir."
        ),
    )
    parser.add_argument(
        "--output-summary-file",
        type=str,
        default=None,
        help=(
            "Optional legacy path for summary JSON. "
            "Stage 6 writes 'analysis_summary.json' under the tables directory instead."
        ),
    )
    parser.add_argument(
        "--judge-model-config",
        type=str,
        default=None,
        help=(
            "Path to judge model config (accepted for compatibility with older pipelines; "
            "not required for offline analysis)."
        ),
    )

    # Pairwise comparison options
    parser.add_argument(
        "--pairwise-results",
        type=str,
        default=None,
        help="Path to specific Stage 5B pairwise comparison file (Legacy single-run mode).",
    )
    parser.add_argument(
        "--pairwise-source",
        type=str,
        choices=["stage5b", "indicator_scores"],
        default="stage5b",
        help=(
            "Source for pairwise win-rate analysis. "
            "'stage5b' (default) uses LLM-judge pairwise results from Stage 5B. "
            "'indicator_scores' synthesizes pairwise outcomes deterministically from "
            "Stage 5c indicator artifacts (requires per-dimension scalar scores, "
            "typically 'rubric_dimension_scores')."
        ),
    )
    parser.add_argument(
        "--pairwise-indicator-score-field",
        type=str,
        default="rubric_dimension_scores",
        help=(
            "When --pairwise-source=indicator_scores, this is the indicator record field "
            "that stores a per-dimension score dict to compare across models. "
            "Default: rubric_dimension_scores (produced by Stage 5c with --include-rubric)."
        ),
    )
    parser.add_argument(
        "--include-pairwise",
        action="store_true",
        default=None,
        help="Include pairwise comparison analysis. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
        help="Skip pairwise comparison analysis even if results are available.",
    )
    parser.add_argument(
        "--pairwise-tie-breaker",
        type=str,
        choices=["strict", "finegrained"],
        default="strict",
        help=(
            "Tie breaker mode for pairwise comparisons. "
            "'strict' (default): requires both orders to agree, marks as tie if they disagree. "
            "'finegrained': uses confidence scores to break ties when orders disagree. "
            "If one order has higher confidence, its winner is used."
        ),
    )
    parser.add_argument(
        "--pairwise-dimension-weighted-winner",
        action="store_true",
        help=(
            "If set, recompute per-sample pairwise overall winners in Stage 6 using a "
            "dimension-weighted vote derived from persona YAML output_dimensions. "
            "This affects Stage-6-only aggregations and joint preference matrices; "
            "Stage 5B artifacts are unchanged."
        ),
    )
    parser.add_argument(
        "--pairwise-dimension-weighted-configs",
        nargs="+",
        default=None,
        help=(
            "Paths to persona YAML configs used to derive per-dimension weights for "
            "--pairwise-dimension-weighted-winner. If omitted, falls back to "
            "Stage 6 defaults."
        ),
    )
    parser.add_argument(
        "--pairwise-persona-configs",
        nargs="+",
        default=None,
        help=(
            "Alias for --pairwise-dimension-weighted-configs. "
            "Paths to persona YAML configs used to derive per-dimension weights."
        ),
    )
    parser.add_argument(
        "--pairwise-correctness-mode",
        type=str,
        choices=list(CORRECTNESS_MODES),
        default="ignore",
        help=(
            "How to incorporate base-test pass@1 correctness into per-sample pairwise "
            "winner determination. 'ignore' (default): no change. 'dimension': treat "
            "correctness as an additional dimension vote (weight=5 when dimension-weighted). "
            "'gate': if one model is correct and the other is not, the correct model "
            "wins automatically (dimension votes are skipped for that sample)."
        ),
    )
    parser.add_argument(
        "--pairwise-include-plus-correctness",
        action="store_true",
        default=False,
        help=(
            "When set and --pairwise-correctness-mode=dimension, include plus-test "
            "pass@1 as an additional correctness dimension. Its weight (when "
            "dimension-weighted) equals the persona's workflow_fit weight."
        ),
    )
    parser.add_argument(
        "--joint-preference-alpha",
        type=float,
        default=0.05,
        help=(
            "Alpha threshold for '*' significance markers in joint preference matrices "
            "(two-sided binomial test on non-tie outcomes). Default: 0.05."
        ),
    )
    parser.add_argument(
        "--joint-preference-paired-n-permutations",
        type=int,
        default=100,
        help=(
            "Number of paired sign-flip permutations for cluster-aware joint preference "
            "tests (treatment/control vs original). Default: 10000."
        ),
    )
    parser.add_argument(
        "--joint-preference-paired-n-bootstrap",
        type=int,
        default=50,
        help=(
            "Number of bootstrap resamples for confidence intervals on mean deltas in "
            "cluster-aware joint preference tests. Default: 2000."
        ),
    )
    parser.add_argument(
        "--joint-preference-paired-alpha",
        type=float,
        default=0.05,
        help=(
            "Alpha used for confidence intervals in cluster-aware paired joint preference "
            "tests (CI level = 1 - alpha). Default: 0.05."
        ),
    )
    parser.add_argument(
        "--joint-preference-paired-seed",
        type=int,
        default=42,
        help=(
            "Seed used for cluster-aware paired joint preference tests (permutation + "
            "bootstrap). This seed is recorded in analysis_summary.json. Default: 42."
        ),
    )
    parser.add_argument(
        "--joint-preference-equivalence-win-margin",
        type=float,
        default=0.02,
        help=(
            "Equivalence margin (±ε) for control vs original win-rate deltas in "
            "cluster-aware paired tests. Default: 0.02."
        ),
    )
    parser.add_argument(
        "--joint-preference-equivalence-score-margin",
        type=float,
        default=0.05,
        help=(
            "Equivalence margin (±ε) for control vs original ordinal score deltas in "
            "cluster-aware paired tests. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--joint-preference-paired-disable-bh-fdr",
        action="store_true",
        help=(
            "If set, do not compute BH-FDR q-values for paired permutation p-values "
            "in joint preference outputs."
        ),
    )
    parser.add_argument(
        "--joint-preference-judge-agreement-disable",
        action="store_true",
        help=(
            "If set, do not compute LLM-judge agreement analysis for joint preference outputs. "
            "When enabled (default), Stage 6 will compute agreement across multiple judges "
            "and augment `joint_preference_long.csv` with agreement/stability columns."
        ),
    )
    parser.add_argument(
        "--joint-preference-judge-agreement-n-bootstrap",
        type=int,
        default=50,
        help=(
            "Number of bootstrap resamples used for judge agreement confidence intervals "
            "in joint preference outputs. Default: 1000."
        ),
    )
    parser.add_argument(
        "--joint-preference-judge-agreement-seed",
        type=int,
        default=42,
        help=(
            "Seed used for judge agreement bootstrap resampling in joint preference outputs. "
            "Default: 42."
        ),
    )
    parser.add_argument(
        "--figure-font-scale",
        type=float,
        default=None,
        help="Override global figure font scale (matplotlib/seaborn).",
    )
    parser.add_argument(
        "--figure-title-size",
        type=int,
        default=None,
        help="Override axes title font size for all figures.",
    )
    parser.add_argument(
        "--figure-suptitle-size",
        type=int,
        default=None,
        help="Override figure suptitle font size for all figures.",
    )
    parser.add_argument(
        "--figure-label-size",
        type=int,
        default=None,
        help="Override axis label font size for all figures.",
    )
    parser.add_argument(
        "--figure-tick-size",
        type=int,
        default=None,
        help="Override tick label font size for all figures.",
    )
    parser.add_argument(
        "--figure-joint-annot-fontsize",
        type=int,
        default=None,
        help="Override joint preference matrix cell annotation font size.",
    )
    parser.add_argument(
        "--no-figure-titles",
        action="store_true",
        help="If set, strip all titles (axes titles + suptitles) from all figures.",
    )
    parser.add_argument(
        "--omit-figure-dimensions",
        nargs="+",
        default=None,
        help=(
            "Dimension tokens to omit from ALL Stage-6 analysis outputs (tables + figures). "
            "Examples: reliability_user_trust frustration. "
            "Aliases supported: frustration->friction_loss_of_control, efficiency->workflow_fit."
        ),
    )
    parser.add_argument(
        "--only-pairwise-dimension-comparison-with-objective",
        action="store_true",
        help=(
            "If set, skip all non-essential plotting and render ONLY the "
            "`pairwise_dimension_comparison_with_objective*.pdf` figure(s) using the "
            "LIMA/ACL style. This is intended for fast iteration on that figure."
        ),
    )
    parser.add_argument(
        "--figure-save-pad-inches",
        type=float,
        default=None,
        help="Padding (inches) used when saving figures with bbox_inches='tight'.",
    )
    parser.add_argument(
        "--figure-tight-layout-pad",
        type=float,
        default=None,
        help="tight_layout pad used for joint preference matrices and on-save layout tightening.",
    )

    add_common_args(parser)
    parsed_args = parser.parse_args(args)

    seed_everything(parsed_args.seed)
    run_context = resolve_run_context("stage_6_analyze_results", parsed_args)
    log_path = run_context.logs / "stage_6_analyze.log"

    # Determine analysis directory
    run_label = run_context.run_id or "unnamed_run"
    analysis_dir_name = f"analysis_run_{run_label}"
    if parsed_args.pairwise_dimension_weighted_winner:
        analysis_dir_name = f"{analysis_dir_name}_dimension_weighted"
    if parsed_args.pairwise_correctness_mode != "ignore":
        analysis_dir_name = (
            f"{analysis_dir_name}_correctness_{parsed_args.pairwise_correctness_mode}"
        )
    if parsed_args.pairwise_include_plus_correctness:
        analysis_dir_name = f"{analysis_dir_name}_plus_correctness"

    if parsed_args.analysis_output_dir:
        # add date to the analysis output directory
        parsed_args.analysis_output_dir = (
            f"{parsed_args.analysis_output_dir}_{datetime.now().strftime('%Y%m%d')}"
        )
        base_analysis_dir = Path(parsed_args.analysis_output_dir)
    else:
        # Fallback to standard pathing if not provided (mostly for single run)
        base_analysis_dir = analysis_stage_dir(
            parsed_args.run_base_dir,
            parsed_args.user_group or "batch_analysis",
            parsed_args.evaluated_model_name or "multi_model",
            parsed_args.judge_model_name or "multi_judge",
            parsed_args.generator_model_name or "multi_gen",
            parsed_args.filter_model_name or "multi_filter",
        )

    # Always write into an analysis_run_* subdirectory to avoid mixing outputs,
    # and to make weighted runs visually distinct on disk.
    analysis_run_dir = base_analysis_dir / analysis_dir_name

    tables_dir = analysis_run_dir / "tables"
    figures_dir = (
        Path(parsed_args.figure_dir)
        if parsed_args.figure_dir
        else analysis_run_dir / "figures"
    )

    logger = setup_logger(
        str(log_path),
        logger_name=get_stage_logger_name("stage_6_analyze_results"),
    )
    logger.info("Starting Stage 6 analysis")
    logger.info("Analysis Output Directory: %s", analysis_run_dir)

    # Minimal mode: only write joint preference long-form CSVs.
    if parsed_args.only_joint_preference_long:
        if parsed_args.skip_pairwise:
            logger.error(
                "--only-joint-preference-long cannot be combined with --skip-pairwise."
            )
            sys.exit(1)

    # Fast-iteration mode for the LIMA-style objective+dimension comparison figure.
    if parsed_args.only_pairwise_dimension_comparison_with_objective:
        parsed_args.skip_subjective_figures = True
        parsed_args.skip_figures = False
        parsed_args.include_pairwise = True

    # Apply runtime figure configuration overrides (used across all plotting helpers).
    # In minimal mode, we explicitly skip all figure rendering and can avoid importing
    # plotting configuration overhead.
    if not parsed_args.only_joint_preference_long:
        try:
            apply_figure_config_overrides(
                {
                    "font_scale": parsed_args.figure_font_scale,
                    "title_size": parsed_args.figure_title_size,
                    "suptitle_size": parsed_args.figure_suptitle_size,
                    "label_size": parsed_args.figure_label_size,
                    "tick_size": parsed_args.figure_tick_size,
                    "joint_annot_fontsize": parsed_args.figure_joint_annot_fontsize,
                    "show_titles": (not parsed_args.no_figure_titles),
                    "save_pad_inches": parsed_args.figure_save_pad_inches,
                    "tight_layout_pad": parsed_args.figure_tight_layout_pad,
                }
            )
            apply_figure_dimension_omits(parsed_args.omit_figure_dimensions or [])
        except Exception as exc:
            logger.exception("Failed to apply figure configuration overrides: %s", exc)
            sys.exit(1)

    loader = AnalysisInputLoader(logger)
    verification_failures: List[Dict[str, Any]] = []
    seen_objective_dirs: set[Path] = set()

    # Load Profiles
    try:
        # Support multiple profile files
        profile_paths = (
            parsed_args.user_profiles
            if isinstance(parsed_args.user_profiles, list)
            else [parsed_args.user_profiles]
        )

        persona_dir_to_user_id = _build_persona_dir_to_user_id_map(
            profile_paths, logger
        )
        if persona_dir_to_user_id:
            logger.info(
                "Inferred %d persona-dir -> user_id mappings from user profiles.",
                len(persona_dir_to_user_id),
            )

        profile_dfs = []
        for p_path in profile_paths:
            profile_dfs.append(loader.load_user_profiles(p_path))

        if not profile_dfs:
            profiles_df = pd.DataFrame()
        else:
            profiles_df = pd.concat(profile_dfs, ignore_index=True)
            # Basic deduplication if user_ids overlap (keep last)
            profiles_df = profiles_df.drop_duplicates(subset=["user_id"], keep="last")

    except Exception as err:
        logger.exception("Failed to load user profiles: %s", err)
        sys.exit(1)

    # Load Results
    obj_frames = []
    subj_frames = []

    # MODE 1: Explicit Single Files (Legacy)
    #
    # Objective results are loaded even in --only-joint-preference-long mode so we can
    # compute accuracy consistency (flip) metrics for joint preference exports when
    # objective artifacts are available.
    if parsed_args.objective_results:
        logger.info(
            "Loading specific objective file: %s", parsed_args.objective_results
        )
        df = loader.load_objective_results(
            parsed_args.objective_results,
            default_model_name=parsed_args.evaluated_model_name
            or parsed_args.model_name,
        )
        # Inject metadata if known from args (for legacy consistency)
        if parsed_args.generator_model_name:
            df["generator_model"] = parsed_args.generator_model_name
        if parsed_args.filter_model_name:
            df["filter_model"] = parsed_args.filter_model_name
        obj_frames.append(df)

    if not parsed_args.only_joint_preference_long and parsed_args.subjective_results:
        logger.info(
            "Loading specific subjective file: %s", parsed_args.subjective_results
        )
        df = loader.load_subjective_results(
            parsed_args.subjective_results,
            default_model_name=parsed_args.evaluated_model_name
            or parsed_args.model_name,
        )
        # Inject metadata
        if parsed_args.generator_model_name:
            df["generator_model"] = parsed_args.generator_model_name
        if parsed_args.filter_model_name:
            df["filter_model"] = parsed_args.filter_model_name
        if parsed_args.judge_model_name:
            df["judge_model"] = parsed_args.judge_model_name
        subj_frames.append(df)

    # MODE 2: Batch Scan
    if parsed_args.personas:
        scan_root = Path(parsed_args.results_dir or parsed_args.run_base_dir)
        logger.info(
            "Scanning for results in %s for personas: %s",
            scan_root,
            parsed_args.personas,
        )
        scanner = ResultScanner(scan_root, logger)

        # OPTIMIZATION: Include reference persona in objective scan to allow reusing
        # original/control evaluations.
        scan_personas_objective = list(parsed_args.personas)
        if (
            parsed_args.reference_persona
            and parsed_args.reference_persona not in scan_personas_objective
        ):
            logger.info(
                "Including reference persona '%s' in objective results scan",
                parsed_args.reference_persona,
            )
            scan_personas_objective.append(parsed_args.reference_persona)

        # Scan Objective (even in --only-joint-preference-long mode)
        for ctx in scanner.scan_objective(scan_personas_objective):
            logger.info("Found objective results: %s", ctx.path)
            try:
                df = loader.load_objective_results(
                    str(ctx.path), default_model_name=ctx.evaluated_model
                )
                # Objective result files do not reliably include user_id. They are stored
                # under a persona directory token (e.g. "novice_user"), but pairwise results
                # and user profiles often use a canonical user_id (e.g. "python_novice_01").
                # Map persona-dir -> canonical user_id when possible so downstream joins
                # (e.g. objective flip metrics onto joint-preference tables) are aligned.
                stable_user_id = persona_dir_to_user_id.get(
                    str(ctx.persona), str(ctx.persona)
                )
                if "user_id" not in df.columns:
                    df["user_id"] = stable_user_id
                else:
                    df["user_id"] = df["user_id"].fillna(stable_user_id)
                    if (
                        stable_user_id != str(ctx.persona)
                        and (df["user_id"].astype(str) == str(ctx.persona)).all()
                    ):
                        df["user_id"] = stable_user_id
                df["generator_model"] = ctx.generator_model
                df["filter_model"] = ctx.filter_model
                df["dataset_type"] = ctx.dataset_type

                # Enforce the scanned evaluated-model name, but keep it canonicalized so
                # objective/pairwise joins do not break on directory aliases like
                # "gpt5" vs "gpt-5.1" or "gpt-oss-low-effort" vs "gpt-oss-20b".
                df["model_name"] = canonicalize_model_name(ctx.evaluated_model)

                obj_frames.append(df)

                # Count verification-gated units (failed Stage-3 verification) for this run.
                # The metrics file lives under <run_dir>/function_eval/*.json|csv, so the
                # containing directory is the stable place to scan for per-sample artifacts.
                function_eval_dir = ctx.path.parent
                if function_eval_dir not in seen_objective_dirs:
                    seen_objective_dirs.add(function_eval_dir)
                    failed_count = count_failed_verification_units(function_eval_dir)
                    if failed_count:
                        verification_failures.append(
                            {
                                "persona": ctx.persona,
                                "evaluated_model": ctx.evaluated_model,
                                "generator_model": ctx.generator_model,
                                "filter_model": ctx.filter_model,
                                "dataset_type": ctx.dataset_type,
                                "failed_verification_units": int(failed_count),
                                "function_eval_dir": str(function_eval_dir),
                            }
                        )
            except Exception as e:
                logger.error("Failed to load %s: %s", ctx.path, e)

        # Scan Subjective (persona-specific), unless we are in minimal mode.
        if not parsed_args.only_joint_preference_long:
            for ctx in scanner.scan_subjective(parsed_args.personas):
                logger.info("Found subjective results: %s", ctx.path)
                try:
                    df = loader.load_subjective_results(
                        str(ctx.path), default_model_name=ctx.evaluated_model
                    )
                    stable_user_id = persona_dir_to_user_id.get(
                        str(ctx.persona), str(ctx.persona)
                    )
                    if "user_id" not in df.columns:
                        df["user_id"] = stable_user_id
                    else:
                        df["user_id"] = df["user_id"].fillna(stable_user_id)
                        if (
                            stable_user_id != str(ctx.persona)
                            and (df["user_id"].astype(str) == str(ctx.persona)).all()
                        ):
                            df["user_id"] = stable_user_id
                    df["generator_model"] = ctx.generator_model
                    df["filter_model"] = ctx.filter_model
                    df["judge_model"] = ctx.judge_model

                    # Enforce the scanned evaluated-model name in canonical form so
                    # downstream joins match pairwise/objective normalization.
                    df["model_name"] = canonicalize_model_name(ctx.evaluated_model)

                    subj_frames.append(df)
                except Exception as e:
                    logger.error("Failed to load %s: %s", ctx.path, e)

    # Load Pairwise Comparison Results
    pairwise_frames = []
    all_indicator_df = pd.DataFrame()

    if parsed_args.pairwise_source == "indicator_scores":
        if parsed_args.pairwise_results:
            raise SystemExit(
                "--pairwise-results is only supported with --pairwise-source=stage5b."
            )
        if parsed_args.personas and not parsed_args.skip_pairwise:
            scan_root = Path(parsed_args.results_dir or parsed_args.run_base_dir)
            scanner = ResultScanner(scan_root, logger)

            indicator_frames = []
            for ctx in scanner.scan_indicator_scores(parsed_args.personas):
                logger.info("Found indicator scores: %s", ctx.path)
                try:
                    df = loader.load_indicator_scores(
                        str(ctx.path),
                        default_model_name=ctx.evaluated_model,
                        default_user_id=ctx.persona,
                    )
                    df["generator_model"] = ctx.generator_model
                    df["filter_model"] = ctx.filter_model
                    df["evaluated_model"] = canonicalize_model_name(
                        ctx.evaluated_model
                    )
                    indicator_frames.append(df)
                except Exception as e:
                    logger.error("Failed to load %s: %s", ctx.path, e)

            if indicator_frames:
                all_indicator_df = pd.concat(indicator_frames, ignore_index=True)
                synthetic_pairwise = build_pairwise_df_from_indicator_scores(
                    all_indicator_df,
                    score_field=parsed_args.pairwise_indicator_score_field,
                    judge_model_name="indicator_scores",
                )
                pairwise_frames.append(synthetic_pairwise)
    else:
        # MODE 1: Explicit Single File (Legacy)
        if parsed_args.pairwise_results:
            logger.info(
                "Loading specific pairwise file: %s", parsed_args.pairwise_results
            )
            try:
                logger.info(
                    "Attempting pairwise artifact load: artifact=%s tie_breaker_mode=%s",
                    parsed_args.pairwise_results,
                    parsed_args.pairwise_tie_breaker,
                )
                df = loader.load_pairwise_results(
                    parsed_args.pairwise_results,
                    tie_breaker_mode=parsed_args.pairwise_tie_breaker,
                )
                df["pairwise_artifact_path"] = str(parsed_args.pairwise_results)
                if parsed_args.generator_model_name:
                    df["generator_model"] = parsed_args.generator_model_name
                if parsed_args.filter_model_name:
                    df["filter_model"] = parsed_args.filter_model_name
                if parsed_args.judge_model_name:
                    df["judge_model_name"] = parsed_args.judge_model_name
                if "pairwise_judgment_type" not in df.columns:
                    df["pairwise_judgment_type"] = PAIRWISE_JUDGMENT_TYPE_PERSONA
                pairwise_frames.append(df)
                logger.info(
                    "Loaded pairwise artifact successfully: artifact=%s rows=%d",
                    parsed_args.pairwise_results,
                    len(df),
                )
            except Exception as exc:
                logger.exception(
                    "Failed to load explicit pairwise artifact: artifact=%s",
                    parsed_args.pairwise_results,
                )
                raise wrap_pairwise_artifact_load_error(
                    exc,
                    context=PairwiseArtifactLoadContext(
                        artifact_path=str(parsed_args.pairwise_results),
                        failure_stage="stage6_explicit_pairwise_load",
                        tie_breaker_mode=parsed_args.pairwise_tie_breaker,
                    ),
                    message="Stage 6 failed to load the requested explicit pairwise artifact.",
                ) from exc

        # MODE 2: Batch Scan for Pairwise
        if parsed_args.personas and not parsed_args.skip_pairwise:
            scan_root = Path(parsed_args.results_dir or parsed_args.run_base_dir)
            scanner = ResultScanner(scan_root, logger)
            pairwise_artifacts_attempted = 0
            pairwise_artifacts_loaded = 0

            for ctx in scanner.scan_pairwise(
                parsed_args.personas,
                reference_persona=parsed_args.reference_persona,
            ):
                pairwise_artifacts_attempted += 1
                logger.info(
                    "Attempting discovered pairwise artifact load: artifact=%s persona=%s source_persona=%s prompt_type=%s judge_model=%s model_pair=%s_vs_%s judgment_type=%s tie_breaker_mode=%s",
                    ctx.path,
                    ctx.persona,
                    ctx.source_persona or ctx.persona,
                    ctx.prompt_type,
                    ctx.judge_model,
                    ctx.model_a,
                    ctx.model_b,
                    ctx.pairwise_judgment_type,
                    parsed_args.pairwise_tie_breaker,
                )
                try:
                    df = loader.load_pairwise_results(
                        str(ctx.path), tie_breaker_mode=parsed_args.pairwise_tie_breaker
                    )
                    stable_user_id = persona_dir_to_user_id.get(
                        str(ctx.persona), str(ctx.persona)
                    )
                    df["user_id"] = stable_user_id
                    df["generator_model"] = ctx.generator_model
                    df["filter_model"] = ctx.filter_model
                    df["pairwise_judgment_type"] = ctx.pairwise_judgment_type
                    df["pairwise_source_persona"] = ctx.source_persona or ctx.persona
                    df["pairwise_artifact_path"] = str(ctx.path)
                    # Note: model_a_name, model_b_name, judge_model_name already in data
                    pairwise_frames.append(df)
                    pairwise_artifacts_loaded += 1
                    logger.info(
                        "Loaded discovered pairwise artifact successfully: artifact=%s rows=%d attempted=%d loaded=%d failed=%d",
                        ctx.path,
                        len(df),
                        pairwise_artifacts_attempted,
                        pairwise_artifacts_loaded,
                        pairwise_artifacts_attempted - pairwise_artifacts_loaded,
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to load discovered pairwise artifact: artifact=%s persona=%s source_persona=%s prompt_type=%s judge_model=%s model_pair=%s_vs_%s judgment_type=%s",
                        ctx.path,
                        ctx.persona,
                        ctx.source_persona or ctx.persona,
                        ctx.prompt_type,
                        ctx.judge_model,
                        ctx.model_a,
                        ctx.model_b,
                        ctx.pairwise_judgment_type,
                    )
                    raise wrap_pairwise_artifact_load_error(
                        exc,
                        context=_pairwise_load_context_from_scan_result(
                            ctx,
                            failure_stage="stage6_scan_pairwise_load",
                            tie_breaker_mode=parsed_args.pairwise_tie_breaker,
                        ),
                        message="Stage 6 failed while loading a discovered pairwise artifact.",
                    ) from exc

            if pairwise_artifacts_attempted > 0:
                logger.info(
                    "Pairwise scan summary: attempted=%d loaded=%d failed=%d",
                    pairwise_artifacts_attempted,
                    pairwise_artifacts_loaded,
                    pairwise_artifacts_attempted - pairwise_artifacts_loaded,
                )

    # Combine Results
    if not obj_frames and not subj_frames and not pairwise_frames:
        logger.error(
            "No results found. Provide specific files or valid personas/paths for scanning."
        )
        sys.exit(1)

    all_objective_df = (
        pd.concat(obj_frames, ignore_index=True) if obj_frames else pd.DataFrame()
    )
    all_objective_df = _canonicalize_model_columns(
        all_objective_df,
        ["model_name", "generator_model", "filter_model"],
    )
    if not all_objective_df.empty:
        # Deduplicate to prevent double-counting when both legacy and prompt-specific files exist.
        # Prefer the first one found (scanner yields prompt-specific then legacy).
        count_before = len(all_objective_df)
        objective_subset = ["task_id", "model_name", "variant_label"]
        # If variant_id exists, it disambiguates multiple dataset variations of the same base task.
        if "variant_id" in all_objective_df.columns:
            objective_subset.insert(1, "variant_id")
        # If user_id exists, keep per-persona objective evaluations distinct.
        if "user_id" in all_objective_df.columns:
            objective_subset.append("user_id")
        all_objective_df = all_objective_df.drop_duplicates(
            subset=objective_subset, keep="first"
        )
        if len(all_objective_df) < count_before:
            logger.info(
                "Deduplicated objective results: %d -> %d rows",
                count_before,
                len(all_objective_df),
            )

    all_subjective_df = (
        pd.concat(subj_frames, ignore_index=True) if subj_frames else pd.DataFrame()
    )
    all_subjective_df = _canonicalize_model_columns(
        all_subjective_df,
        ["model_name", "generator_model", "filter_model", "judge_model"],
    )
    if not all_subjective_df.empty:
        count_before = len(all_subjective_df)
        # For subjective, include user_id and judge_model in the key
        subset = ["task_id", "model_name", "variant_label", "user_id"]
        if "variant_id" in all_subjective_df.columns:
            subset.insert(1, "variant_id")
        if "judge_model" in all_subjective_df.columns:
            subset.append("judge_model")
        all_subjective_df = all_subjective_df.drop_duplicates(
            subset=subset, keep="first"
        )
        if len(all_subjective_df) < count_before:
            logger.info(
                "Deduplicated subjective results: %d -> %d rows",
                count_before,
                len(all_subjective_df),
            )

    all_pairwise_df = (
        pd.concat(pairwise_frames, ignore_index=True)
        if pairwise_frames
        else pd.DataFrame()
    )
    all_pairwise_df = _canonicalize_model_columns(
        all_pairwise_df,
        [
            "model_a_name",
            "model_b_name",
            "judge_model_name",
            "generator_model",
            "filter_model",
            "pairwise_source_persona",
        ],
    )
    if not all_pairwise_df.empty:
        count_before = len(all_pairwise_df)
        # For pairwise, include model pair components, judge, and user.
        #
        # IMPORTANT:
        # - `variant_id` is NOT globally unique (e.g. control variations can be "control_1/control_2"
        #   repeated across many base tasks). Dedup must therefore NOT use variant_id alone.
        # - The safest unique identifier is `raw_task_id` (base task + variation suffix).
        # - If raw_task_id is missing, fall back to (task_id, variant_id) when variant_id exists,
        #   otherwise just task_id.
        if "raw_task_id" in all_pairwise_df.columns:
            subset = [
                "raw_task_id",
                "model_a_name",
                "model_b_name",
                "variant_label",
                "user_id",
            ]
        elif "variant_id" in all_pairwise_df.columns:
            subset = [
                "task_id",
                "variant_id",
                "model_a_name",
                "model_b_name",
                "variant_label",
                "user_id",
            ]
        else:
            subset = [
                "task_id",
                "model_a_name",
                "model_b_name",
                "variant_label",
                "user_id",
            ]
        if "judge_model_name" in all_pairwise_df.columns:
            subset.append("judge_model_name")
        if "pairwise_judgment_type" in all_pairwise_df.columns:
            subset.append("pairwise_judgment_type")
        all_pairwise_df = all_pairwise_df.drop_duplicates(subset=subset, keep="first")
        if len(all_pairwise_df) < count_before:
            logger.info(
                "Deduplicated pairwise results: %d -> %d rows",
                count_before,
                len(all_pairwise_df),
            )
        _assert_pairwise_rows_canonical(all_pairwise_df, logger=logger)

    all_indicator_df = _canonicalize_model_columns(
        all_indicator_df,
        ["model_name", "generator_model", "filter_model", "evaluated_model"],
    )

    logger.info(
        "Total Data Loaded - Objective Rows: %d, Subjective Rows: %d, Pairwise Rows: %d",
        len(all_objective_df),
        len(all_subjective_df),
        len(all_pairwise_df),
    )

    # ------------------------------------------------------------------
    # Dimension omission (applies to ALL Stage-6 artifacts, not only figures)
    # ------------------------------------------------------------------
    omit_pairwise: set[str] = set()
    omit_subjective: set[str] = set()
    if parsed_args.omit_figure_dimensions:
        try:
            omit_pairwise, omit_subjective = normalize_omit_dimensions(
                parsed_args.omit_figure_dimensions
            )
            if omit_pairwise or omit_subjective:
                logger.info(
                    "Omitting dimensions from analysis outputs. Pairwise=%s; Subjective=%s",
                    sorted(omit_pairwise),
                    sorted(omit_subjective),
                )
                if not all_subjective_df.empty:
                    all_subjective_df = apply_subjective_dimension_omits(
                        all_subjective_df,
                        omit_pairwise_keys=omit_pairwise,
                        omit_subjective_cols=omit_subjective,
                        strip_subjective_metadata=True,
                    )
        except Exception as exc:
            logger.exception("Failed to apply dimension omission: %s", exc)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Pairwise sanitization (Stage 6 only)
    #
    # Goals:
    # - Recompute overall winners using only non-omitted dimensions.
    # - Exclude per-dimension evaluation/parsing errors from the vote count.
    # - If a sample has no usable (non-error) dimensions, exclude it from
    #   win/tie-rate analysis rather than counting it as a tie.
    # - Optionally incorporate pass@1 correctness into winner determination.
    # ------------------------------------------------------------------
    dimension_weights_by_user: Optional[Dict[str, Dict[str, float]]] = None
    if parsed_args.pairwise_dimension_weighted_winner:
        try:
            # Check for alias
            config_paths_arg = parsed_args.pairwise_dimension_weighted_configs
            if config_paths_arg is None:
                config_paths_arg = parsed_args.pairwise_persona_configs

            if config_paths_arg:
                config_paths = list(config_paths_arg)
            else:
                # Fallback to defaults
                config_paths = [
                    "configs/user_profiles/novice_user_2.yaml",
                    "configs/user_profiles/intermediate_learner.yaml",
                    "configs/user_profiles/researcher_user_2.yaml",
                    "configs/user_profiles/advanced_developer.yaml",
                ]
                logger.info(
                    "No config paths provided for --pairwise-dimension-weighted-winner; "
                    "using Stage 6 defaults: %s",
                    config_paths,
                )

            dimension_weights_by_user = _load_persona_pairwise_dimension_weights(
                logger, config_paths
            )
        except Exception as exc:
            logger.exception("Failed to load persona dimension weight configs: %s", exc)
            sys.exit(1)

    # Build objective lookup for correctness-aware recomputation (and later
    # aggregation).  Built once here and reused in the pairwise analysis loop.
    correctness_mode = str(parsed_args.pairwise_correctness_mode)
    include_plus_correctness = bool(parsed_args.pairwise_include_plus_correctness)
    cached_objective_lookup = (
        _build_objective_lookup(all_objective_df)
        if not all_objective_df.empty
        else None
    )

    if correctness_mode != "ignore":
        if cached_objective_lookup is None or not cached_objective_lookup:
            logger.error(
                "--pairwise-correctness-mode=%s requires objective results, "
                "but no objective data was loaded. Provide Stage 4 objective "
                "results via --personas/--results-dir or --objective-results.",
                correctness_mode,
            )
            sys.exit(1)
        logger.info(
            "Correctness-aware pairwise recomputation: mode=%s, "
            "include_plus_correctness=%s, objective_lookup_size=%d",
            correctness_mode,
            include_plus_correctness,
            len(cached_objective_lookup),
        )

    if not all_pairwise_df.empty:
        try:
            if dimension_weights_by_user is not None:
                all_pairwise_df = recompute_pairwise_overall_winner_dimension_weighted(
                    all_pairwise_df,
                    dimension_weights_by_user=dimension_weights_by_user,
                    omit_pairwise_keys=omit_pairwise,
                    exclude_evaluation_error_dimensions=True,
                    correctness_mode=correctness_mode,
                    objective_lookup=cached_objective_lookup,
                    include_plus_correctness=include_plus_correctness,
                )
            else:
                all_pairwise_df = recompute_pairwise_overall_winner(
                    all_pairwise_df,
                    omit_pairwise_keys=omit_pairwise,
                    exclude_evaluation_error_dimensions=True,
                    correctness_mode=correctness_mode,
                    objective_lookup=cached_objective_lookup,
                    include_plus_correctness=include_plus_correctness,
                )
        except Exception as exc:
            logger.exception("Failed to recompute pairwise overall winners: %s", exc)
            sys.exit(1)

        if "pairwise_valid_for_overall" not in all_pairwise_df.columns:
            logger.error(
                "Internal error: missing 'pairwise_valid_for_overall' after "
                "recomputing pairwise winners."
            )
            sys.exit(1)

        invalid_count = int((~all_pairwise_df["pairwise_valid_for_overall"]).sum())
        if invalid_count > 0:
            logger.warning(
                "Excluding %d pairwise sample(s) with no usable dimensions "
                "(evaluation/parsing errors across all included dimensions).",
                invalid_count,
            )
            all_pairwise_df = all_pairwise_df[
                all_pairwise_df["pairwise_valid_for_overall"]
            ].copy()

        # Apply pairwise dimension omissions to exported artifacts AFTER recomputing winners.
        if omit_pairwise:
            try:
                all_pairwise_df = apply_pairwise_dimension_omits(
                    all_pairwise_df, omit_pairwise_keys=omit_pairwise
                )
            except Exception as exc:
                logger.exception("Failed to apply pairwise dimension omission: %s", exc)
                sys.exit(1)

        # Preserve the originally loaded scope partitioning here. Some downstream
        # agreement-oriented consumers may choose to align judgment types locally,
        # but mutating the full Stage-6 frame before the per-judgment-type export
        # split can collapse shared-reference rows into persona slices.

    # ------------------------------------------------------------------
    # Minimal joint preference mode (write the two long-form CSVs + two LaTeX tables)
    # ------------------------------------------------------------------
    objective_flip_baseline_user_id: Optional[str] = None
    if (
        parsed_args.reference_persona
        and all_objective_df is not None
        and not all_objective_df.empty
        and "user_id" in all_objective_df.columns
        and "variant_label" in all_objective_df.columns
    ):
        ref_user_id = persona_dir_to_user_id.get(str(parsed_args.reference_persona))
        orig_user_ids = sorted(
            {
                str(u)
                for u in all_objective_df.loc[
                    all_objective_df["variant_label"] == "original", "user_id"
                ]
                .dropna()
                .unique()
                .tolist()
            }
        )
        if (
            ref_user_id
            and len(orig_user_ids) == 1
            and str(ref_user_id) in orig_user_ids
        ):
            objective_flip_baseline_user_id = str(ref_user_id)
            logger.info(
                "Using reference baseline user_id=%s for objective flip metrics "
                "(original objective results were only found for this user).",
                objective_flip_baseline_user_id,
            )

    # ------------------------------------------------------------------
    # Separate human judges from LLM judges in pairwise data.
    # Human judges are used ONLY in:
    #   - joint_preference_long_by_judge.csv (per-judge rows)
    #   - agreement LaTeX/CSV outputs (human vs LLM comparison)
    # All other computations (win-rate matrices, paired tests, per-pair
    # aggregations, figures) use LLM judges exclusively.
    # ------------------------------------------------------------------
    if not all_pairwise_df.empty and "judge_model_name" in all_pairwise_df.columns:
        llm_pairwise_df, human_pairwise_df = filter_human_judges_from_df(
            all_pairwise_df, judge_column="judge_model_name"
        )
        if not human_pairwise_df.empty:
            logger.info(
                "Separated %d human judge rows from %d total pairwise rows "
                "(%d LLM-only). Human judges will only appear in agreement "
                "outputs and joint_preference_long_by_judge.",
                len(human_pairwise_df),
                len(all_pairwise_df),
                len(llm_pairwise_df),
            )
        if llm_pairwise_df.empty:
            logger.warning(
                "No LLM judge rows remain after filtering human judges. "
                "Pairwise analysis will be skipped."
            )
    else:
        llm_pairwise_df = all_pairwise_df.copy() if not all_pairwise_df.empty else all_pairwise_df
        human_pairwise_df = pd.DataFrame()

    if parsed_args.only_joint_preference_long:
        if all_pairwise_df is None or all_pairwise_df.empty:
            logger.error(
                "--only-joint-preference-long was set, but no pairwise results were loaded."
            )
            sys.exit(1)
        if "judge_model_name" not in all_pairwise_df.columns:
            logger.error(
                "--only-joint-preference-long requires 'judge_model_name' in pairwise results "
                "to write joint_preference_long_by_judge.csv."
            )
            sys.exit(1)

        try:
            out_paths: Dict[str, str] = {}
            judgment_types = (
                sorted(
                    all_pairwise_df["pairwise_judgment_type"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if "pairwise_judgment_type" in all_pairwise_df.columns
                else [PAIRWISE_JUDGMENT_TYPE_PERSONA]
            )
            for judgment_type in judgment_types:
                joint_tables_dir = (
                    tables_dir / "pairwise_joint" / f"judgment_type_{judgment_type}"
                )
                joint_tables_dir.mkdir(parents=True, exist_ok=True)
                llm_joint_df = llm_pairwise_df[
                    llm_pairwise_df["pairwise_judgment_type"] == judgment_type
                ].copy()
                full_joint_df = all_pairwise_df[
                    all_pairwise_df["pairwise_judgment_type"] == judgment_type
                ].copy()
                if llm_joint_df.empty:
                    logger.warning(
                        "No LLM judge rows for judgment_type=%s in "
                        "--only-joint-preference-long mode; skipping.",
                        judgment_type,
                    )
                    continue
                judgment_out_paths = _write_joint_preference_long_only_outputs(
                    pairwise_df=llm_joint_df,
                    tables_dir=joint_tables_dir,
                    logger=logger,
                    alpha=float(parsed_args.joint_preference_alpha),
                    paired_n_permutations=int(
                        parsed_args.joint_preference_paired_n_permutations
                    ),
                    paired_n_bootstrap=int(
                        parsed_args.joint_preference_paired_n_bootstrap
                    ),
                    paired_alpha=float(parsed_args.joint_preference_paired_alpha),
                    paired_seed=int(parsed_args.joint_preference_paired_seed),
                    equivalence_win_margin=float(
                        parsed_args.joint_preference_equivalence_win_margin
                    ),
                    equivalence_score_margin=float(
                        parsed_args.joint_preference_equivalence_score_margin
                    ),
                    paired_apply_bh_fdr=bool(
                        not parsed_args.joint_preference_paired_disable_bh_fdr
                    ),
                    judge_agreement_disable=bool(
                        parsed_args.joint_preference_judge_agreement_disable
                    ),
                    judge_agreement_n_bootstrap=int(
                        parsed_args.joint_preference_judge_agreement_n_bootstrap
                    ),
                    judge_agreement_seed=int(
                        parsed_args.joint_preference_judge_agreement_seed
                    ),
                    objective_df=(
                        all_objective_df if not all_objective_df.empty else None
                    ),
                    objective_flip_baseline_user_id=objective_flip_baseline_user_id,
                    full_pairwise_df=full_joint_df,
                )
                for key, value in judgment_out_paths.items():
                    out_paths[f"{judgment_type}/{key}"] = value
        except Exception as exc:
            logger.exception(
                "Minimal joint preference long export failed unexpectedly: %s", exc
            )
            sys.exit(1)

        print("Stage 6 (minimal) joint preference export complete.")
        print(f"Tables directory: {tables_dir}")
        for k in sorted(out_paths.keys()):
            print(f"- {k}: {out_paths[k]}")
        return

    if parsed_args.pairwise_dimension_weighted_winner:
        if all_pairwise_df.empty or "user_id" not in all_pairwise_df.columns:
            logger.error(
                "--pairwise-dimension-weighted-winner was set, but pairwise data is missing "
                "or lacks a user_id column."
            )
            sys.exit(1)
        if dimension_weights_by_user is None:
            logger.error(
                "Internal error: dimension_weights_by_user was not loaded for "
                "--pairwise-dimension-weighted-winner."
            )
            sys.exit(1)
        missing_users = sorted(
            {
                str(u)
                for u in all_pairwise_df["user_id"].dropna().unique().tolist()
                if str(u) not in dimension_weights_by_user
            }
        )
        if missing_users:
            logger.error(
                "Missing persona dimension weights for %d persona(s): %s. "
                "Provide configs that cover all personas present in pairwise results.",
                len(missing_users),
                ", ".join(missing_users),
            )
            sys.exit(1)

    # Run Aggregation
    try:
        has_main_inputs = not all_objective_df.empty or not all_subjective_df.empty
        pairwise_only_mode = (
            parsed_args.skip_subjective_figures and not all_pairwise_df.empty
        )

        if has_main_inputs:
            bundle = run_full_aggregation(
                all_objective_df, all_subjective_df, profiles_df
            )
        elif pairwise_only_mode:
            logger.info(
                "No objective or subjective results found; proceeding with pairwise-only analysis "
                "because --skip-subjective-figures was set."
            )
            bundle = AggregationBundle(
                sample_level=pd.DataFrame(),
                user_model_variant=pd.DataFrame(),
                user_model_deltas=pd.DataFrame(),
                persona_summary=pd.DataFrame(),
                global_summary=pd.DataFrame(),
                ranking_reversals=pd.DataFrame(),
            )
        else:
            logger.error(
                "Aggregation failed: objective and subjective results are missing. "
                "Provide input results or enable --skip-subjective-figures with pairwise data."
            )
            sys.exit(1)
    except AnalysisDataError as err:
        logger.error("Aggregation failed: %s", err)
        sys.exit(1)

    # Export Artifacts
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Use user_id as primary identifier (user_profile_type kept for reference only)
    variant_with_persona = bundle.user_model_variant.copy()
    delta_with_persona = bundle.user_model_deltas.copy()

    table_paths = _write_core_tables(
        tables_dir=tables_dir,
        bundle=bundle,
        profiles_df=profiles_df,
        variant_with_persona=variant_with_persona,
        delta_with_persona=delta_with_persona,
    )
    if verification_failures:
        verification_failures_path = tables_dir / "verification_failures_summary.csv"
        pd.DataFrame(verification_failures).to_csv(
            verification_failures_path, index=False
        )
        table_paths["verification_failures_summary"] = str(verification_failures_path)

    figure_paths: Dict[str, str] = {}
    if (
        not parsed_args.skip_figures
        and not parsed_args.skip_subjective_figures
        and not parsed_args.only_pairwise_dimension_comparison_with_objective
    ):
        figure_paths = _render_figures(
            figures_dir=figures_dir,
            bundle=bundle,
            variant_with_persona=variant_with_persona,
            delta_with_persona=delta_with_persona,
            persona_df=bundle.persona_summary,
            objective_override=parsed_args.objective_metric,
        )
    elif parsed_args.skip_subjective_figures:
        logger.info(
            "Skipping standard objective/subjective figures due to --skip-subjective-figures"
        )

    # ------------------------------------------------------------------
    # Component-specific (judge-only and rubric-only) subjective analyses
    # ------------------------------------------------------------------
    component_variants = {
        "judge_only": loader.build_component_subjective_frame(
            all_subjective_df, component_key="judge"
        ),
        "rubric_only": loader.build_component_subjective_frame(
            all_subjective_df, component_key="rubric"
        ),
    }

    component_artifacts: Dict[str, Dict[str, Dict[str, str]]] = {}
    for component_label, component_df in component_variants.items():
        if component_df.empty:
            logger.info(
                "Skipping %s analysis: no component scores available.", component_label
            )
            continue

        logger.info("Running %s subjective analysis...", component_label)
        try:
            component_bundle = run_full_aggregation(
                all_objective_df, component_df, profiles_df
            )
        except AnalysisDataError as err:
            logger.warning(
                "%s analysis failed during aggregation: %s", component_label, err
            )
            continue

        component_base_dir = analysis_run_dir / component_label
        component_tables_dir = component_base_dir / "tables"
        component_figures_dir = component_base_dir / "figures"
        component_tables_dir.mkdir(parents=True, exist_ok=True)
        component_figures_dir.mkdir(parents=True, exist_ok=True)

        component_variant = component_bundle.user_model_variant.copy()
        component_delta = component_bundle.user_model_deltas.copy()

        comp_table_paths = _write_core_tables(
            tables_dir=component_tables_dir,
            bundle=component_bundle,
            profiles_df=profiles_df,
            variant_with_persona=component_variant,
            delta_with_persona=component_delta,
        )

        comp_figure_paths: Dict[str, str] = {}
        if not parsed_args.skip_figures and not parsed_args.skip_subjective_figures:
            comp_figure_paths = _render_figures(
                figures_dir=component_figures_dir,
                bundle=component_bundle,
                variant_with_persona=component_variant,
                delta_with_persona=component_delta,
                persona_df=component_bundle.persona_summary,
                objective_override=parsed_args.objective_metric,
            )

        component_artifacts[component_label] = {
            "tables": comp_table_paths,
            "figures": comp_figure_paths,
            "base_dir": str(component_base_dir),
        }

    # Pairwise Analysis (if data available and not skipped)
    # Structure: persona -> judge_model -> variant -> analysis
    # - persona_all/judge_all/pairwise_combined, ...
    # - persona_all/judge_{name}/pairwise_combined, ...
    # - persona_{name}/judge_all/pairwise_combined, ...
    pairwise_bundle: Optional[PairwiseAggregationBundle] = None
    pairwise_table_paths: Dict[str, str] = {}
    pairwise_figure_paths: Dict[str, str] = {}
    pairwise_bundles: Dict[str, PairwiseAggregationBundle] = {}
    personas_analyzed: List[str] = []
    judges_analyzed: List[str] = []
    judgment_types_analyzed: List[str] = []

    should_run_pairwise = (
        not parsed_args.skip_pairwise
        and not llm_pairwise_df.empty
        and (parsed_args.include_pairwise is None or parsed_args.include_pairwise)
    )

    if should_run_pairwise:
        logger.info("Running pairwise comparison analysis...")
        logger.info(
            "Pairwise tie breaker mode: %s",
            parsed_args.pairwise_tie_breaker,
        )

        # Check which variants exist in the data (LLM-only scope)
        available_variants = (
            llm_pairwise_df["variant_label"].unique().tolist()
            if "variant_label" in llm_pairwise_df.columns
            else []
        )

        # Use LLM-only judges for per-pair aggregation
        available_judges = (
            llm_pairwise_df["judge_model_name"].unique().tolist()
            if "judge_model_name" in llm_pairwise_df.columns
            else []
        )
        _human_judges_excluded, _llm_judges_only = split_judges_by_group(
            available_judges
        )
        if _human_judges_excluded:
            logger.warning(
                "Unexpected human judge tokens in llm_pairwise_df after "
                "filtering: %s. These should have been removed earlier.",
                _human_judges_excluded,
            )

        # Check which users/personas exist in the data
        available_users = (
            llm_pairwise_df["user_id"].unique().tolist()
            if "user_id" in llm_pairwise_df.columns
            else []
        )
        available_judgment_types = (
            sorted(llm_pairwise_df["pairwise_judgment_type"].dropna().unique().tolist())
            if "pairwise_judgment_type" in llm_pairwise_df.columns
            else [PAIRWISE_JUDGMENT_TYPE_PERSONA]
        )

        # Define persona/user filters: None = all personas combined, then each specific
        user_filters: List[Tuple[Optional[str], str]] = [(None, "all")]
        if len(available_users) > 1:
            # Only create per-persona splits if there are multiple personas
            for user_id in available_users:
                # Create a safe directory name from user_id
                safe_user_name = (
                    user_id.replace("/", "-").replace(":", "-").replace(" ", "_")
                )
                user_filters.append((user_id, safe_user_name))

        # Define judge filters: None = all LLM judges combined, then each specific judge.
        # Human judges are excluded -- they only appear in agreement outputs.
        judge_filters: List[Tuple[Optional[str], str]] = [(None, "all")]
        if len(available_judges) > 1:
            for judge in available_judges:
                safe_judge_name = (
                    judge.replace("/", "-").replace(":", "-").replace(" ", "_")
                )
                judge_filters.append((judge, safe_judge_name))

        judgment_type_filters: List[Tuple[str, str]] = []
        for judgment_type in available_judgment_types:
            safe_judgment_name = (
                str(judgment_type).replace("/", "-").replace(":", "-").replace(" ", "_")
            )
            judgment_type_filters.append((judgment_type, safe_judgment_name))

        # Define variant filters: None = combined, then specific variants
        variant_filters: List[Tuple[Optional[str], str]] = [
            (None, "combined"),  # Aggregate across all variants
            ("original", "original"),
            ("personalized", "personalized"),
            ("control", "control"),
        ]

        # Extract unique model pairs from the LLM-only data
        model_pairs: List[Tuple[str, str, str]] = []
        if (
            "model_a_name" in llm_pairwise_df.columns
            and "model_b_name" in llm_pairwise_df.columns
        ):
            unique_pairs = (
                llm_pairwise_df[["model_a_name", "model_b_name", "model_pair"]]
                .drop_duplicates()
                .values.tolist()
            )
            for model_a, model_b, model_pair in unique_pairs:
                # Create safe directory names
                safe_model_a = (
                    str(model_a).replace("/", "-").replace(":", "-").replace(" ", "_")
                )
                safe_model_b = (
                    str(model_b).replace("/", "-").replace(":", "-").replace(" ", "_")
                )
                pair_dir_name = f"{safe_model_a}_vs_{safe_model_b}"
                model_pairs.append((model_a, model_b, pair_dir_name))

        if not model_pairs:
            logger.warning("No model pairs found in pairwise data")
            model_pairs = [("unknown_a", "unknown_b", "unknown_a_vs_unknown_b")]

        # Track unique judges analyzed (across all personas)
        judges_analyzed_set: set = set()

        # cached_objective_lookup was built earlier (before pairwise winner
        # recomputation) and is reused across all aggregation calls below.

        # Pre-filter LLM-only pairwise data by model pair
        pairwise_by_pair: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        for model_a, model_b, pair_dir_name in model_pairs:
            pair_df = llm_pairwise_df.query(
                "model_a_name == @model_a and model_b_name == @model_b"
            )
            if not pair_df.empty:
                pairwise_by_pair[(model_a, model_b, pair_dir_name)] = pair_df.copy()

        # Restructure: judgment_type -> model_pair -> persona -> judge -> variant
        for judgment_filter, judgment_name in judgment_type_filters:
            logger.info("Processing pairwise judgment type: %s", judgment_name)
            if judgment_name not in judgment_types_analyzed:
                judgment_types_analyzed.append(judgment_name)

            judgment_df = llm_pairwise_df[
                llm_pairwise_df["pairwise_judgment_type"] == judgment_filter
            ].copy()
            if judgment_df.empty:
                logger.warning(
                    "No pairwise rows available for judgment_type=%s", judgment_name
                )
                continue

            for model_a, model_b, pair_dir_name in model_pairs:
                logger.info("Processing model pair: %s vs %s", model_a, model_b)

                pair_tables_dir = (
                    tables_dir / f"judgment_type_{judgment_name}" / pair_dir_name
                )
                pair_figures_dir = (
                    figures_dir / f"judgment_type_{judgment_name}" / pair_dir_name
                )

                for user_filter, user_name in user_filters:
                    logger.info("  Processing persona: %s", user_name)
                    if user_name not in personas_analyzed:
                        personas_analyzed.append(user_name)

                    persona_tables_dir = pair_tables_dir / f"persona_{user_name}"
                    persona_figures_dir = pair_figures_dir / f"persona_{user_name}"

                    for judge_filter, judge_name in judge_filters:
                        logger.info("    Processing judge: %s", judge_name)
                        judges_analyzed_set.add(judge_name)

                        judge_tables_dir = persona_tables_dir / f"judge_{judge_name}"
                        judge_figures_dir = persona_figures_dir / f"judge_{judge_name}"

                        for variant_filter, variant_name in variant_filters:
                            # Skip specific variants if they don't exist in data
                            if (
                                variant_filter is not None
                                and variant_filter not in available_variants
                            ):
                                logger.debug(
                                    "Skipping variant '%s' for pair '%s', persona '%s', judge '%s'",
                                    variant_name,
                                    pair_dir_name,
                                    user_name,
                                    judge_name,
                                )
                                continue

                            try:
                                logger.info(
                                    "      Processing variant: %s", variant_name
                                )

                                pair_df = pairwise_by_pair.get(
                                    (model_a, model_b, pair_dir_name)
                                )

                                if pair_df is None or pair_df.empty:
                                    logger.debug(
                                        "No data for pair '%s', persona '%s', judge '%s', variant '%s'",
                                        pair_dir_name,
                                        user_name,
                                        judge_name,
                                        variant_name,
                                    )
                                    continue

                                filtered_pair_df = pair_df[
                                    pair_df["pairwise_judgment_type"] == judgment_filter
                                ].copy()
                                if filtered_pair_df.empty:
                                    logger.debug(
                                        "No data for pair '%s', persona '%s', judge '%s', "
                                        "variant '%s', judgment_type '%s'",
                                        pair_dir_name,
                                        user_name,
                                        judge_name,
                                        variant_name,
                                        judgment_name,
                                    )
                                    continue

                                bundle = run_pairwise_aggregation(
                                    filtered_pair_df,
                                    profiles_df,
                                    variant_filter=variant_filter,
                                    judge_filter=judge_filter,
                                    user_filter=user_filter,
                                    objective_df=all_objective_df,
                                    objective_lookup=cached_objective_lookup,
                                )
                                bundle_key = (
                                    f"{judgment_name}/{pair_dir_name}/"
                                    f"{user_name}/{judge_name}/{variant_name}"
                                )
                                pairwise_bundles[bundle_key] = bundle

                                if (
                                    user_filter is None
                                    and judge_filter is None
                                    and variant_filter is None
                                    and judgment_filter == available_judgment_types[0]
                                ):
                                    pairwise_bundle = bundle

                                variant_tables_dir = (
                                    judge_tables_dir / f"pairwise_{variant_name}"
                                )
                                variant_tables_dir.mkdir(parents=True, exist_ok=True)

                                variant_table_paths = _write_pairwise_tables(
                                    tables_dir=variant_tables_dir,
                                    pairwise_bundle=bundle,
                                    logger=logger,
                                )
                                path_prefix = (
                                    f"{judgment_name}/{pair_dir_name}/"
                                    f"{user_name}/{judge_name}/{variant_name}"
                                )
                                for key, path in variant_table_paths.items():
                                    pairwise_table_paths[f"{path_prefix}/{key}"] = path
                                table_paths[
                                    "pairwise_"
                                    f"{judgment_name}_{pair_dir_name}_{user_name}_"
                                    f"{judge_name}_{variant_name}_tables"
                                ] = str(variant_tables_dir)

                                if not parsed_args.skip_figures:
                                    variant_figures_dir = (
                                        judge_figures_dir / f"pairwise_{variant_name}"
                                    )
                                    variant_figures_dir.mkdir(
                                        parents=True, exist_ok=True
                                    )

                                    variant_figure_paths = _render_pairwise_figures(
                                        figures_dir=variant_figures_dir,
                                        pairwise_bundle=bundle,
                                        logger=logger,
                                        only_dimension_comparison_with_objective=bool(
                                            parsed_args.only_pairwise_dimension_comparison_with_objective
                                        ),
                                    )
                                    for key, path in variant_figure_paths.items():
                                        pairwise_figure_paths[
                                            f"{path_prefix}/{key}"
                                        ] = path
                                    figure_paths[
                                        "pairwise_"
                                        f"{judgment_name}_{pair_dir_name}_{user_name}_"
                                        f"{judge_name}_{variant_name}_figs"
                                    ] = str(variant_figures_dir)

                            except PairwiseAnalysisError as err:
                                logger.warning(
                                    "Pairwise analysis for pair '%s', persona '%s', judge '%s', "
                                    "variant '%s', judgment_type '%s' skipped: %s",
                                    pair_dir_name,
                                    user_name,
                                    judge_name,
                                    variant_name,
                                    judgment_name,
                                    err,
                                )

        judges_analyzed = sorted(judges_analyzed_set)

        # ------------------------------------------------------------------
        # Joint preference matrices (multi-model) per persona x prompt type
        # ------------------------------------------------------------------
        try:
            for judgment_filter, judgment_name in judgment_type_filters:
                llm_joint_df = llm_pairwise_df[
                    llm_pairwise_df["pairwise_judgment_type"] == judgment_filter
                ].copy()
                full_joint_df = all_pairwise_df[
                    all_pairwise_df["pairwise_judgment_type"] == judgment_filter
                ].copy()
                if llm_joint_df.empty:
                    logger.warning(
                        "No LLM judge rows for judgment_type=%s; "
                        "skipping joint preference outputs.",
                        judgment_name,
                    )
                    continue
                joint_tables_root = (
                    tables_dir / "pairwise_joint" / f"judgment_type_{judgment_name}"
                )
                joint_figures_root = (
                    figures_dir / "pairwise_joint" / f"judgment_type_{judgment_name}"
                )
                joint_table_paths, joint_figure_paths = _write_joint_preference_outputs(
                    pairwise_df=llm_joint_df,
                    tables_dir=joint_tables_root,
                    figures_dir=joint_figures_root,
                    logger=logger,
                    alpha=float(parsed_args.joint_preference_alpha),
                    skip_figures=bool(parsed_args.skip_figures),
                    paired_n_permutations=int(
                        parsed_args.joint_preference_paired_n_permutations
                    ),
                    paired_n_bootstrap=int(
                        parsed_args.joint_preference_paired_n_bootstrap
                    ),
                    paired_alpha=float(parsed_args.joint_preference_paired_alpha),
                    paired_seed=int(parsed_args.joint_preference_paired_seed),
                    equivalence_win_margin=float(
                        parsed_args.joint_preference_equivalence_win_margin
                    ),
                    equivalence_score_margin=float(
                        parsed_args.joint_preference_equivalence_score_margin
                    ),
                    paired_apply_bh_fdr=bool(
                        not parsed_args.joint_preference_paired_disable_bh_fdr
                    ),
                    judge_agreement_disable=bool(
                        parsed_args.joint_preference_judge_agreement_disable
                    ),
                    judge_agreement_n_bootstrap=int(
                        parsed_args.joint_preference_judge_agreement_n_bootstrap
                    ),
                    judge_agreement_seed=int(
                        parsed_args.joint_preference_judge_agreement_seed
                    ),
                    indicator_df=(
                        all_indicator_df if not all_indicator_df.empty else None
                    ),
                    objective_df=(
                        all_objective_df if not all_objective_df.empty else None
                    ),
                    objective_flip_baseline_user_id=objective_flip_baseline_user_id,
                    full_pairwise_df=full_joint_df,
                )
                for k, v in joint_table_paths.items():
                    table_paths[f"pairwise_joint/{judgment_name}/{k}"] = v
                for k, v in joint_figure_paths.items():
                    figure_paths[f"pairwise_joint/{judgment_name}/{k}"] = v
        except Exception as exc:
            logger.exception("Joint preference matrix export failed: %s", exc)
            sys.exit(1)

    elif not all_pairwise_df.empty and parsed_args.skip_pairwise:
        logger.info("Pairwise results found but skipped due to --skip-pairwise flag.")

    # Build configuration details from loaded data
    config_details = _extract_configuration_details(
        all_objective_df, all_subjective_df, profiles_df, variant_with_persona
    )

    # Add pairwise configuration details
    if pairwise_bundles:
        all_variants = set()
        for bundle_key in pairwise_bundles.keys():
            all_variants.add(bundle_key.rsplit("/", 1)[-1])

        config_details["pairwise"] = {
            "model_pairs": (
                all_pairwise_df["model_pair"].unique().tolist()
                if "model_pair" in all_pairwise_df.columns
                else []
            ),
            "num_comparisons": len(all_pairwise_df),
            "personas_analyzed": personas_analyzed,
            "judges_analyzed": judges_analyzed,
            "judgment_types_analyzed": judgment_types_analyzed,
            "variants_analyzed": sorted(all_variants),
            "correctness_mode": correctness_mode,
            "include_plus_correctness": include_plus_correctness,
        }
        config_details["pairwise_joint_paired_tests"] = {
            "treatment_label": "personalized",
            "control_label": "control",
            "original_label": "original",
            "n_permutations": int(parsed_args.joint_preference_paired_n_permutations),
            "n_bootstrap": int(parsed_args.joint_preference_paired_n_bootstrap),
            "alpha": float(parsed_args.joint_preference_paired_alpha),
            "seed": int(parsed_args.joint_preference_paired_seed),
            "equivalence_win_margin": float(
                parsed_args.joint_preference_equivalence_win_margin
            ),
            "equivalence_score_margin": float(
                parsed_args.joint_preference_equivalence_score_margin
            ),
            "bh_fdr_enabled": bool(
                not parsed_args.joint_preference_paired_disable_bh_fdr
            ),
        }
        config_details["pairwise_joint_judge_agreement"] = {
            "enabled": bool(not parsed_args.joint_preference_judge_agreement_disable),
            "n_bootstrap": int(
                parsed_args.joint_preference_judge_agreement_n_bootstrap
            ),
            "seed": int(parsed_args.joint_preference_judge_agreement_seed),
        }

    # Summary JSON
    summary_payload = {
        "counts": {
            "users": int(profiles_df["user_id"].nunique()),
            "models": (
                int(variant_with_persona["model_name"].nunique())
                if not variant_with_persona.empty
                else 0
            ),
            "variants": (
                int(variant_with_persona["variant_label"].nunique())
                if not variant_with_persona.empty
                else 0
            ),
            "pairwise_comparisons": len(all_pairwise_df),
            "failed_verification_units": int(
                sum(
                    int(r.get("failed_verification_units", 0))
                    for r in verification_failures
                )
            ),
        },
        "configuration": config_details,
        "tables": table_paths,
        "figures": figure_paths,
        "component_analyses": component_artifacts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective_metric": figure_paths.get("objective_metric_used"),
    }
    summary_path = tables_dir / "analysis_summary.json"
    save_json(summary_payload, str(summary_path))
    logger.info("Stage 6 summary written to %s", summary_path)

    print("Stage 6 analysis complete.")
    print(f"Tables directory: {tables_dir}")
    if figure_paths:
        print(f"Figures directory: {figures_dir}")
    if pairwise_bundles:
        total_bundles = len(pairwise_bundles)
        print(
            f"Pairwise analysis: {len(personas_analyzed)} persona(s), "
            f"{len(judges_analyzed)} judge(s), {len(judgment_types_analyzed)} "
            f"judgment type(s), {total_bundles} total analysis sets"
        )


def _write_core_tables(
    tables_dir: Path,
    bundle,
    profiles_df,
    variant_with_persona,
    delta_with_persona,
) -> Dict[str, str]:
    """Write the required CSV artifacts."""
    table_paths: Dict[str, str] = {}
    variant_path = write_user_model_variant_summary(
        variant_with_persona,
        str(tables_dir / "user_model_variant_summary.csv"),
        profiles_df,
    )
    table_paths["user_model_variant_summary"] = str(variant_path)

    delta_path = write_user_model_deltas(
        delta_with_persona, str(tables_dir / "user_model_deltas.csv"), profiles_df
    )
    table_paths["user_model_deltas"] = str(delta_path)

    global_path = write_model_overall_summary(
        bundle.global_summary, str(tables_dir / "model_overall_summary.csv")
    )
    table_paths["model_overall_summary"] = str(global_path)

    sample_path = write_sample_level_flat(
        bundle.sample_level.merge(
            profiles_df[["user_id", "user_profile_type"]], on="user_id", how="left"
        ),
        str(tables_dir / "sample_level_flat.csv"),
    )
    table_paths["sample_level_flat"] = str(sample_path)

    ranking_path = tables_dir / "ranking_reversals.csv"
    if not bundle.ranking_reversals.empty:
        bundle.ranking_reversals.to_csv(ranking_path, index=False)
        table_paths["ranking_reversals"] = str(ranking_path)

    persona_path = tables_dir / "persona_summary.csv"
    if not bundle.persona_summary.empty:
        bundle.persona_summary.to_csv(persona_path, index=False)
        table_paths["persona_summary"] = str(persona_path)

    return table_paths


def _render_figures(
    figures_dir: Path,
    bundle,
    variant_with_persona,
    delta_with_persona,
    persona_df,
    objective_override: Optional[str],
    render_subjective_figures: bool = True,
) -> Dict[str, str]:
    """Render the Stage 6 figure set (PDF + PNG formats)."""
    figure_paths: Dict[str, str] = {}
    objective_metric = objective_override or _select_objective_metric(
        variant_with_persona.columns
    )

    # Get sample-level data for error bar computation
    sample_df = bundle.sample_level if hasattr(bundle, "sample_level") else None

    # Note: figures.py now saves both PDF and PNG; paths returned are PDF (primary)
    # Passing sample_df enables error bars on bar charts

    all_personas_user_id = "_".join(persona_df["user_id"].unique().tolist())

    if objective_metric and not persona_df.empty:
        figure_paths["persona_objective_bars"] = str(
            plot_persona_metric_bars(
                persona_df,
                metric=objective_metric,
                metric_label="Objective Score",
                title="Objective Performance by User",
                output_path=str(
                    figures_dir / f"persona_objective_bars_{all_personas_user_id}.pdf"
                ),
                sample_df=sample_df,
                show_error_bars=True,
            )
        )

    if "subj_overall_mean" in persona_df.columns and not persona_df.empty:
        figure_paths["persona_subjective_bars"] = str(
            plot_persona_metric_bars(
                persona_df,
                metric="subj_overall_mean",
                metric_label="Subjective Score",
                title="Subjective Experience by User",
                output_path=str(
                    figures_dir / f"persona_subjective_bars_{all_personas_user_id}.pdf"
                ),
                sample_df=sample_df,
                show_error_bars=True,
            )
        )

    if "combined_score_mean" in persona_df.columns and not persona_df.empty:
        figure_paths["persona_combined_bars"] = str(
            plot_persona_metric_bars(
                persona_df,
                metric="combined_score_mean",
                metric_label="Combined Score",
                title="Combined Score by User",
                output_path=str(
                    figures_dir / f"persona_combined_bars_{all_personas_user_id}.pdf"
                ),
                sample_df=sample_df,
                show_error_bars=True,
            )
        )

    delta_metrics = {
        col: label
        for col, label in {
            "obj_overall_pass_at_1_mean_delta": "Δ Pass@1",
            "obj_plus_pass_at_1_mean_delta": "Δ Pass@1 (plus)",
            "obj_base_pass_at_1_mean_delta": "Δ Pass@1 (base)",
            "subj_overall_mean_delta": "Δ Subjective",
            "combined_score_mean_delta": "Δ Combined",
        }.items()
        if col in delta_with_persona.columns
    }
    if delta_metrics and not delta_with_persona.empty:
        figure_paths["personalization_deltas"] = str(
            plot_personalization_deltas(
                delta_with_persona,
                metrics=delta_metrics,
                output_path=str(
                    figures_dir / f"personalization_deltas_{all_personas_user_id}.pdf"
                ),
            )
        )

    # Vibe dimension plots: separate figures for original vs personalized vs control
    # Each compares models by color for each dimension
    if sample_df is not None and not sample_df.empty:
        # Plot for Original prompts
        if "original" in sample_df["variant_label"].values:
            figure_paths["vibe_dimensions_original"] = str(
                plot_vibe_dimension_by_variant(
                    sample_df,
                    variant_label="original",
                    output_path=str(
                        figures_dir
                        / f"vibe_dimensions_original_{all_personas_user_id}.pdf"
                    ),
                    show_error_bars=True,
                )
            )

        # Plot for Personalized prompts
        if "personalized" in sample_df["variant_label"].values:
            figure_paths["vibe_dimensions_personalized"] = str(
                plot_vibe_dimension_by_variant(
                    sample_df,
                    variant_label="personalized",
                    output_path=str(
                        figures_dir
                        / f"vibe_dimensions_personalized_{all_personas_user_id}.pdf"
                    ),
                    show_error_bars=True,
                )
            )

        # Plot for Control prompts
        if "control" in sample_df["variant_label"].values:
            figure_paths["vibe_dimensions_control"] = str(
                plot_vibe_dimension_by_variant(
                    sample_df,
                    variant_label="control",
                    output_path=str(
                        figures_dir
                        / f"vibe_dimensions_control_{all_personas_user_id}.pdf"
                    ),
                    show_error_bars=True,
                )
            )
    elif not variant_with_persona.empty:
        # Fallback to legacy combined view if no sample data
        figure_paths["vibe_dimensions"] = str(
            plot_vibe_dimension_bars(
                variant_with_persona,
                output_path=str(
                    figures_dir / f"vibe_dimension_panels_{all_personas_user_id}.pdf"
                ),
            )
        )

    if objective_metric and "subj_overall_mean" in variant_with_persona.columns:
        figure_paths["objective_vs_subjective"] = str(
            plot_objective_vs_subjective_scatter(
                variant_with_persona,
                objective_metric=objective_metric,
                output_path=str(
                    figures_dir / f"objective_vs_subjective_{all_personas_user_id}.pdf"
                ),
            )
        )
        figure_paths["objective_metric_used"] = objective_metric

    return figure_paths


def _select_objective_metric(columns: List[str]) -> Optional[str]:
    """Choose the most informative objective metric available."""
    for candidate in [
        "obj_overall_pass_at_1_mean",
        "obj_plus_pass_at_1_mean",
        "obj_base_pass_at_1_mean",
    ]:
        if candidate in columns:
            return candidate
    for column in columns:
        if column.startswith("obj_") and column.endswith("_mean"):
            return column
    return None


def _extract_configuration_details(
    objective_df: pd.DataFrame,
    subjective_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    variant_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Extract configuration metadata from loaded dataframes.

    Returns a dictionary with details about:
    - generator_models: Models used to generate prompts
    - evaluated_models: Models that were evaluated
    - judge_models: Models used for subjective evaluation
    - filter_models: Filter configurations used
    - dataset_types: Types of datasets included
    - user_ids: List of user identifiers
    """
    config: Dict[str, Any] = {}

    # Generator models
    if not objective_df.empty and "generator_model" in objective_df.columns:
        config["generator_models"] = sorted(
            objective_df["generator_model"].dropna().unique().tolist()
        )
    elif not subjective_df.empty and "generator_model" in subjective_df.columns:
        config["generator_models"] = sorted(
            subjective_df["generator_model"].dropna().unique().tolist()
        )
    else:
        config["generator_models"] = []

    # Evaluated models
    if not variant_df.empty and "model_name" in variant_df.columns:
        config["evaluated_models"] = sorted(
            variant_df["model_name"].dropna().unique().tolist()
        )
    else:
        config["evaluated_models"] = []

    # Judge models (from subjective evaluation)
    if not subjective_df.empty and "judge_model" in subjective_df.columns:
        config["judge_models"] = sorted(
            subjective_df["judge_model"].dropna().unique().tolist()
        )
    else:
        config["judge_models"] = []

    # Filter models
    if not objective_df.empty and "filter_model" in objective_df.columns:
        config["filter_models"] = sorted(
            objective_df["filter_model"].dropna().unique().tolist()
        )
    elif not subjective_df.empty and "filter_model" in subjective_df.columns:
        config["filter_models"] = sorted(
            subjective_df["filter_model"].dropna().unique().tolist()
        )
    else:
        config["filter_models"] = []

    # Dataset types
    if not objective_df.empty and "dataset_type" in objective_df.columns:
        config["dataset_types"] = sorted(
            objective_df["dataset_type"].dropna().unique().tolist()
        )
    else:
        config["dataset_types"] = []

    # User IDs
    if not profiles_df.empty and "user_id" in profiles_df.columns:
        config["user_ids"] = sorted(profiles_df["user_id"].dropna().unique().tolist())
    else:
        config["user_ids"] = []

    return config


def _write_pairwise_tables(
    tables_dir: Path,
    pairwise_bundle: PairwiseAggregationBundle,
    logger: logging.Logger,
) -> Dict[str, str]:
    """
    Write pairwise comparison analysis tables.

    Args:
        tables_dir: Directory to write tables to.
        pairwise_bundle: Aggregated pairwise analysis results.
        logger: Logger for status messages.

    Returns:
        Dict mapping table names to file paths.
    """
    table_paths: Dict[str, str] = {}

    # Sample-level data
    if not pairwise_bundle.sample_level.empty:
        path = write_pairwise_sample_level(
            pairwise_bundle.sample_level,
            str(tables_dir / "pairwise_sample_level.csv"),
        )
        table_paths["pairwise_sample_level"] = str(path)

    # Pair summary
    if not pairwise_bundle.pair_summary.empty:
        path = write_pairwise_pair_summary(
            pairwise_bundle.pair_summary,
            str(tables_dir / "pairwise_pair_summary.csv"),
        )
        table_paths["pairwise_pair_summary"] = str(path)

    # Dimension summary
    if not pairwise_bundle.dimension_summary.empty:
        path = write_pairwise_dimension_summary(
            pairwise_bundle.dimension_summary,
            str(tables_dir / "pairwise_dimension_summary.csv"),
        )
        table_paths["pairwise_dimension_summary"] = str(path)

    # User summary
    if not pairwise_bundle.user_pair_summary.empty:
        path = write_pairwise_user_summary(
            pairwise_bundle.user_pair_summary,
            str(tables_dir / "pairwise_user_summary.csv"),
        )
        table_paths["pairwise_user_summary"] = str(path)

    # Preference matrix
    if not pairwise_bundle.preference_matrix.empty:
        path = write_pairwise_preference_matrix(
            pairwise_bundle.preference_matrix,
            str(tables_dir / "pairwise_preference_matrix.csv"),
        )
        table_paths["pairwise_preference_matrix"] = str(path)

    # Statistical tests
    if not pairwise_bundle.statistical_tests.empty:
        path = write_pairwise_statistical_tests(
            pairwise_bundle.statistical_tests,
            str(tables_dir / "pairwise_statistical_tests.csv"),
        )
        table_paths["pairwise_statistical_tests"] = str(path)

    # Model rankings (computed separately)
    if not pairwise_bundle.pair_summary.empty:
        rankings = compute_model_rankings(pairwise_bundle.pair_summary)
        if not rankings.empty:
            rankings_path = tables_dir / "pairwise_model_rankings.csv"
            rankings.to_csv(rankings_path, index=False)
            table_paths["pairwise_model_rankings"] = str(rankings_path)

    logger.info("Wrote %d pairwise tables", len(table_paths))
    return table_paths


def _assert_pairwise_rows_canonical(
    pairwise_df: pd.DataFrame, *, logger: logging.Logger
) -> None:
    """
    Assert that pairwise rows use canonical (model_a_name, model_b_name) ordering.

    Stage 6 assumes that a given unordered model pair is represented consistently.
    The Stage-6 pairwise loader (`AnalysisInputLoader.load_pairwise_results`) is
    responsible for canonicalizing A/B ordering; this guard catches regressions or
    mixed-source frames early and fails loudly with diagnostics.

    Args:
        pairwise_df: Pairwise DataFrame.
        logger: Logger used for diagnostics.

    Raises:
        SystemExit: If non-canonical rows are detected.
    """
    if pairwise_df is None or pairwise_df.empty:
        return
    required = ["model_a_name", "model_b_name"]
    missing = [c for c in required if c not in pairwise_df.columns]
    if missing:
        logger.error(
            "Pairwise DataFrame missing required columns for canonicality check: %s",
            missing,
        )
        raise SystemExit(1)

    def _key(x: object) -> tuple[str, str]:
        s = str(x)
        return (normalize_token(s), s)

    keys_a = pairwise_df["model_a_name"].map(_key)
    keys_b = pairwise_df["model_b_name"].map(_key)
    bad_mask = keys_a > keys_b
    bad_count = int(bad_mask.sum())
    if bad_count <= 0:
        return

    sample_cols = [
        c
        for c in [
            "user_id",
            "raw_task_id",
            "task_id",
            "variant_label",
            "variant_id",
            "judge_model_name",
            "model_a_name",
            "model_b_name",
            "raw_model_a_name",
            "raw_model_b_name",
            "pairwise_swapped_to_canonical",
        ]
        if c in pairwise_df.columns
    ]
    sample = pairwise_df.loc[bad_mask, sample_cols].head(8)
    logger.error(
        "Detected %d non-canonical pairwise row(s) where model_a_name sorts after model_b_name. "
        "This indicates a loader canonicalization regression or mixed-source data. "
        "Sample rows:\n%s",
        bad_count,
        sample.to_string(index=False),
    )
    raise SystemExit(1)


def _render_pairwise_figures(
    figures_dir: Path,
    pairwise_bundle: PairwiseAggregationBundle,
    logger: logging.Logger,
    only_dimension_comparison_with_objective: bool = False,
) -> Dict[str, str]:
    """
    Render pairwise comparison analysis figures.

    Args:
        figures_dir: Directory to save figures to.
        pairwise_bundle: Aggregated pairwise analysis results.
        logger: Logger for status messages.

    Returns:
        Dict mapping figure names to file paths.
    """
    figure_paths: Dict[str, str] = {}

    try:
        # Get sample-level data for judge model info
        sample_df = (
            pairwise_bundle.sample_level
            if not pairwise_bundle.sample_level.empty
            else None
        )

        # Win rates bar chart
        if not pairwise_bundle.pair_summary.empty:
            if only_dimension_comparison_with_objective:
                logger.info(
                    "Only rendering pairwise dimension comparison with objective (fast mode)."
                )
            else:
                path = plot_pairwise_win_rates(
                    pairwise_bundle.pair_summary,
                    str(figures_dir / "pairwise_win_rates.pdf"),
                    sample_df=sample_df,
                )
                figure_paths["pairwise_win_rates"] = str(path)

        # Fast mode: only render the LIMA-style dimension comparison that includes objective pass@k.
        if only_dimension_comparison_with_objective:
            if (
                pairwise_bundle.dimension_summary.empty
                or "objective_pass_at_1"
                not in pairwise_bundle.dimension_summary["dimension"].values
            ):
                logger.info(
                    "Fast mode: objective pass@k dimensions are not available; skipping."
                )
                return figure_paths

            dimension_labels = list(PAIRWISE_DIMENSION_LABELS.keys())
            dim_with_obj = pairwise_bundle.dimension_summary.query(
                "dimension in @dimension_labels"
            )
            model_pairs = dim_with_obj["model_pair"].unique()
            if len(model_pairs) == 1:
                path = plot_pairwise_dimension_comparison(
                    dim_with_obj,
                    str(
                        figures_dir / "pairwise_dimension_comparison_with_objective.pdf"
                    ),
                    model_pair=model_pairs[0],
                    sample_df=sample_df,
                    style="lima",
                )
                figure_paths["pairwise_dimension_comparison_with_objective"] = str(path)
            else:
                for pair in model_pairs:
                    safe_pair = pair.replace("/", "-")
                    pair_dim = dim_with_obj.query("model_pair == @pair")
                    path = plot_pairwise_dimension_comparison(
                        pair_dim,
                        str(
                            figures_dir
                            / f"pairwise_dimension_comparison_with_objective_{safe_pair}.pdf"
                        ),
                        model_pair=pair,
                        sample_df=sample_df,
                        style="lima",
                    )
                    figure_paths[
                        f"pairwise_dimension_comparison_with_objective_{safe_pair}"
                    ] = str(path)

            logger.info("Rendered %d pairwise figure(s) (fast mode)", len(figure_paths))
            return figure_paths

        # Objective pass@k win rates (if available)
        if not pairwise_bundle.pair_summary.empty:
            model_pairs = (
                pairwise_bundle.pair_summary["model_pair"].unique()
                if "model_pair" in pairwise_bundle.pair_summary.columns
                else []
            )
            if len(model_pairs) == 1:
                try:
                    path = plot_pairwise_objective_passk(
                        pairwise_bundle.pair_summary,
                        str(figures_dir / "pairwise_objective_passk.pdf"),
                        sample_df=sample_df,
                    )
                    figure_paths["pairwise_objective_passk"] = str(path)
                except AnalysisDataError:
                    pass
            else:
                for pair in model_pairs:
                    pair_data = pairwise_bundle.pair_summary.query(
                        "model_pair == @pair"
                    )
                    try:
                        safe_pair = pair.replace("/", "-")
                        path = plot_pairwise_objective_passk(
                            pair_data,
                            str(
                                figures_dir
                                / f"pairwise_objective_passk_{safe_pair}.pdf"
                            ),
                            sample_df=sample_df,
                        )
                        figure_paths[f"pairwise_objective_passk_{safe_pair}"] = str(
                            path
                        )
                    except AnalysisDataError:
                        continue

        # Dimension comparison (for each model pair, or aggregated if only one)
        if not pairwise_bundle.dimension_summary.empty:
            model_pairs = pairwise_bundle.dimension_summary["model_pair"].unique()

            # Get model names for labels
            dim_data = pairwise_bundle.dimension_summary
            model_a_name = (
                dim_data["model_a_name"].iloc[0]
                if "model_a_name" in dim_data.columns
                else None
            )
            model_b_name = (
                dim_data["model_b_name"].iloc[0]
                if "model_b_name" in dim_data.columns
                else None
            )

            if len(model_pairs) == 1:
                # Stacked bar chart
                path = plot_pairwise_dimension_comparison(
                    pairwise_bundle.dimension_summary,
                    str(figures_dir / "pairwise_dimension_comparison.pdf"),
                    model_pair=model_pairs[0],
                    model_a_name=model_a_name,
                    model_b_name=model_b_name,
                    sample_df=sample_df,
                )
                figure_paths["pairwise_dimension_comparison"] = str(path)

                # Dimension heatmap
                path = plot_pairwise_dimension_heatmap(
                    pairwise_bundle.dimension_summary,
                    str(figures_dir / "pairwise_dimension_heatmap.pdf"),
                    sample_df=sample_df,
                    alpha=0.05,
                )
                figure_paths["pairwise_dimension_heatmap"] = str(path)
            else:
                # Multiple pairs - create one per pair
                for pair in model_pairs:
                    safe_pair = pair.replace("/", "-")
                    pair_data = dim_data.query("model_pair == @pair")
                    ma_name = (
                        pair_data["model_a_name"].iloc[0]
                        if "model_a_name" in pair_data.columns
                        else None
                    )
                    mb_name = (
                        pair_data["model_b_name"].iloc[0]
                        if "model_b_name" in pair_data.columns
                        else None
                    )
                    path = plot_pairwise_dimension_comparison(
                        pairwise_bundle.dimension_summary,
                        str(figures_dir / f"pairwise_dimension_{safe_pair}.pdf"),
                        model_pair=pair,
                        model_a_name=ma_name,
                        model_b_name=mb_name,
                        sample_df=sample_df,
                    )
                    figure_paths[f"pairwise_dimension_{safe_pair}"] = str(path)

        # Dimension comparisons including objective pass@k (if present)
        if (
            not pairwise_bundle.dimension_summary.empty
            and "objective_pass_at_1"
            in pairwise_bundle.dimension_summary["dimension"].values
        ):
            dimension_labels = list(PAIRWISE_DIMENSION_LABELS.keys())
            dim_with_obj = pairwise_bundle.dimension_summary.query(
                "dimension in @dimension_labels"
            )
            model_pairs = dim_with_obj["model_pair"].unique()
            if len(model_pairs) == 1:
                path = plot_pairwise_dimension_comparison(
                    dim_with_obj,
                    str(
                        figures_dir / "pairwise_dimension_comparison_with_objective.pdf"
                    ),
                    model_pair=model_pairs[0],
                    sample_df=sample_df,
                    style="lima",
                )
                figure_paths["pairwise_dimension_comparison_with_objective"] = str(path)

                path = plot_pairwise_dimension_heatmap(
                    dim_with_obj,
                    str(figures_dir / "pairwise_dimension_heatmap_with_objective.pdf"),
                    sample_df=sample_df,
                    alpha=0.05,
                )
                figure_paths["pairwise_dimension_heatmap_with_objective"] = str(path)
            else:
                for pair in model_pairs:
                    safe_pair = pair.replace("/", "-")
                    pair_dim = dim_with_obj.query("model_pair == @pair")
                    path = plot_pairwise_dimension_comparison(
                        pair_dim,
                        str(
                            figures_dir
                            / f"pairwise_dimension_comparison_with_objective_{safe_pair}.pdf"
                        ),
                        model_pair=pair,
                        sample_df=sample_df,
                        style="lima",
                    )
                    figure_paths[
                        f"pairwise_dimension_comparison_with_objective_{safe_pair}"
                    ] = str(path)

                    path = plot_pairwise_dimension_heatmap(
                        pair_dim,
                        str(
                            figures_dir
                            / f"pairwise_dimension_heatmap_with_objective_{safe_pair}.pdf"
                        ),
                        sample_df=sample_df,
                        alpha=0.05,
                    )
                    figure_paths[
                        f"pairwise_dimension_heatmap_with_objective_{safe_pair}"
                    ] = str(path)

        # Position bias rates
        if (
            not pairwise_bundle.dimension_summary.empty
            and "position_bias_rate" in pairwise_bundle.dimension_summary.columns
        ):
            path = plot_position_bias_rates(
                pairwise_bundle.dimension_summary,
                str(figures_dir / "pairwise_position_bias.pdf"),
                sample_df=sample_df,
            )
            figure_paths["pairwise_position_bias"] = str(path)

        # User preferences (if multiple users)
        if (
            not pairwise_bundle.user_pair_summary.empty
            and pairwise_bundle.user_pair_summary["user_id"].nunique() > 1
        ):
            path = plot_pairwise_by_user(
                pairwise_bundle.user_pair_summary,
                str(figures_dir / "pairwise_by_user.pdf"),
                sample_df=sample_df,
            )
            figure_paths["pairwise_by_user"] = str(path)

        # Preference matrix heatmap (if multiple model pairs)
        if (
            not pairwise_bundle.preference_matrix.empty
            and len(pairwise_bundle.preference_matrix) > 1
        ):
            path = plot_preference_matrix_heatmap(
                pairwise_bundle.preference_matrix,
                str(figures_dir / "pairwise_preference_matrix.pdf"),
                sample_df=sample_df,
                statistical_tests=pairwise_bundle.statistical_tests,
                alpha=0.05,
            )
            figure_paths["pairwise_preference_matrix"] = str(path)

        # Forest plot with CIs
        if not pairwise_bundle.pair_summary.empty:
            path = plot_pairwise_forest(
                pairwise_bundle.pair_summary,
                str(figures_dir / "pairwise_forest.pdf"),
                statistical_tests=pairwise_bundle.statistical_tests,
                sample_df=sample_df,
            )
            figure_paths["pairwise_forest"] = str(path)

        # Model rankings (if multiple models)
        if not pairwise_bundle.pair_summary.empty:
            rankings = compute_model_rankings(pairwise_bundle.pair_summary)
            if not rankings.empty and len(rankings) > 1:
                path = plot_model_ranking(
                    rankings,
                    str(figures_dir / "pairwise_model_ranking.pdf"),
                    sample_df=sample_df,
                )
                figure_paths["pairwise_model_ranking"] = str(path)

    except Exception as e:
        logger.error("Error rendering pairwise figures: %s", e)

    logger.info("Rendered %d pairwise figures", len(figure_paths))
    return figure_paths


def _write_joint_preference_long_only_outputs(
    pairwise_df: pd.DataFrame,
    tables_dir: Path,
    logger: logging.Logger,
    alpha: float,
    paired_n_permutations: int,
    paired_n_bootstrap: int,
    paired_alpha: float,
    paired_seed: int,
    equivalence_win_margin: float,
    equivalence_score_margin: float,
    paired_apply_bh_fdr: bool,
    judge_agreement_disable: bool,
    judge_agreement_n_bootstrap: int,
    judge_agreement_seed: int,
    objective_df: Optional[pd.DataFrame] = None,
    objective_flip_baseline_user_id: Optional[str] = None,
    full_pairwise_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """
    Compute and export ONLY the long-form joint preference CSV artifacts.

    This helper is the minimal Stage-6 joint preference export path. It writes:
    - joint_preference_long.csv (LLM judges only)
    - joint_preference_long_by_judge.csv (all judges incl. human)
    - joint_preference_overall.tex (LLM judges only)
    - joint_preference_judge_agreement.tex (LLM judges only)
    - joint_preference_human_streamlit_agreement.tex (all judges incl. human)
    - joint_preference_human_streamlit_dimension_agreement.tex (all judges incl. human)

    Args:
        pairwise_df: LLM-only pairwise data for win-rate matrices and paired tests.
        full_pairwise_df: Full pairwise data (human + LLM) for agreement outputs
            and joint_preference_long_by_judge. Falls back to pairwise_df when None.

    It performs only computations that affect those CSV contents (paired tests,
    optional judge agreement merge, formatting/marking columns, and the four LaTeX
    table exports) and does NOT write any additional matrices, figures, or summary
    artifacts.
    """
    if pairwise_df is None or pairwise_df.empty:
        raise ValueError(
            "pairwise_df is empty; cannot export joint preference long tables."
        )
    if "judge_model_name" not in pairwise_df.columns:
        raise ValueError(
            "pairwise_df is missing 'judge_model_name'; cannot export "
            "joint_preference_long_by_judge.csv."
        )
    if full_pairwise_df is None:
        full_pairwise_df = pairwise_df

    tables_dir.mkdir(parents=True, exist_ok=True)

    # Use canonical prompt types ordering; only include types that exist in the data
    prompt_types = [
        pt for pt in JOINT_PROMPT_TYPES if pt in pairwise_df["variant_label"].unique()
    ]
    if not prompt_types:
        raise ValueError(
            "No known prompt types present in pairwise data; cannot export joint preference tables."
        )

    personas = sorted(pairwise_df["user_id"].dropna().unique().tolist())
    if not personas:
        raise ValueError(
            "pairwise_df has no non-null user_id values; cannot compute persona joint preference."
        )

    matrices = compute_joint_preference_matrices(
        pairwise_df,
        personas=personas,
        prompt_types=prompt_types,
        alpha=alpha,
    )
    if not matrices:
        raise ValueError(
            "No joint matrices produced (insufficient model coverage); cannot export joint preference tables."
        )

    # Global long-form CSV (across judges)
    long_frames = [
        bundle.pairwise_long
        for bundle in matrices.values()
        if bundle.pairwise_long is not None and not bundle.pairwise_long.empty
    ]
    long_df = (
        pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    )
    if long_df is None or long_df.empty:
        raise ValueError(
            "Joint preference long table produced 0 rows; cannot export joint preference tables."
        )

    def _ensure_joint_paired_stat_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure paired-test columns exist even when tests are not computable.

        This keeps joint preference CSV schemas stable across runs that may omit
        one or more conditions (e.g., no control prompt type present).
        """
        if frame is None or frame.empty:
            return frame

        expected_cols = [
            # Treatment vs Original
            "paired_treat_vs_orig_n_base_items",
            "paired_treat_vs_orig_n_with_treatment",
            "paired_treat_vs_orig_n_with_original",
            "paired_treat_vs_orig_delta_win_mean",
            "paired_treat_vs_orig_delta_win_ci_lower",
            "paired_treat_vs_orig_delta_win_ci_upper",
            "paired_treat_vs_orig_delta_win_p_value",
            "paired_treat_vs_orig_delta_win_q_value",
            "paired_treat_vs_orig_delta_score_mean",
            "paired_treat_vs_orig_delta_score_ci_lower",
            "paired_treat_vs_orig_delta_score_ci_upper",
            "paired_treat_vs_orig_delta_score_p_value",
            "paired_treat_vs_orig_delta_score_q_value",
            # Control vs Original
            "paired_ctrl_vs_orig_n_base_items",
            "paired_ctrl_vs_orig_n_with_control",
            "paired_ctrl_vs_orig_n_with_original",
            "paired_ctrl_vs_orig_delta_win_mean",
            "paired_ctrl_vs_orig_delta_win_ci_lower",
            "paired_ctrl_vs_orig_delta_win_ci_upper",
            "paired_ctrl_vs_orig_delta_win_p_value",
            "paired_ctrl_vs_orig_delta_win_q_value",
            "paired_ctrl_vs_orig_delta_win_equiv_margin",
            "paired_ctrl_vs_orig_delta_win_equivalent",
            "paired_ctrl_vs_orig_delta_score_mean",
            "paired_ctrl_vs_orig_delta_score_ci_lower",
            "paired_ctrl_vs_orig_delta_score_ci_upper",
            "paired_ctrl_vs_orig_delta_score_p_value",
            "paired_ctrl_vs_orig_delta_score_q_value",
            "paired_ctrl_vs_orig_delta_score_equiv_margin",
            "paired_ctrl_vs_orig_delta_score_equivalent",
            # Treatment vs Control
            "paired_treat_vs_ctrl_n_base_items",
            "paired_treat_vs_ctrl_n_with_treatment",
            "paired_treat_vs_ctrl_n_with_control",
            "paired_treat_vs_ctrl_delta_win_mean",
            "paired_treat_vs_ctrl_delta_win_ci_lower",
            "paired_treat_vs_ctrl_delta_win_ci_upper",
            "paired_treat_vs_ctrl_delta_win_p_value",
            "paired_treat_vs_ctrl_delta_win_q_value",
            "paired_treat_vs_ctrl_delta_score_mean",
            "paired_treat_vs_ctrl_delta_score_ci_lower",
            "paired_treat_vs_ctrl_delta_score_ci_upper",
            "paired_treat_vs_ctrl_delta_score_p_value",
            "paired_treat_vs_ctrl_delta_score_q_value",
        ]

        out = frame.copy()
        for col in expected_cols:
            if col not in out.columns:
                out[col] = pd.NA
        # Formatted marking columns (computed later; default to empty).
        for col in ["paired_vs_original_mark_win", "paired_vs_original_mark_score"]:
            if col not in out.columns:
                out[col] = ""
        return out

    # ------------------------------------------------------------------
    # Cluster-aware paired tests (treatment/control vs original)
    # ------------------------------------------------------------------
    paired_stats = compute_cluster_aware_paired_tests_for_joint_preference(
        pairwise_df,
        treatment_label="personalized",
        control_label="control",
        original_label="original",
        n_permutations=int(paired_n_permutations),
        n_bootstrap=int(paired_n_bootstrap),
        alpha=float(paired_alpha),
        seed=int(paired_seed),
        equivalence_win_margin=float(equivalence_win_margin),
        equivalence_score_margin=float(equivalence_score_margin),
        two_sided=True,
        apply_bh_fdr=bool(paired_apply_bh_fdr),
        group_by_judge=False,
    )
    if paired_stats is not None and not paired_stats.empty and not long_df.empty:
        join_cols = ["persona", "row_model", "col_model"]
        long_df = long_df.merge(paired_stats, on=join_cols, how="left")
        logger.info(
            "Augmented joint preference long table with paired stats (%d rows).",
            len(paired_stats),
        )
    long_df = _ensure_joint_paired_stat_columns(long_df)

    # ------------------------------------------------------------------
    # LLM-judge agreement analysis (overall + per-dimension)
    # ------------------------------------------------------------------
    def _ensure_joint_judge_agreement_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure judge-agreement columns exist in the joint preference long table.

        This keeps schemas stable across runs that have a single judge, or where
        judge agreement is explicitly disabled.
        """
        if frame is None or frame.empty:
            return frame

        dims_present = []
        for dim in PAIRWISE_DIMENSIONS:
            col = f"dim_{dim}_winner_label"
            if col in pairwise_df.columns:
                dims_present.append(dim)

        base_prefixes = ["judge_agreement_overall"] + [
            f"judge_agreement_dim_{d}" for d in dims_present
        ]
        expected_suffixes = [
            "n_items_any",
            "n_judges",
            "n_judge_pairs",
            "pair_overlap_items_mean",
            "pair_overlap_items_min",
            "percent_agreement_mean_pairs",
            "percent_agreement_mean_pairs_ci_lower",
            "percent_agreement_mean_pairs_ci_upper",
            "cohen_kappa_mean_pairs",
            "cohen_kappa_mean_pairs_ci_lower",
            "cohen_kappa_mean_pairs_ci_upper",
            "judge_win_rate_mean",
            "judge_win_rate_std",
            "judge_win_rate_min",
            "judge_win_rate_max",
            "judge_win_rate_judge_count_used",
            "judge_win_rate_n_items_mean",
            "fleiss_kappa_complete",
            "fleiss_kappa_complete_n_items",
            "fleiss_kappa_complete_ci_lower",
            "fleiss_kappa_complete_ci_upper",
        ]

        out = frame.copy()
        if "judge_agreement_n_judges" not in out.columns:
            out["judge_agreement_n_judges"] = pd.NA
        if "judge_agreement_n_items_total" not in out.columns:
            out["judge_agreement_n_items_total"] = pd.NA

        for prefix in base_prefixes:
            for suf in expected_suffixes:
                col = f"{prefix}_{suf}"
                if col not in out.columns:
                    out[col] = pd.NA
        return out

    if not judge_agreement_disable:
        try:
            _validate_joint_preference_agreement_input(
                pairwise_df,
                logger=logger,
            )
            judges = sorted(
                pairwise_df["judge_model_name"].dropna().astype(str).unique()
            )
            if len(judges) >= 2:
                artifacts = compute_judge_agreement_for_joint_preference(
                    pairwise_df,
                    n_bootstrap=int(judge_agreement_n_bootstrap),
                    seed=int(judge_agreement_seed),
                    judge_column="judge_model_name",
                )
                condition_summary = artifacts.get(
                    "agreement_condition_summary", pd.DataFrame()
                )

                if condition_summary is None or condition_summary.empty:
                    raise ValueError(
                        "Judge agreement analysis produced an empty condition summary. "
                        "This typically indicates missing overlap across judges."
                    )

                join_cols = ["persona", "prompt_type", "row_model", "col_model"]
                if condition_summary.duplicated(subset=join_cols).any():
                    raise ValueError(
                        "Judge agreement condition summary contains duplicate keys for "
                        f"{join_cols}. This indicates ambiguous grouping."
                    )

                long_df = long_df.merge(condition_summary, on=join_cols, how="left")
                logger.info(
                    "Augmented joint preference long table with judge agreement columns (%d rows).",
                    len(condition_summary),
                )
            else:
                logger.info(
                    "Judge agreement analysis skipped: need >=2 judges, found %d.",
                    len(judges),
                )
        except JudgeAgreementError as err:
            logger.exception("Judge agreement analysis failed: %s", err)
            raise
        except Exception as err:
            logger.exception("Judge agreement analysis failed unexpectedly: %s", err)
            raise
    else:
        logger.info("Judge agreement analysis disabled via CLI flag.")

    long_df = _ensure_joint_judge_agreement_columns(long_df)

    def _mark_words(
        delta: object, *, p_value: object = None, q_value: object = None
    ) -> str:
        """
        Format a CSV-friendly significance mark for a paired delta.

        Returns one of: '', 'Higher', 'Lower', 'Equal'.

        Decision rule:
        - Use q_value when provided and numeric; otherwise fall back to p_value.
        - Only return a word when the chosen value <= paired_alpha.
        - Direction comes from the sign of delta.
        """

        def _to_float(x: object) -> Optional[float]:
            if x is None:
                return None
            try:
                if pd.isna(x):
                    return None
            except Exception:
                pass
            try:
                return float(x)
            except Exception:
                return None

        d = _to_float(delta)
        q = _to_float(q_value)
        p = _to_float(p_value)
        pv = q if q is not None else p
        if d is None or pv is None:
            return ""
        if float(pv) > float(paired_alpha):
            return ""
        if float(d) > 0:
            return "Higher"
        if float(d) < 0:
            return "Lower"
        return "Equal"

    def _mark_p(delta: object, p_value: object) -> str:
        """p-based mark (ignores q)."""
        return _mark_words(delta, p_value=p_value, q_value=None)

    def _mark_q(delta: object, q_value: object) -> str:
        """q-based mark (ignores p)."""
        return _mark_words(delta, p_value=None, q_value=q_value)

    if long_df is not None and not long_df.empty:

        def _row_mark_win(r: pd.Series) -> str:
            pt = str(r.get("prompt_type") or "")
            if pt == "personalized":
                return _mark_words(
                    r.get("paired_treat_vs_orig_delta_win_mean"),
                    p_value=r.get("paired_treat_vs_orig_delta_win_p_value"),
                    q_value=r.get("paired_treat_vs_orig_delta_win_q_value"),
                )
            if pt == "control":
                return _mark_words(
                    r.get("paired_ctrl_vs_orig_delta_win_mean"),
                    p_value=r.get("paired_ctrl_vs_orig_delta_win_p_value"),
                    q_value=r.get("paired_ctrl_vs_orig_delta_win_q_value"),
                )
            return ""

        def _row_mark_score(r: pd.Series) -> str:
            pt = str(r.get("prompt_type") or "")
            if pt == "personalized":
                return _mark_words(
                    r.get("paired_treat_vs_orig_delta_score_mean"),
                    p_value=r.get("paired_treat_vs_orig_delta_score_p_value"),
                    q_value=r.get("paired_treat_vs_orig_delta_score_q_value"),
                )
            if pt == "control":
                return _mark_words(
                    r.get("paired_ctrl_vs_orig_delta_score_mean"),
                    p_value=r.get("paired_ctrl_vs_orig_delta_score_p_value"),
                    q_value=r.get("paired_ctrl_vs_orig_delta_score_q_value"),
                )
            return ""

        long_df["paired_vs_original_mark_win"] = long_df.apply(_row_mark_win, axis=1)
        long_df["paired_vs_original_mark_score"] = long_df.apply(
            _row_mark_score, axis=1
        )

        def _row_formatted_personalized_vs_baselines(r: pd.Series) -> str:
            """
            Build a verbose formatted win-rate cell for personalized rows only.

            Format:
              '{formatted_win_rate} o(p:<mark>,q:<mark>) c(p:<mark>,q:<mark>)'
            Where marks reflect personalized-vs-original and personalized-vs-control,
            computed separately for raw p-values and BH-FDR q-values.
            """
            pt = str(r.get("prompt_type") or "")
            if pt != "personalized":
                return ""

            base = str(r.get("formatted_win_rate") or "").strip()
            if not base:
                return ""

            o_p = _mark_p(
                r.get("paired_treat_vs_orig_delta_win_mean"),
                r.get("paired_treat_vs_orig_delta_win_p_value"),
            )
            o_q = _mark_q(
                r.get("paired_treat_vs_orig_delta_win_mean"),
                r.get("paired_treat_vs_orig_delta_win_q_value"),
            )
            c_p = _mark_p(
                r.get("paired_treat_vs_ctrl_delta_win_mean"),
                r.get("paired_treat_vs_ctrl_delta_win_p_value"),
            )
            c_q = _mark_q(
                r.get("paired_treat_vs_ctrl_delta_win_mean"),
                r.get("paired_treat_vs_ctrl_delta_win_q_value"),
            )
            return f"{base} o(p:{o_p},q:{o_q}) c(p:{c_p},q:{c_q})"

        long_df["formatted_win_rate_personalized_vs_baselines"] = long_df.apply(
            _row_formatted_personalized_vs_baselines, axis=1
        )

    def _ensure_objective_flip_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure objective flip columns exist for stable CSV schemas."""
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        for col in [
            "objective_flip_rate_vs_original",
            "objective_reverse_flip_rate_vs_original",
        ]:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    def _attach_objective_flip_metrics(frame: pd.DataFrame) -> pd.DataFrame:
        """Attach objective flip metrics keyed by (persona, prompt_type, row_model)."""
        if frame is None or frame.empty:
            return frame
        if objective_df is None or objective_df.empty:
            logger.warning(
                "Objective results are missing; skipping objective flip metrics for joint preference tables."
            )
            return _ensure_objective_flip_columns(frame)

        try:
            flips = compute_objective_accuracy_consistency_metrics(
                objective_df,
                baseline_user_id=objective_flip_baseline_user_id,
            )
        except ObjectiveConsistencyError as exc:
            raise ValueError(
                "Failed to compute objective flip metrics needed for joint preference tables: "
                + str(exc)
            ) from exc

        if flips is None or flips.empty:
            logger.warning(
                "Objective flip metrics produced 0 rows; leaving flip columns empty in joint preference tables."
            )
            return _ensure_objective_flip_columns(frame)

        join_left = frame.rename(columns={"row_model": "model_name"}).copy()
        out = join_left.merge(
            flips[
                [
                    "persona",
                    "prompt_type",
                    "model_name",
                    "objective_flip_rate_vs_original",
                    "objective_reverse_flip_rate_vs_original",
                ]
            ],
            on=["persona", "prompt_type", "model_name"],
            how="left",
        )
        out = out.rename(columns={"model_name": "row_model"})
        return _ensure_objective_flip_columns(out)

    long_df = _attach_objective_flip_metrics(long_df)

    out_paths: Dict[str, str] = {}
    out_paths["joint_preference_long"] = str(
        write_joint_preference_long_table(
            long_df, str(tables_dir / "joint_preference_long.csv")
        )
    )
    if not (tables_dir / "joint_preference_long.csv").exists():
        raise ValueError(
            "Failed to write joint_preference_long.csv (table is empty or write did not occur)."
        )

    # Judge-separated long-form CSV (uses full data incl. human judges).
    judge_long_df = compute_joint_preference_long_by_judge(
        full_pairwise_df,
        personas=personas,
        prompt_types=prompt_types,
        alpha=alpha,
        judge_column="judge_model_name",
    )
    if judge_long_df is None or judge_long_df.empty:
        raise ValueError(
            "joint_preference_long_by_judge produced 0 rows; cannot export joint preference by-judge table."
        )

    paired_stats_by_judge = compute_cluster_aware_paired_tests_for_joint_preference(
        full_pairwise_df,
        treatment_label="personalized",
        control_label="control",
        original_label="original",
        n_permutations=int(paired_n_permutations),
        n_bootstrap=int(paired_n_bootstrap),
        alpha=float(paired_alpha),
        seed=int(paired_seed),
        equivalence_win_margin=float(equivalence_win_margin),
        equivalence_score_margin=float(equivalence_score_margin),
        two_sided=True,
        apply_bh_fdr=bool(paired_apply_bh_fdr),
        group_by_judge=True,
        judge_column="judge_model_name",
    )
    if paired_stats_by_judge is not None and not paired_stats_by_judge.empty:
        join_cols = ["persona", "row_model", "col_model", "judge_model_name"]
        judge_long_df = judge_long_df.merge(
            paired_stats_by_judge, on=join_cols, how="left"
        )
        logger.info(
            "Augmented judge-separated joint preference long table with paired stats (%d rows).",
            len(paired_stats_by_judge),
        )
    judge_long_df = _ensure_joint_paired_stat_columns(judge_long_df)
    judge_long_df = _attach_objective_flip_metrics(judge_long_df)

    if judge_long_df is not None and not judge_long_df.empty:

        def _row_mark_win_j(r: pd.Series) -> str:
            pt = str(r.get("prompt_type") or "")
            if pt == "personalized":
                return _mark_words(
                    r.get("paired_treat_vs_orig_delta_win_mean"),
                    p_value=r.get("paired_treat_vs_orig_delta_win_p_value"),
                    q_value=r.get("paired_treat_vs_orig_delta_win_q_value"),
                )
            if pt == "control":
                return _mark_words(
                    r.get("paired_ctrl_vs_orig_delta_win_mean"),
                    p_value=r.get("paired_ctrl_vs_orig_delta_win_p_value"),
                    q_value=r.get("paired_ctrl_vs_orig_delta_win_q_value"),
                )
            return ""

        def _row_mark_score_j(r: pd.Series) -> str:
            pt = str(r.get("prompt_type") or "")
            if pt == "personalized":
                return _mark_words(
                    r.get("paired_treat_vs_orig_delta_score_mean"),
                    p_value=r.get("paired_treat_vs_orig_delta_score_p_value"),
                    q_value=r.get("paired_treat_vs_orig_delta_score_q_value"),
                )
            if pt == "control":
                return _mark_words(
                    r.get("paired_ctrl_vs_orig_delta_score_mean"),
                    p_value=r.get("paired_ctrl_vs_orig_delta_score_p_value"),
                    q_value=r.get("paired_ctrl_vs_orig_delta_score_q_value"),
                )
            return ""

        judge_long_df["paired_vs_original_mark_win"] = judge_long_df.apply(
            _row_mark_win_j, axis=1
        )
        judge_long_df["paired_vs_original_mark_score"] = judge_long_df.apply(
            _row_mark_score_j, axis=1
        )

        def _row_formatted_personalized_vs_baselines_j(r: pd.Series) -> str:
            pt = str(r.get("prompt_type") or "")
            if pt != "personalized":
                return ""

            base = str(r.get("formatted_win_rate") or "").strip()
            if not base:
                return ""

            o_p = _mark_p(
                r.get("paired_treat_vs_orig_delta_win_mean"),
                r.get("paired_treat_vs_orig_delta_win_p_value"),
            )
            o_q = _mark_q(
                r.get("paired_treat_vs_orig_delta_win_mean"),
                r.get("paired_treat_vs_orig_delta_win_q_value"),
            )
            c_p = _mark_p(
                r.get("paired_treat_vs_ctrl_delta_win_mean"),
                r.get("paired_treat_vs_ctrl_delta_win_p_value"),
            )
            c_q = _mark_q(
                r.get("paired_treat_vs_ctrl_delta_win_mean"),
                r.get("paired_treat_vs_ctrl_delta_win_q_value"),
            )
            return f"{base} o(p:{o_p},q:{o_q}) c(p:{c_p},q:{c_q})"

        judge_long_df["formatted_win_rate_personalized_vs_baselines"] = (
            judge_long_df.apply(_row_formatted_personalized_vs_baselines_j, axis=1)
        )

    out_paths["joint_preference_long_by_judge"] = str(
        write_joint_preference_long_table(
            judge_long_df, str(tables_dir / "joint_preference_long_by_judge.csv")
        )
    )
    if not (tables_dir / "joint_preference_long_by_judge.csv").exists():
        raise ValueError(
            "Failed to write joint_preference_long_by_judge.csv (table is empty or write did not occur)."
        )

    out_paths["overall_latex"] = str(
        write_joint_preference_overall_latex(
            matrices=matrices,
            output_path=str(tables_dir / "joint_preference_overall.tex"),
            personas=personas,
            prompt_types=prompt_types,
        )
    )
    out_paths["judge_agreement_latex"] = str(
        write_joint_preference_judge_agreement_latex(
            pairwise_df=pairwise_df,
            output_path=str(tables_dir / "joint_preference_judge_agreement.tex"),
            n_bootstrap=int(judge_agreement_n_bootstrap),
            seed=int(judge_agreement_seed),
            judge_column="judge_model_name",
            disable_metrics=bool(judge_agreement_disable),
        )
    )
    out_paths["human_streamlit_agreement_latex"] = str(
        write_joint_preference_streamlit_agreement_latex(
            pairwise_df=full_pairwise_df,
            output_path=str(tables_dir / "joint_preference_human_streamlit_agreement.tex"),
            judge_column="judge_model_name",
            disable_metrics=bool(judge_agreement_disable),
        )
    )
    out_paths["human_streamlit_dimension_agreement_latex"] = str(
        write_joint_preference_streamlit_dimension_agreement_latex(
            pairwise_df=full_pairwise_df,
            output_path=str(
                tables_dir / "joint_preference_human_streamlit_dimension_agreement.tex"
            ),
            judge_column="judge_model_name",
            disable_metrics=bool(judge_agreement_disable),
        )
    )

    return out_paths


def _write_joint_preference_outputs(
    pairwise_df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    logger: logging.Logger,
    alpha: float,
    skip_figures: bool,
    paired_n_permutations: int,
    paired_n_bootstrap: int,
    paired_alpha: float,
    paired_seed: int,
    equivalence_win_margin: float,
    equivalence_score_margin: float,
    paired_apply_bh_fdr: bool,
    judge_agreement_disable: bool,
    judge_agreement_n_bootstrap: int,
    judge_agreement_seed: int,
    persona_importance_weights: Optional[Dict[str, float]] = None,
    indicator_df: Optional[pd.DataFrame] = None,
    objective_df: Optional[pd.DataFrame] = None,
    objective_flip_baseline_user_id: Optional[str] = None,
    full_pairwise_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Compute and export joint preference matrices per persona and prompt type.

    Args:
        pairwise_df: LLM-only pairwise data for matrices, paired tests, and figures.
        full_pairwise_df: Full pairwise data (human + LLM) for agreement outputs
            and joint_preference_long_by_judge. Falls back to pairwise_df when None.

    Outputs:
    - Level A: One figure per (persona, prompt type) -- LLM judges only
    - Level B: One persona figure with three prompt-type matrices -- LLM judges only
    - Level C: One overall grid figure for all personas x prompt types -- LLM judges only
    - CSV exports: per-slice matrices + a global long-form table -- LLM judges only
    - joint_preference_long_by_judge.csv -- all judges incl. human
    - LaTeX export: overall (Level C) view -- LLM judges only
    - LaTeX agreement exports -- all judges incl. human
    """
    if pairwise_df is None or pairwise_df.empty:
        raise ValueError(
            "pairwise_df is empty; cannot compute joint preference matrices."
        )
    if full_pairwise_df is None:
        full_pairwise_df = pairwise_df

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Use canonical prompt types ordering; only include types that exist in the data
    prompt_types = [
        pt
        for pt in JOINT_PROMPT_TYPES
        if pt in pairwise_df["variant_label"].unique().tolist()
    ]
    if not prompt_types:
        logger.info(
            "No known prompt types present in pairwise data; skipping joint matrices."
        )
        return {}, {}

    personas = sorted(pairwise_df["user_id"].dropna().unique().tolist())
    if not personas:
        raise ValueError(
            "pairwise_df has no non-null user_id values; cannot compute persona matrices."
        )

    matrices = compute_joint_preference_matrices(
        pairwise_df,
        personas=personas,
        prompt_types=prompt_types,
        alpha=alpha,
    )
    if not matrices:
        logger.info("No joint matrices produced (insufficient model coverage).")
        return {}, {}

    table_paths: Dict[str, str] = {}
    figure_paths: Dict[str, str] = {}

    # Global long-form CSV
    long_frames = [
        bundle.pairwise_long
        for bundle in matrices.values()
        if not bundle.pairwise_long.empty
    ]
    long_df = (
        pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    )

    def _ensure_joint_paired_stat_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure paired-test columns exist even when tests are not computable.

        This keeps joint preference CSV schemas stable across runs that may omit
        one or more conditions (e.g., no control prompt type present).
        """
        if frame is None or frame.empty:
            return frame

        expected_cols = [
            # Treatment vs Original
            "paired_treat_vs_orig_n_base_items",
            "paired_treat_vs_orig_n_with_treatment",
            "paired_treat_vs_orig_n_with_original",
            "paired_treat_vs_orig_delta_win_mean",
            "paired_treat_vs_orig_delta_win_ci_lower",
            "paired_treat_vs_orig_delta_win_ci_upper",
            "paired_treat_vs_orig_delta_win_p_value",
            "paired_treat_vs_orig_delta_win_q_value",
            "paired_treat_vs_orig_delta_score_mean",
            "paired_treat_vs_orig_delta_score_ci_lower",
            "paired_treat_vs_orig_delta_score_ci_upper",
            "paired_treat_vs_orig_delta_score_p_value",
            "paired_treat_vs_orig_delta_score_q_value",
            # Control vs Original
            "paired_ctrl_vs_orig_n_base_items",
            "paired_ctrl_vs_orig_n_with_control",
            "paired_ctrl_vs_orig_n_with_original",
            "paired_ctrl_vs_orig_delta_win_mean",
            "paired_ctrl_vs_orig_delta_win_ci_lower",
            "paired_ctrl_vs_orig_delta_win_ci_upper",
            "paired_ctrl_vs_orig_delta_win_p_value",
            "paired_ctrl_vs_orig_delta_win_q_value",
            "paired_ctrl_vs_orig_delta_win_equiv_margin",
            "paired_ctrl_vs_orig_delta_win_equivalent",
            "paired_ctrl_vs_orig_delta_score_mean",
            "paired_ctrl_vs_orig_delta_score_ci_lower",
            "paired_ctrl_vs_orig_delta_score_ci_upper",
            "paired_ctrl_vs_orig_delta_score_p_value",
            "paired_ctrl_vs_orig_delta_score_q_value",
            "paired_ctrl_vs_orig_delta_score_equiv_margin",
            "paired_ctrl_vs_orig_delta_score_equivalent",
            # Treatment vs Control
            "paired_treat_vs_ctrl_n_base_items",
            "paired_treat_vs_ctrl_n_with_treatment",
            "paired_treat_vs_ctrl_n_with_control",
            "paired_treat_vs_ctrl_delta_win_mean",
            "paired_treat_vs_ctrl_delta_win_ci_lower",
            "paired_treat_vs_ctrl_delta_win_ci_upper",
            "paired_treat_vs_ctrl_delta_win_p_value",
            "paired_treat_vs_ctrl_delta_win_q_value",
            "paired_treat_vs_ctrl_delta_score_mean",
            "paired_treat_vs_ctrl_delta_score_ci_lower",
            "paired_treat_vs_ctrl_delta_score_ci_upper",
            "paired_treat_vs_ctrl_delta_score_p_value",
            "paired_treat_vs_ctrl_delta_score_q_value",
        ]

        out = frame.copy()
        for col in expected_cols:
            if col not in out.columns:
                out[col] = pd.NA
        # Formatted marking columns (computed later; default to empty).
        for col in ["paired_vs_original_mark_win", "paired_vs_original_mark_score"]:
            if col not in out.columns:
                out[col] = ""
        return out

    # ------------------------------------------------------------------
    # Cluster-aware paired tests (treatment/control vs original)
    # ------------------------------------------------------------------
    paired_stats = compute_cluster_aware_paired_tests_for_joint_preference(
        pairwise_df,
        treatment_label="personalized",
        control_label="control",
        original_label="original",
        n_permutations=int(paired_n_permutations),
        n_bootstrap=int(paired_n_bootstrap),
        alpha=float(paired_alpha),
        seed=int(paired_seed),
        equivalence_win_margin=float(equivalence_win_margin),
        equivalence_score_margin=float(equivalence_score_margin),
        two_sided=True,
        apply_bh_fdr=bool(paired_apply_bh_fdr),
        group_by_judge=False,
    )
    if paired_stats is not None and not paired_stats.empty and not long_df.empty:
        join_cols = ["persona", "row_model", "col_model"]
        long_df = long_df.merge(paired_stats, on=join_cols, how="left")
        logger.info(
            "Augmented joint preference long table with paired stats (%d rows).",
            len(paired_stats),
        )
    long_df = _ensure_joint_paired_stat_columns(long_df)

    # ------------------------------------------------------------------
    # LLM-judge agreement analysis (overall + per-dimension)
    # ------------------------------------------------------------------
    def _ensure_joint_judge_agreement_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure judge-agreement columns exist in the joint preference long table.

        This keeps schemas stable across runs that have a single judge, or where
        judge agreement is explicitly disabled.
        """
        if frame is None or frame.empty:
            return frame

        dims_present = []
        for dim in PAIRWISE_DIMENSIONS:
            col = f"dim_{dim}_winner_label"
            if col in pairwise_df.columns:
                dims_present.append(dim)

        base_prefixes = ["judge_agreement_overall"] + [
            f"judge_agreement_dim_{d}" for d in dims_present
        ]
        expected_suffixes = [
            "n_items_any",
            "n_judges",
            "n_judge_pairs",
            "pair_overlap_items_mean",
            "pair_overlap_items_min",
            "percent_agreement_mean_pairs",
            "percent_agreement_mean_pairs_ci_lower",
            "percent_agreement_mean_pairs_ci_upper",
            "cohen_kappa_mean_pairs",
            "cohen_kappa_mean_pairs_ci_lower",
            "cohen_kappa_mean_pairs_ci_upper",
            "judge_win_rate_mean",
            "judge_win_rate_std",
            "judge_win_rate_min",
            "judge_win_rate_max",
            "judge_win_rate_judge_count_used",
            "judge_win_rate_n_items_mean",
            "fleiss_kappa_complete",
            "fleiss_kappa_complete_n_items",
            "fleiss_kappa_complete_ci_lower",
            "fleiss_kappa_complete_ci_upper",
        ]

        out = frame.copy()
        if "judge_agreement_n_judges" not in out.columns:
            out["judge_agreement_n_judges"] = pd.NA
        if "judge_agreement_n_items_total" not in out.columns:
            out["judge_agreement_n_items_total"] = pd.NA

        for prefix in base_prefixes:
            for suf in expected_suffixes:
                col = f"{prefix}_{suf}"
                if col not in out.columns:
                    out[col] = pd.NA
        return out

    if not judge_agreement_disable:
        try:
            judges = sorted(
                pairwise_df["judge_model_name"].dropna().astype(str).unique()
            )
            if len(judges) >= 2:
                agreement_dir = tables_dir / "judge_agreement"
                agreement_dir.mkdir(parents=True, exist_ok=True)

                artifacts = compute_judge_agreement_for_joint_preference(
                    pairwise_df,
                    n_bootstrap=int(judge_agreement_n_bootstrap),
                    seed=int(judge_agreement_seed),
                    judge_column="judge_model_name",
                )
                condition_summary = artifacts.get(
                    "agreement_condition_summary", pd.DataFrame()
                )

                if condition_summary is None or condition_summary.empty:
                    raise ValueError(
                        "Judge agreement analysis produced an empty condition summary. "
                        "This typically indicates missing overlap across judges."
                    )

                join_cols = ["persona", "prompt_type", "row_model", "col_model"]
                if condition_summary.duplicated(subset=join_cols).any():
                    raise ValueError(
                        "Judge agreement condition summary contains duplicate keys for "
                        f"{join_cols}. This indicates ambiguous grouping."
                    )

                # Merge into the main joint preference long table.
                if long_df is not None and not long_df.empty:
                    long_df = long_df.merge(condition_summary, on=join_cols, how="left")
                    logger.info(
                        "Augmented joint preference long table with judge agreement columns (%d rows).",
                        len(condition_summary),
                    )

                # Write detailed agreement artifacts (LLM judges only).
                cond_path = agreement_dir / "agreement_condition_summary.csv"
                condition_summary.to_csv(cond_path, index=False)
                table_paths["judge_agreement/condition_summary"] = str(cond_path)

                pairs_overall = artifacts.get(
                    "agreement_judge_pairs_overall", pd.DataFrame()
                )
                if pairs_overall is not None and not pairs_overall.empty:
                    p = agreement_dir / "pairwise_agreement_overall.csv"
                    pairs_overall.to_csv(p, index=False)
                    table_paths["judge_agreement/pairwise_agreement_overall"] = str(p)

                pairs_dim = artifacts.get(
                    "agreement_judge_pairs_by_dimension", pd.DataFrame()
                )
                if pairs_dim is not None and not pairs_dim.empty:
                    for dim in sorted(
                        pairs_dim["dimension"].dropna().astype(str).unique().tolist()
                    ):
                        dim_df = pairs_dim[
                            pairs_dim["dimension"].astype(str) == str(dim)
                        ].copy()
                        p = agreement_dir / f"pairwise_agreement_dim_{dim}.csv"
                        dim_df.to_csv(p, index=False)
                        table_paths[f"judge_agreement/pairwise_agreement_dim_{dim}"] = (
                            str(p)
                        )

                ranking = artifacts.get("ranking_stability", pd.DataFrame())
                if ranking is not None and not ranking.empty:
                    p = agreement_dir / "ranking_stability.csv"
                    ranking.to_csv(p, index=False)
                    table_paths["judge_agreement/ranking_stability"] = str(p)

                summary = artifacts.get("agreement_summary", pd.DataFrame())
                if summary is not None and not summary.empty:
                    p = agreement_dir / "agreement_summary.csv"
                    summary.to_csv(p, index=False)
                    table_paths["judge_agreement/summary"] = str(p)
            else:
                logger.info(
                    "Judge agreement analysis skipped: need >=2 judges, found %d.",
                    len(judges),
                )
        except JudgeAgreementError as err:
            logger.exception("Judge agreement analysis failed: %s", err)
            raise
        except Exception as err:
            logger.exception("Judge agreement analysis failed unexpectedly: %s", err)
            raise
    else:
        logger.info("Judge agreement analysis disabled via CLI flag.")

    long_df = _ensure_joint_judge_agreement_columns(long_df)

    def _mark_words(
        delta: object, *, p_value: object = None, q_value: object = None
    ) -> str:
        """
        Format a CSV-friendly significance mark for a paired delta.

        Returns one of: '', 'Higher', 'Lower', 'Equal'.

        Decision rule:
        - Use q_value when provided and numeric; otherwise fall back to p_value.
        - Only return a word when the chosen value <= paired_alpha.
        - Direction comes from the sign of delta.
        """

        def _to_float(x: object) -> Optional[float]:
            if x is None:
                return None
            try:
                if pd.isna(x):
                    return None
            except Exception:
                pass
            try:
                return float(x)
            except Exception:
                return None

        d = _to_float(delta)
        q = _to_float(q_value)
        p = _to_float(p_value)
        pv = q if q is not None else p
        if d is None or pv is None:
            return ""
        if float(pv) > float(paired_alpha):
            return ""
        if float(d) > 0:
            return "Higher"
        if float(d) < 0:
            return "Lower"
        return "Equal"

    def _mark_p(delta: object, p_value: object) -> str:
        """p-based mark (ignores q)."""
        return _mark_words(delta, p_value=p_value, q_value=None)

    def _mark_q(delta: object, q_value: object) -> str:
        """q-based mark (ignores p)."""
        return _mark_words(delta, p_value=None, q_value=q_value)

    # Comment: Add formatted mark columns per row, depending on prompt_type.
    if long_df is not None and not long_df.empty:

        def _row_mark_win(r: pd.Series) -> str:
            pt = str(r.get("prompt_type") or "")
            if pt == "personalized":
                return _mark_words(
                    r.get("paired_treat_vs_orig_delta_win_mean"),
                    p_value=r.get("paired_treat_vs_orig_delta_win_p_value"),
                    q_value=r.get("paired_treat_vs_orig_delta_win_q_value"),
                )
            if pt == "control":
                return _mark_words(
                    r.get("paired_ctrl_vs_orig_delta_win_mean"),
                    p_value=r.get("paired_ctrl_vs_orig_delta_win_p_value"),
                    q_value=r.get("paired_ctrl_vs_orig_delta_win_q_value"),
                )
            return ""

        def _row_mark_score(r: pd.Series) -> str:
            pt = str(r.get("prompt_type") or "")
            if pt == "personalized":
                return _mark_words(
                    r.get("paired_treat_vs_orig_delta_score_mean"),
                    p_value=r.get("paired_treat_vs_orig_delta_score_p_value"),
                    q_value=r.get("paired_treat_vs_orig_delta_score_q_value"),
                )
            if pt == "control":
                return _mark_words(
                    r.get("paired_ctrl_vs_orig_delta_score_mean"),
                    p_value=r.get("paired_ctrl_vs_orig_delta_score_p_value"),
                    q_value=r.get("paired_ctrl_vs_orig_delta_score_q_value"),
                )
            return ""

        long_df["paired_vs_original_mark_win"] = long_df.apply(_row_mark_win, axis=1)
        long_df["paired_vs_original_mark_score"] = long_df.apply(
            _row_mark_score, axis=1
        )

        def _row_formatted_personalized_vs_baselines(r: pd.Series) -> str:
            """
            Build a verbose formatted win-rate cell for personalized rows only.

            Format:
              '{formatted_win_rate} o(p:<mark>,q:<mark>) c(p:<mark>,q:<mark>)'
            Where marks reflect personalized-vs-original and personalized-vs-control,
            computed separately for raw p-values and BH-FDR q-values.
            """
            pt = str(r.get("prompt_type") or "")
            if pt != "personalized":
                return ""

            base = str(r.get("formatted_win_rate") or "").strip()
            if not base:
                return ""

            o_p = _mark_p(
                r.get("paired_treat_vs_orig_delta_win_mean"),
                r.get("paired_treat_vs_orig_delta_win_p_value"),
            )
            o_q = _mark_q(
                r.get("paired_treat_vs_orig_delta_win_mean"),
                r.get("paired_treat_vs_orig_delta_win_q_value"),
            )
            c_p = _mark_p(
                r.get("paired_treat_vs_ctrl_delta_win_mean"),
                r.get("paired_treat_vs_ctrl_delta_win_p_value"),
            )
            c_q = _mark_q(
                r.get("paired_treat_vs_ctrl_delta_win_mean"),
                r.get("paired_treat_vs_ctrl_delta_win_q_value"),
            )
            return f"{base} o(p:{o_p},q:{o_q}) c(p:{c_p},q:{c_q})"

        long_df["formatted_win_rate_personalized_vs_baselines"] = long_df.apply(
            _row_formatted_personalized_vs_baselines, axis=1
        )

    def _ensure_objective_flip_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure objective flip columns exist for stable CSV schemas."""
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        for col in [
            "objective_flip_rate_vs_original",
            "objective_reverse_flip_rate_vs_original",
        ]:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    def _attach_objective_flip_metrics(frame: pd.DataFrame) -> pd.DataFrame:
        """Attach objective flip metrics keyed by (persona, prompt_type, row_model)."""
        if frame is None or frame.empty:
            return frame
        if objective_df is None or objective_df.empty:
            logger.warning(
                "Objective results are missing; skipping objective flip metrics for joint preference tables."
            )
            return _ensure_objective_flip_columns(frame)

        try:
            flips = compute_objective_accuracy_consistency_metrics(
                objective_df,
                baseline_user_id=objective_flip_baseline_user_id,
            )
        except ObjectiveConsistencyError as exc:
            raise ValueError(
                "Failed to compute objective flip metrics needed for joint preference tables: "
                + str(exc)
            ) from exc

        if flips is None or flips.empty:
            logger.warning(
                "Objective flip metrics produced 0 rows; leaving flip columns empty in joint preference tables."
            )
            return _ensure_objective_flip_columns(frame)

        join_left = frame.rename(columns={"row_model": "model_name"}).copy()
        out = join_left.merge(
            flips[
                [
                    "persona",
                    "prompt_type",
                    "model_name",
                    "objective_flip_rate_vs_original",
                    "objective_reverse_flip_rate_vs_original",
                ]
            ],
            on=["persona", "prompt_type", "model_name"],
            how="left",
        )
        out = out.rename(columns={"model_name": "row_model"})
        return _ensure_objective_flip_columns(out)

    long_df = _attach_objective_flip_metrics(long_df)

    table_paths["joint_preference_long"] = str(
        write_joint_preference_long_table(
            long_df, str(tables_dir / "joint_preference_long.csv")
        )
    )

    # Rubric detail summary (diagnostics for what drives wins).
    #
    # This is intentionally additive: it does not affect existing joint preference outputs.
    # It requires Stage 5c indicator records that include `rubric_dimension_details`.
    if indicator_df is not None and not indicator_df.empty:
        rubric_summary_df = compute_pairwise_rubric_detail_summary(
            pairwise_df=pairwise_df,
            indicator_df=indicator_df,
        )
        table_paths["joint_preference_rubric_detail_summary"] = str(
            write_joint_preference_rubric_detail_summary(
                rubric_summary_df,
                str(tables_dir / "joint_preference_rubric_detail_summary.csv"),
            )
        )

    # Global long-form CSV (judge-separated, includes human judges)
    #
    # This is intentionally an additive artifact. The legacy `joint_preference_long.csv`
    # remains aggregated across LLM judges so downstream consumers are unchanged.
    if "judge_model_name" in full_pairwise_df.columns:
        judge_long_df = compute_joint_preference_long_by_judge(
            full_pairwise_df,
            personas=personas,
            prompt_types=prompt_types,
            alpha=alpha,
            judge_column="judge_model_name",
        )
        if judge_long_df is not None and not judge_long_df.empty:
            paired_stats_by_judge = (
                compute_cluster_aware_paired_tests_for_joint_preference(
                    full_pairwise_df,
                    treatment_label="personalized",
                    control_label="control",
                    original_label="original",
                    n_permutations=int(paired_n_permutations),
                    n_bootstrap=int(paired_n_bootstrap),
                    alpha=float(paired_alpha),
                    seed=int(paired_seed),
                    equivalence_win_margin=float(equivalence_win_margin),
                    equivalence_score_margin=float(equivalence_score_margin),
                    two_sided=True,
                    apply_bh_fdr=bool(paired_apply_bh_fdr),
                    group_by_judge=True,
                    judge_column="judge_model_name",
                )
            )
            if paired_stats_by_judge is not None and not paired_stats_by_judge.empty:
                join_cols = ["persona", "row_model", "col_model", "judge_model_name"]
                judge_long_df = judge_long_df.merge(
                    paired_stats_by_judge, on=join_cols, how="left"
                )
                logger.info(
                    "Augmented judge-separated joint preference long table with paired stats (%d rows).",
                    len(paired_stats_by_judge),
                )
        judge_long_df = _ensure_joint_paired_stat_columns(judge_long_df)
        judge_long_df = _attach_objective_flip_metrics(judge_long_df)
        if judge_long_df is not None and not judge_long_df.empty:

            def _row_mark_win_j(r: pd.Series) -> str:
                pt = str(r.get("prompt_type") or "")
                if pt == "personalized":
                    return _mark_words(
                        r.get("paired_treat_vs_orig_delta_win_mean"),
                        p_value=r.get("paired_treat_vs_orig_delta_win_p_value"),
                        q_value=r.get("paired_treat_vs_orig_delta_win_q_value"),
                    )
                if pt == "control":
                    return _mark_words(
                        r.get("paired_ctrl_vs_orig_delta_win_mean"),
                        p_value=r.get("paired_ctrl_vs_orig_delta_win_p_value"),
                        q_value=r.get("paired_ctrl_vs_orig_delta_win_q_value"),
                    )
                return ""

            def _row_mark_score_j(r: pd.Series) -> str:
                pt = str(r.get("prompt_type") or "")
                if pt == "personalized":
                    return _mark_words(
                        r.get("paired_treat_vs_orig_delta_score_mean"),
                        p_value=r.get("paired_treat_vs_orig_delta_score_p_value"),
                        q_value=r.get("paired_treat_vs_orig_delta_score_q_value"),
                    )
                if pt == "control":
                    return _mark_words(
                        r.get("paired_ctrl_vs_orig_delta_score_mean"),
                        p_value=r.get("paired_ctrl_vs_orig_delta_score_p_value"),
                        q_value=r.get("paired_ctrl_vs_orig_delta_score_q_value"),
                    )
                return ""

            judge_long_df["paired_vs_original_mark_win"] = judge_long_df.apply(
                _row_mark_win_j, axis=1
            )
            judge_long_df["paired_vs_original_mark_score"] = judge_long_df.apply(
                _row_mark_score_j, axis=1
            )

            def _row_formatted_personalized_vs_baselines_j(r: pd.Series) -> str:
                pt = str(r.get("prompt_type") or "")
                if pt != "personalized":
                    return ""

                base = str(r.get("formatted_win_rate") or "").strip()
                if not base:
                    return ""

                o_p = _mark_p(
                    r.get("paired_treat_vs_orig_delta_win_mean"),
                    r.get("paired_treat_vs_orig_delta_win_p_value"),
                )
                o_q = _mark_q(
                    r.get("paired_treat_vs_orig_delta_win_mean"),
                    r.get("paired_treat_vs_orig_delta_win_q_value"),
                )
                c_p = _mark_p(
                    r.get("paired_treat_vs_ctrl_delta_win_mean"),
                    r.get("paired_treat_vs_ctrl_delta_win_p_value"),
                )
                c_q = _mark_q(
                    r.get("paired_treat_vs_ctrl_delta_win_mean"),
                    r.get("paired_treat_vs_ctrl_delta_win_q_value"),
                )
                return f"{base} o(p:{o_p},q:{o_q}) c(p:{c_p},q:{c_q})"

            judge_long_df["formatted_win_rate_personalized_vs_baselines"] = (
                judge_long_df.apply(_row_formatted_personalized_vs_baselines_j, axis=1)
            )
        if judge_long_df.empty:
            logger.warning(
                "Joint preference by-judge table produced 0 rows. "
                "This may indicate missing judge metadata in pairwise results."
            )
        else:
            table_paths["joint_preference_long_by_judge"] = str(
                write_joint_preference_long_table(
                    judge_long_df,
                    str(tables_dir / "joint_preference_long_by_judge.csv"),
                )
            )
    else:
        logger.warning(
            "Pairwise data is missing 'judge_model_name'; skipping judge-separated "
            "joint preference CSV export."
        )

    # Per-slice matrices + Level A figures
    paired_mark_lookup: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if paired_stats is not None and not paired_stats.empty:
        for _, row in paired_stats.iterrows():
            persona = str(row.get("persona"))
            rm = str(row.get("row_model"))
            cm = str(row.get("col_model"))
            paired_mark_lookup[(persona, rm, cm)] = {
                "personalized": format_paired_delta_mark(
                    row.get("paired_treat_vs_orig_delta_win_mean"),
                    p_value=row.get("paired_treat_vs_orig_delta_win_p_value"),
                    q_value=row.get("paired_treat_vs_orig_delta_win_q_value"),
                ),
                "control": format_paired_delta_mark(
                    row.get("paired_ctrl_vs_orig_delta_win_mean"),
                    p_value=row.get("paired_ctrl_vs_orig_delta_win_p_value"),
                    q_value=row.get("paired_ctrl_vs_orig_delta_win_q_value"),
                ),
            }

    for (persona, prompt_type), bundle in matrices.items():
        safe_persona = (
            str(persona).replace("/", "-").replace(":", "-").replace(" ", "_")
        )
        slice_tables_dir = (
            tables_dir / f"persona_{safe_persona}" / f"prompt_type_{prompt_type}"
        )
        slice_figures_dir = (
            figures_dir / f"persona_{safe_persona}" / f"prompt_type_{prompt_type}"
        )
        slice_tables_dir.mkdir(parents=True, exist_ok=True)
        slice_figures_dir.mkdir(parents=True, exist_ok=True)

        key_prefix = f"persona_{safe_persona}/prompt_type_{prompt_type}"
        table_paths[f"{key_prefix}/win_rate_matrix"] = str(
            write_joint_preference_matrix(
                bundle.win_rate_matrix,
                str(slice_tables_dir / "joint_preference_win_rate_matrix.csv"),
            )
        )
        formatted_matrix = bundle.formatted_matrix.copy()
        if prompt_type in {"personalized", "control"} and paired_mark_lookup:
            for rm in formatted_matrix.index:
                for cm in formatted_matrix.columns:
                    if rm == cm:
                        continue
                    current = formatted_matrix.loc[rm, cm]
                    if not current or str(current).strip() in {"", "--"}:
                        continue
                    mark = paired_mark_lookup.get(
                        (str(persona), str(rm), str(cm)), {}
                    ).get(str(prompt_type), "")
                    if mark:
                        formatted_matrix.loc[rm, cm] = f"{current}{mark}"

        table_paths[f"{key_prefix}/formatted_matrix"] = str(
            write_joint_preference_matrix(
                formatted_matrix,
                str(slice_tables_dir / "joint_preference_formatted_matrix.csv"),
            )
        )

        if not skip_figures:
            fig_path = plot_joint_preference_matrix_heatmap(
                bundle.win_rate_matrix,
                formatted_matrix,
                str(slice_figures_dir / "joint_preference_matrix.pdf"),
                title=f"Joint Preference Matrix — {persona} — {prompt_type.title()}",
                sample_df=pairwise_df,
            )
            figure_paths[f"{key_prefix}/matrix"] = str(fig_path)

    # Level B: persona panels (three prompt types)
    if not skip_figures:
        for persona in personas:
            safe_persona = (
                str(persona).replace("/", "-").replace(":", "-").replace(" ", "_")
            )
            persona_fig_dir = figures_dir / f"persona_{safe_persona}"
            persona_fig_dir.mkdir(parents=True, exist_ok=True)

            panel_inputs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
            for pt in prompt_types:
                bundle = matrices.get((persona, pt))
                if bundle is None:
                    continue
                panel_inputs[pt] = (bundle.win_rate_matrix, bundle.formatted_matrix)

            if not panel_inputs:
                continue

            fig_path = plot_joint_preference_persona_panels(
                panel_inputs,
                prompt_types=prompt_types,
                output_path=str(persona_fig_dir / "joint_preference_panels.pdf"),
                title=f"Joint Preference Matrices — {persona}",
                sample_df=pairwise_df,
            )
            figure_paths[f"persona_{safe_persona}/panels"] = str(fig_path)

        # Level C: overall grid
        grid_inputs: Dict[Tuple[str, str], Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for (persona, pt), bundle in matrices.items():
            grid_inputs[(persona, pt)] = (
                bundle.win_rate_matrix,
                bundle.formatted_matrix,
            )

        fig_path = plot_joint_preference_overall_grid(
            grid_inputs,
            personas=personas,
            prompt_types=prompt_types,
            output_path=str(figures_dir / "joint_preference_overall_grid.pdf"),
            title="Joint Preference Matrices (All Personas × Prompt Types)",
            sample_df=pairwise_df,
        )
        figure_paths["overall_grid"] = str(fig_path)

    # LaTeX (overall view + pooled judge agreement view + Streamlit-style agreement view)
    table_paths["overall_latex"] = str(
        write_joint_preference_overall_latex(
            matrices=matrices,
            output_path=str(tables_dir / "joint_preference_overall.tex"),
            personas=personas,
            prompt_types=prompt_types,
        )
    )
    table_paths["judge_agreement_latex"] = str(
        write_joint_preference_judge_agreement_latex(
            pairwise_df=pairwise_df,
            output_path=str(tables_dir / "joint_preference_judge_agreement.tex"),
            n_bootstrap=int(judge_agreement_n_bootstrap),
            seed=int(judge_agreement_seed),
            judge_column="judge_model_name",
            disable_metrics=bool(judge_agreement_disable),
        )
    )
    table_paths["human_streamlit_agreement_latex"] = str(
        write_joint_preference_streamlit_agreement_latex(
            pairwise_df=full_pairwise_df,
            output_path=str(tables_dir / "joint_preference_human_streamlit_agreement.tex"),
            judge_column="judge_model_name",
            disable_metrics=bool(judge_agreement_disable),
        )
    )
    table_paths["human_streamlit_dimension_agreement_latex"] = str(
        write_joint_preference_streamlit_dimension_agreement_latex(
            pairwise_df=full_pairwise_df,
            output_path=str(
                tables_dir / "joint_preference_human_streamlit_dimension_agreement.tex"
            ),
            judge_column="judge_model_name",
            disable_metrics=bool(judge_agreement_disable),
        )
    )

    logger.info(
        "Joint preference outputs: %d table(s), %d figure(s)",
        len(table_paths),
        len(figure_paths),
    )
    return table_paths, figure_paths


if __name__ == "__main__":
    main()
