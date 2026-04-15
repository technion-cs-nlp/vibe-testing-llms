#!/usr/bin/env python3
"""
Experiment Orchestrator

Run vibe-testing experiments from a configuration file with support for:
- Incremental execution (skips completed tasks)
- Runtime filtering (by persona, model, tags)
- Parallel execution (via --only-* flags or --task-id)

Usage:
    # Run full experiment
    python scripts/run_experiment.py configs/experiments/example_experiment.yaml

    # Dry-run to see what would execute
    python scripts/run_experiment.py configs/experiments/example_experiment.yaml --dry-run

    # Run only specific personas (for parallel execution on different machines)
    python scripts/run_experiment.py experiment.yaml --only-personas researcher_user

    # Run only local models (requires GPU)
    python scripts/run_experiment.py experiment.yaml --model-tags local

    # Run only API models (can run anywhere)
    python scripts/run_experiment.py experiment.yaml --model-tags api

    # Force re-run even if completed
    python scripts/run_experiment.py experiment.yaml --force

    # List all tasks as JSON (for job arrays)
    python scripts/run_experiment.py experiment.yaml --list-tasks
"""

from __future__ import annotations

# IMPORTANT (Unsloth):
# Unsloth warns if it is imported after transformers, but on CPU-only machines
# importing unsloth can raise (it requires a torch accelerator). To remain usable
# on CPU-only runs (e.g., `--status`), we only import unsloth when CUDA is available.
try:
    import torch

    if torch.cuda.is_available():
        try:
            import unsloth  # type: ignore  # noqa: F401
        except Exception:
            # Best-effort only: this import is an optimization hint, not a requirement.
            pass
except Exception:
    pass

import argparse
import json
import logging
import os
import sys
from itertools import combinations, product
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.experiment.config import ConfigError, ExperimentConfig
from scripts.experiment.runner import ExperimentRunner
from scripts.experiment.state import ExperimentState
from src.vibe_testing.pairwise_judgment_types import (
    normalize_pairwise_judgment_type,
)
from src.vibe_testing.pathing import normalize_token
from scripts.experiment.tasks import Task, TaskGenerator

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """
    Parse a CLI integer argument and ensure it is >= 1.

    Args:
        value: Raw string value from argparse.

    Returns:
        Parsed integer value.

    Raises:
        argparse.ArgumentTypeError: When the value is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {value!r}") from e
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected integer >= 1, got: {parsed}")
    return parsed


def _non_negative_int(value: str) -> int:
    """
    Parse a CLI integer argument and ensure it is >= 0.

    Args:
        value: Raw string value from argparse.

    Returns:
        Parsed integer value.

    Raises:
        argparse.ArgumentTypeError: When the value is not a non-negative integer.
    """
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {value!r}") from e
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected integer >= 0, got: {parsed}")
    return parsed


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Build a simple ASCII table string for CLI output.

    Args:
        headers: Column headers.
        rows: Table rows as lists of stringable values.

    Returns:
        A formatted table string.
    """
    if not headers:
        return ""

    col_widths = []
    for idx, header in enumerate(headers):
        max_row_width = max(
            (len(str(row[idx])) for row in rows), default=0  # type: ignore[index]
        )
        col_widths.append(max(len(str(header)), max_row_width))

    border = "+ " + " + ".join("-" * width for width in col_widths) + " +"
    header_line = (
        "| "
        + " | ".join(
            f"{str(header):{col_widths[idx]}}" for idx, header in enumerate(headers)
        )
        + " |"
    )

    lines = [border, header_line, border]
    for row in rows:
        line = (
            "| "
            + " | ".join(
                f"{str(row[idx]):{col_widths[idx]}}"  # type: ignore[index]
                for idx in range(len(headers))
            )
            + " |"
        )
        lines.append(line)
    lines.append(border)
    return "\n".join(lines)


def _prompt_types_for_stage(
    config: ExperimentConfig, stage: str
) -> List[Optional[str]]:
    """
    Normalize the configured prompt types for a given stage.

    Args:
        config: The experiment configuration.
        stage: Stage identifier.

    Returns:
        List of normalized prompt type tokens, or [None] when none are configured.
    """
    prompt_types = config.get_prompt_types(stage)
    if not prompt_types:
        return [None]

    normalized: List[Optional[str]] = []
    seen: Set[Optional[str]] = set()
    for prompt_type in prompt_types:
        try:
            token = normalize_token(prompt_type)
        except ValueError:
            token = prompt_type
        if token not in seen:
            normalized.append(token)
            seen.add(token)

    return normalized if normalized else [None]


def _format_prompt_type(prompt_type: Optional[str]) -> str:
    return prompt_type or "legacy"


def _format_pairwise_judgment_type(judgment_type: Optional[str]) -> str:
    return judgment_type or "persona"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the orchestrator."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run vibe-testing experiments from configuration files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML configuration file",
    )

    # Stage selection
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "dataset",
            "objective",
            "subjective",
            "indicators",
            "pairwise",
            "analyze",
            "all",
        ],
        default=["all"],
        help="Which stages to run (default: all)",
    )

    # Item filtering
    parser.add_argument(
        "--only-personas",
        nargs="+",
        metavar="NAME",
        help="Only run tasks for these specific personas",
    )
    parser.add_argument(
        "--only-models",
        nargs="+",
        metavar="NAME",
        help="Only run tasks for these specific models",
    )
    parser.add_argument(
        "--only-judges",
        nargs="+",
        metavar="NAME",
        help="Only run tasks for these specific judge models",
    )

    # Tag filtering
    parser.add_argument(
        "--model-tags",
        nargs="+",
        metavar="TAG",
        help="Only run models with these tags (e.g., 'local', 'api')",
    )
    parser.add_argument(
        "--persona-tags",
        nargs="+",
        metavar="TAG",
        help="Only run personas with these tags",
    )

    # Task-based execution (for parallel/SLURM)
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all tasks as JSON (one per line) instead of executing",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        metavar="SPEC",
        help="Run specific task(s) by ID. Supports: single (5), range (3-7), list (1,3,5)",
    )

    # Execution control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without actually running",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks even if they appear completed",
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="Force recreation of datasets even if they exist",
    )
    parser.add_argument(
        "--force-objective",
        action="store_true",
        help="Force recreation of objective evaluations even if they exist",
    )
    parser.add_argument(
        "--force-subjective",
        action="store_true",
        help="Force recreation of subjective evaluations even if they exist",
    )
    parser.add_argument(
        "--force-indicators",
        action="store_true",
        help="Force recreation of indicator score artifacts even if they exist",
    )
    parser.add_argument(
        "--force-pairwise",
        action="store_true",
        help="Force recreation of pairwise comparisons even if they exist",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--indicator-include-rubric",
        action="store_true",
        help=(
            "When running the 'indicators' stage, include rubric-only per-dimension scores "
            "in Stage 5c outputs (enables downstream indicator-only pairwise analysis)."
        ),
    )

    # Analysis (Stage 6) options
    # These mirror scripts/stage_6_analyze_results.py so callers can run analysis via the
    # experiment orchestrator (including through submit_slurm_experiment.sh).
    parser.add_argument(
        "--pairwise-source",
        type=str,
        choices=["stage5b", "indicator_scores"],
        default="stage5b",
        help=(
            "When running the 'analyze' stage, choose the source for pairwise win-rate "
            "analysis. 'stage5b' uses judge pairwise artifacts; 'indicator_scores' "
            "synthesizes pairwise outcomes from Stage 5c indicator artifacts."
        ),
    )
    parser.add_argument(
        "--pairwise-indicator-score-field",
        type=str,
        default="rubric_dimension_scores",
        help=(
            "When running the 'analyze' stage with --pairwise-source=indicator_scores, "
            "this is the indicator record field containing per-dimension scores. "
            "Default: rubric_dimension_scores (produced by Stage 5c with --include-rubric)."
        ),
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=str,
        default=None,
        help=(
            "When running the 'analyze' stage, optional override for --analysis-output-dir "
            "passed to scripts/stage_6_analyze_results.py."
        ),
    )
    parser.add_argument(
        "--include-pairwise",
        action="store_true",
        help="When running the 'analyze' stage, pass --include-pairwise to Stage 6.",
    )
    parser.add_argument(
        "--skip-subjective-figures",
        action="store_true",
        help="When running the 'analyze' stage, pass --skip-subjective-figures to Stage 6.",
    )
    parser.add_argument(
        "--only-joint-preference-long",
        action="store_true",
        help=(
            "When running the 'analyze' stage, pass --only-joint-preference-long to Stage 6 "
            "to export the joint preference long-form CSVs plus the joint-preference "
            "LaTeX tables, and skip all other analysis outputs."
        ),
    )
    parser.add_argument(
        "--pairwise-tie-breaker",
        type=str,
        choices=["strict", "finegrained"],
        default="strict",
        help="When running the 'analyze' stage, pass --pairwise-tie-breaker to Stage 6.",
    )
    parser.add_argument(
        "--omit-figure-dimensions",
        nargs="+",
        default=None,
        help="When running the 'analyze' stage, pass --omit-figure-dimensions to Stage 6.",
    )
    parser.add_argument(
        "--no-figure-titles",
        action="store_true",
        help="When running the 'analyze' stage, pass --no-figure-titles to Stage 6.",
    )
    parser.add_argument(
        "--figure-font-scale",
        type=str,
        default=None,
        help="When running the 'analyze' stage, pass --figure-font-scale to Stage 6.",
    )
    parser.add_argument(
        "--figure-label-size",
        type=str,
        default=None,
        help="When running the 'analyze' stage, pass --figure-label-size to Stage 6.",
    )
    parser.add_argument(
        "--figure-tick-size",
        type=str,
        default=None,
        help="When running the 'analyze' stage, pass --figure-tick-size to Stage 6.",
    )
    parser.add_argument(
        "--pairwise-dimension-weighted-winner",
        action="store_true",
        help=(
            "When running the 'analyze' stage, pass --pairwise-dimension-weighted-winner "
            "to Stage 6. This recomputes per-sample pairwise winners using persona-derived "
            "dimension weights (Stage-6-only)."
        ),
    )
    parser.add_argument(
        "--pairwise-dimension-weighted-configs",
        nargs="+",
        default=None,
        help=(
            "When running the 'analyze' stage, pass --pairwise-dimension-weighted-configs "
            "to Stage 6. When omitted, Stage 6 falls back to the same configs used for "
            "persona importance weighting (or Stage 6 defaults if those are omitted)."
        ),
    )
    parser.add_argument(
        "--pairwise-correctness-mode",
        type=str,
        choices=["ignore", "dimension", "gate"],
        default="ignore",
        help=(
            "When running the 'analyze' stage, pass --pairwise-correctness-mode "
            "to Stage 6. Controls how pass@1 correctness affects pairwise winners."
        ),
    )
    parser.add_argument(
        "--pairwise-include-plus-correctness",
        action="store_true",
        help=(
            "When running the 'analyze' stage, pass --pairwise-include-plus-correctness "
            "to Stage 6. Adds plus-test pass@1 as an additional correctness dimension."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for stage console/file logging (default: INFO).",
    )

    # Prompt-type override
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=["original", "personalized", "control"],
        help=(
            "Override defaults.prompt_types from the config. When provided, "
            "only these prompt types will be used for objective/subjective/"
            "pairwise stages. When omitted, the config defaults (if any) "
            "are used; otherwise all prompt types are processed together."
        ),
    )
    parser.add_argument(
        "--pairwise-judgment-types",
        nargs="+",
        choices=["persona", "general_user"],
        help=(
            "Override pairwise judgment types from the experiment config. "
            "When provided, pairwise task generation uses only these values."
        ),
    )

    # Batch-size override
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help=(
            "Optional override for LLM batch size (>= 1). When provided, this "
            "value is forwarded to stage scripts via --batch-size and overrides "
            "any per-model or experiment defaults."
        ),
    )

    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=42,
        help=(
            "Random seed forwarded to pipeline stages and evaluation scripts "
            "(default: 42)."
        ),
    )

    # Objective evaluation (Stage 4) robustness options
    parser.add_argument(
        "--enable-llm-code-extraction",
        action="store_true",
        help=(
            "Enable LLM-assisted code extraction fallback for Stage 4 objective evaluation. "
            "When enabled, execution failures (timeout/error/stdin_error) will trigger a "
            "secondary model call to extract runnable code and retry execution."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-prompt",
        type=str,
        default=None,
        help=(
            "Optional prompt template string used for LLM code extraction fallback "
            "(forwarded to Stage 4)."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-prompt-file",
        type=str,
        default=None,
        help=(
            "Optional path to a text file containing the LLM code extraction prompt template "
            "(forwarded to Stage 4). Takes precedence over --llm-code-extraction-prompt."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-generation-kwargs",
        type=str,
        default=None,
        help=(
            "Optional JSON object (as a string) with generation kwargs used for the "
            "LLM code extraction fallback (forwarded to Stage 4)."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-retry-on-timeout",
        type=_non_negative_int,
        default=None,
        help=(
            "Optional override for the number of extra retries on timeout when executing "
            "LLM-extracted code (forwarded to Stage 4). Use 0 to disable retry."
        ),
    )
    parser.add_argument(
        "--llm-code-extraction-retry-timeout",
        type=_positive_int,
        default=None,
        help=(
            "Optional override for the sandbox timeout (seconds) used when executing "
            "LLM-extracted code (forwarded to Stage 4)."
        ),
    )

    # Status
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show experiment completion status and exit",
    )
    parser.add_argument(
        "--status-no-tree",
        action="store_true",
        help=(
            "Disable the results tree section in --status output. By default, "
            "a tree of completed results is printed."
        ),
    )

    return parser.parse_args()


def _indent_lines(lines: List[str], indent: int) -> List[str]:
    """
    Indent non-empty lines by a fixed number of spaces.

    Args:
        lines: Input lines.
        indent: Number of spaces to indent.

    Returns:
        Indented lines.
    """
    prefix = " " * indent
    return [f"{prefix}{line}" if line else line for line in lines]


def _format_persona_summary(personas: Set[str], max_items: int = 4) -> str:
    """
    Format a compact persona list for tree output.

    Args:
        personas: Set of persona names.
        max_items: Maximum number of persona names to show before truncation.

    Returns:
        Compact persona summary string.
    """
    ordered = sorted(personas)
    if not ordered:
        return "personas: (none)"
    shown = ordered[:max_items]
    remaining = len(ordered) - len(shown)
    if remaining > 0:
        return f"personas: {', '.join(shown)} ... (+{remaining} more)"
    return f"personas: {', '.join(shown)}"


def _render_ascii_tree(root: str, children: List[Tuple[str, List[str]]]) -> List[str]:
    """
    Render a small ASCII tree under a single root label.

    Args:
        root: Root label.
        children: List of (label, subtree_lines) where subtree_lines are already-rendered
            child lines (without any leading connectors).

    Returns:
        Rendered lines including the root.
    """
    lines = [root]
    for idx, (label, subtree) in enumerate(children):
        is_last = idx == len(children) - 1
        branch = "└── " if is_last else "├── "
        lines.append(f"{branch}{label}")
        if subtree:
            pad = "    " if is_last else "│   "
            for subline in subtree:
                lines.append(f"{pad}{subline}")
    return lines


def _build_results_tree_lines(
    expected_personas: Set[str],
    expected_objective_pairs: Set[Tuple[str, str, Optional[str]]],
    expected_pairwise_quads: Set[Tuple[str, str, str, str, Optional[str], str]],
    datasets_completed: Set[str],
    objective_completed: Set[Tuple[str, str, Optional[str]]],
    pairwise_completed: Set[Tuple[str, str, str, str, Optional[str], str]],
) -> List[str]:
    """
    Build a printed "results tree" for the experiment using the desired hierarchy.

    Important: The printed hierarchy is not the filesystem layout. We regroup
    completed tasks so personas appear at the lowest level.

    Desired order:
      Experiment -> Stage -> (Model OR Judge) -> Pair (pairwise only)
      -> Prompt type -> Persona

    Args:
        expected_personas: Personas included in the experiment.
        expected_objective_pairs: Expected objective tuples.
        expected_pairwise_quads: Expected pairwise tuples.
        datasets_completed: Completed dataset personas.
        objective_completed: Completed objective tuples.
        pairwise_completed: Completed pairwise tuples.

    Returns:
        List of printable lines.
    """
    # We only print "valid results" that are also expected for this experiment.
    completed_datasets = sorted(expected_personas & datasets_completed)
    completed_objective = sorted(expected_objective_pairs & objective_completed)
    completed_pairwise = sorted(expected_pairwise_quads & pairwise_completed)

    stage_nodes: List[Tuple[str, List[str]]] = []

    # Datasets: Stage -> Persona
    if completed_datasets:
        ds_children: List[Tuple[str, List[str]]] = []
        for p in completed_datasets:
            ds_children.append((p, []))
        stage_nodes.append(
            (
                f"Datasets ({len(completed_datasets)})",
                _render_ascii_tree("", ds_children)[1:],
            )
        )

    # Objective: Stage -> Model -> Prompt type -> Persona
    if completed_objective:
        # model -> prompt -> personas
        by_model: dict[str, dict[str, Set[str]]] = {}
        for persona, model, prompt_type in completed_objective:
            prompt_label = _format_prompt_type(prompt_type)
            by_model.setdefault(model, {}).setdefault(prompt_label, set()).add(persona)

        # Keep the tree compact: limit number of models printed.
        max_models = 8
        models_sorted = sorted(by_model.keys())
        shown_models = models_sorted[:max_models]
        remaining_models = len(models_sorted) - len(shown_models)

        model_children: List[Tuple[str, List[str]]] = []
        for model in shown_models:
            prompt_children: List[Tuple[str, List[str]]] = []
            for prompt_label in sorted(by_model[model].keys()):
                personas = by_model[model][prompt_label]
                # Put personas on a single leaf line to keep output short.
                prompt_children.append(
                    (
                        f"{prompt_label} ({len(personas)})",
                        [_format_persona_summary(personas)],
                    )
                )
            model_children.append((model, _render_ascii_tree("", prompt_children)[1:]))

        if remaining_models > 0:
            model_children.append((f"... (+{remaining_models} more models)", []))

        stage_nodes.append(
            (
                f"Objective ({len(completed_objective)})",
                _render_ascii_tree("", model_children)[1:],
            )
        )

    # Pairwise: Stage -> Judgment type -> Judge -> Pair -> Prompt type -> Persona
    if completed_pairwise:
        # judgment_type -> judge -> pair -> prompt -> personas
        by_judgment: dict[str, dict[str, dict[str, dict[str, Set[str]]]]] = {}
        for (
            persona,
            model_a,
            model_b,
            judge,
            prompt_type,
            judgment_type,
        ) in completed_pairwise:
            pair_label = f"{model_a}_vs_{model_b}"
            prompt_label = _format_prompt_type(prompt_type)
            judgment_label = _format_pairwise_judgment_type(judgment_type)
            (
                by_judgment.setdefault(judgment_label, {})
                .setdefault(judge, {})
                .setdefault(pair_label, {})
                .setdefault(prompt_label, set())
                .add(persona)
            )
        judgment_children: List[Tuple[str, List[str]]] = []
        for judgment_type in sorted(by_judgment.keys()):
            judge_children: List[Tuple[str, List[str]]] = []
            for judge in sorted(by_judgment[judgment_type].keys()):
                pairs_sorted = sorted(by_judgment[judgment_type][judge].keys())
                max_pairs = 8
                shown_pairs = pairs_sorted[:max_pairs]
                remaining_pairs = len(pairs_sorted) - len(shown_pairs)

                pair_children: List[Tuple[str, List[str]]] = []
                for pair_label in shown_pairs:
                    prompt_children: List[Tuple[str, List[str]]] = []
                    for prompt_label in sorted(
                        by_judgment[judgment_type][judge][pair_label].keys()
                    ):
                        personas = by_judgment[judgment_type][judge][pair_label][
                            prompt_label
                        ]
                        prompt_children.append(
                            (
                                f"{prompt_label} ({len(personas)})",
                                [_format_persona_summary(personas)],
                            )
                        )
                    pair_children.append(
                        (pair_label, _render_ascii_tree("", prompt_children)[1:])
                    )

                if remaining_pairs > 0:
                    pair_children.append((f"... (+{remaining_pairs} more pairs)", []))

                judge_children.append(
                    (judge, _render_ascii_tree("", pair_children)[1:])
                )
            judgment_children.append(
                (judgment_type, _render_ascii_tree("", judge_children)[1:])
            )

        stage_nodes.append(
            (
                f"Pairwise ({len(completed_pairwise)})",
                _render_ascii_tree("", judgment_children)[1:],
            )
        )

    if not stage_nodes:
        return ["(no completed results found)"]

    # Stage blocks are already rendered as trees; we only need to return the
    # top-level lines (stage names + their subtrees) for printing.
    rendered = _render_ascii_tree("", stage_nodes)
    return rendered[1:]


def show_status(
    config: ExperimentConfig,
    state: ExperimentState,
    show_tree: bool = True,
) -> None:
    """Display experiment completion status."""
    # Compute the "expected" task matrix using TaskGenerator, so --status matches
    # the actual orchestration semantics (including objective-stage optimizations).
    task_generator = TaskGenerator(config)
    expected_tasks = task_generator.generate_all_tasks(
        stages=["dataset", "objective", "pairwise"]
    )

    expected_personas = {t.persona for t in expected_tasks if t.stage == "dataset"}
    expected_objective_pairs = {
        (t.persona, t.model, t.prompt_type)
        for t in expected_tasks
        if t.stage == "objective" and t.model
    }
    expected_pairwise_quads = set()
    for t in expected_tasks:
        if t.stage != "pairwise" or not t.model or not t.judge:
            continue
        if "_vs_" not in t.model:
            continue
        model_a, model_b = t.model.split("_vs_", 1)
        model_x, model_y = sorted([model_a, model_b])
        expected_pairwise_quads.add(
            (
                t.persona,
                model_x,
                model_y,
                t.judge,
                t.prompt_type,
                _format_pairwise_judgment_type(t.pairwise_judgment_type),
            )
        )

    # Completed sets are discovered from disk.
    datasets_completed = state.get_existing_datasets()
    completed_pairs = state.get_existing_objective_evals()
    completed_quads = state.get_existing_pairwise_evals()

    print(f"\nExperiment: {config.name}")
    print(f"Base Directory: {config.base_dir}")
    print()

    print("Configuration:")
    print(f"  Personas: {', '.join(config.use_personas)}")
    print(f"  Models: {', '.join(config.use_models)}")
    print(f"  Judges: {', '.join(config.use_judges)}")
    print(
        "  Pairwise judgment types: "
        f"{', '.join(config.get_pairwise_judgment_types())}"
    )
    print(f"  Generator: {config.generator}")
    print()

    datasets_completed_sorted = sorted(datasets_completed)
    missing_datasets = sorted(expected_personas - set(datasets_completed_sorted))

    completed_pairs_set = set(completed_pairs)
    missing_pairs = sorted(
        expected_objective_pairs - completed_pairs_set,
        key=lambda pair: (pair[0], pair[1], pair[2]),
    )

    completed_quads_set = set(completed_quads)
    missing_quads = sorted(
        expected_pairwise_quads - completed_quads_set,
        key=lambda quad: (quad[0], quad[1], quad[2], quad[3], quad[4], quad[5]),
    )

    print("Completion Status (summary):")
    summary_headers = ["Stage", "Completed", "Expected", "Missing"]
    summary_rows = [
        [
            "Datasets",
            len(expected_personas & set(datasets_completed_sorted)),
            len(expected_personas),
            len(missing_datasets),
        ],
        [
            "Objective",
            len(expected_objective_pairs & completed_pairs_set),
            len(expected_objective_pairs),
            len(missing_pairs),
        ],
        [
            "Pairwise",
            len(expected_pairwise_quads & completed_quads_set),
            len(expected_pairwise_quads),
            len(missing_quads),
        ],
    ]
    print(_format_table(summary_headers, summary_rows))
    print()

    if show_tree:
        print("Completed Results Tree (valid results only):")
        tree_lines = _build_results_tree_lines(
            expected_personas=expected_personas,
            expected_objective_pairs=expected_objective_pairs,
            expected_pairwise_quads=expected_pairwise_quads,
            datasets_completed=datasets_completed,
            objective_completed=set(completed_pairs),
            pairwise_completed=set(completed_quads),
        )
        # Display as a compact ASCII tree.
        print(config.name)
        for line in tree_lines:
            print(line)
        print()

    # print("Completion Status (details):")
    # print("- Datasets (missing personas):")
    # if missing_datasets:
    #     ds_headers = ["Persona"]
    #     ds_rows = [[persona] for persona in missing_datasets]
    #     print(_format_table(ds_headers, ds_rows))
    # else:
    #     print("  None missing")
    # print()

    # print("- Objective evaluations (missing persona-model pairs):")
    # if missing_pairs:
    #     obj_headers = ["Persona", "Model", "Prompt Type"]
    #     obj_rows = [
    #         [persona, model, _format_prompt_type(prompt_type)]
    #         for persona, model, prompt_type in missing_pairs
    #     ]
    #     print(_format_table(obj_headers, obj_rows))
    # else:
    #     print("  None missing")
    # print()

    # print("- Subjective evaluations (missing persona-model-judge triples):")
    # if missing_triples:
    #     subj_headers = ["Persona", "Model", "Judge", "Prompt Type"]
    #     subj_rows = [
    #         [
    #             persona,
    #             model,
    #             judge,
    #             _format_prompt_type(prompt_type),
    #         ]
    #         for persona, model, judge, prompt_type in missing_triples
    #     ]
    #     print(_format_table(subj_headers, subj_rows))
    # else:
    #     print("  None missing")
    # print()

    # print("- Pairwise evaluations (missing persona-model_a-model_b-judge quads):")
    # if missing_quads:
    #     pair_headers = [
    #         "Persona",
    #         "Model A",
    #         "Model B",
    #         "Judge",
    #         "Prompt Type",
    #     ]
    #     pair_rows = [
    #         [
    #             persona,
    #             model_a,
    #             model_b,
    #             judge,
    #             _format_prompt_type(prompt_type),
    #         ]
    #         for persona, model_a, model_b, judge, prompt_type in missing_quads
    #     ]
    #     print(_format_table(pair_headers, pair_rows))
    # else:
    #     print("  None missing")
    print()

    print("Per-model outstanding tasks (summary):")
    model_headers = [
        "Model",
        "Missing as generator (personas)",
        "Missing as evaluated (objective pairs)",
        "Missing as judge (pairwise tasks)",
    ]
    model_rows: List[List[str]] = []
    for model in sorted(config.use_models):
        generator_missing = missing_datasets if model == config.generator else []
        missing_eval_objective = [
            (p, m, pt) for p, m, pt in missing_pairs if m == model
        ]
        missing_judge_pairwise = [
            (p, ma, mb, j, pt, jt)
            for p, ma, mb, j, pt, jt in missing_quads
            if j == model
        ]

        model_rows.append(
            [
                model,
                len(generator_missing),
                len(missing_eval_objective),
                len(missing_judge_pairwise),
            ]
        )

    print(_format_table(model_headers, model_rows))
    print()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job_id:
        logger.info(
            "Detected Slurm environment: job_id=%s, array_task_id=%s",
            slurm_job_id,
            slurm_task_id or "none",
        )

    # Load configuration
    try:
        config = ExperimentConfig.load(args.config)
        logger.info("Loaded experiment config: %s", config.name)
    except (ConfigError, FileNotFoundError) as e:
        logger.error("Failed to load config: %s", e)
        return 1

    if args.pairwise_judgment_types:
        config.pairwise_judgment_types = [
            normalize_pairwise_judgment_type(item)
            for item in args.pairwise_judgment_types
        ]
        logger.info(
            "Overriding pairwise judgment types via CLI: %s",
            config.pairwise_judgment_types,
        )

    # Initialize state and task generator
    state = ExperimentState(config)
    generator = TaskGenerator(config)

    # Status mode
    if args.status:
        show_status(config, state, show_tree=not args.status_no_tree)
        return 0

    # Determine which stages to run
    stages: Optional[List[str]] = None
    if "all" not in args.stages:
        stages = args.stages

    # Apply item filters
    personas = args.only_personas
    models = args.only_models
    judges = args.only_judges

    # Apply tag filters to narrow down items
    if args.model_tags and not models:
        models = config.filter_models_by_tags(args.model_tags)
        logger.info("Filtered models by tags %s: %s", args.model_tags, models)

    if args.persona_tags and not personas:
        personas = config.filter_personas_by_tags(args.persona_tags)
        logger.info("Filtered personas by tags %s: %s", args.persona_tags, personas)

    # Generate tasks
    tasks = generator.generate_all_tasks(
        personas=personas,
        models=models,
        judges=judges,
        stages=stages,
        prompt_types_override=args.prompt_types,
        pairwise_judgment_types_override=args.pairwise_judgment_types,
    )

    # Apply additional tag filters on tasks (for cases where both item and tag filters used)
    if args.model_tags:
        tasks = generator.filter_by_tags(tasks, model_tags=args.model_tags)
    if args.persona_tags:
        tasks = generator.filter_by_tags(tasks, persona_tags=args.persona_tags)

    # Apply task ID filter (for parallel execution)
    if args.task_id:
        task_ids = generator.parse_task_id_spec(args.task_id)
        tasks = generator.filter_by_task_ids(tasks, task_ids)
        logger.info("Filtered to task IDs %s: %d tasks", args.task_id, len(tasks))

    # Filter completed tasks (unless --force)
    if not args.force and not args.list_tasks:
        force_stages = []
        if args.force_dataset:
            force_stages.append("dataset")
        if args.force_objective:
            force_stages.append("objective")
        if args.force_subjective:
            force_stages.append("subjective")
        if args.force_indicators:
            force_stages.append("indicators")
        if args.force_pairwise:
            force_stages.append("pairwise")
        tasks = generator.filter_completed(tasks, state, force_stages=force_stages)

    # List mode - output tasks as JSON
    if args.list_tasks:
        for task in tasks:
            print(json.dumps(task.to_dict()))
        return 0

    # Check if there's anything to do
    if not tasks:
        logger.info("No tasks to run. Use --force to re-run completed tasks.")
        return 0

    # Print summary
    print(f"\nExperiment: {config.name}")
    print(f"Tasks to run: {len(tasks)}")
    print()

    if args.dry_run:
        print("DRY-RUN MODE - No tasks will be executed\n")

    # Show task breakdown
    stage_counts = {}
    for task in tasks:
        stage_counts[task.stage] = stage_counts.get(task.stage, 0) + 1

    print("Task breakdown:")
    for stage in ["dataset", "objective", "subjective", "pairwise", "analyze"]:
        if stage in stage_counts:
            print(f"  {stage}: {stage_counts[stage]}")
    print()

    # Execute tasks
    runner = ExperimentRunner(
        config,
        dry_run=args.dry_run,
        verbose=args.verbose,
        log_level=args.log_level,
        seed=args.seed,
        batch_size_override=args.batch_size,
        force_dataset=args.force_dataset,
        force_objective=args.force_objective,
        force_subjective=args.force_subjective,
        force_indicators=args.force_indicators,
        force_pairwise=args.force_pairwise,
        indicator_include_rubric=args.indicator_include_rubric,
        analysis_pairwise_source=args.pairwise_source,
        analysis_pairwise_indicator_score_field=args.pairwise_indicator_score_field,
        analysis_output_dir=args.analysis_output_dir,
        analysis_include_pairwise=args.include_pairwise,
        analysis_skip_subjective_figures=args.skip_subjective_figures,
        analysis_only_joint_preference_long=args.only_joint_preference_long,
        analysis_pairwise_tie_breaker=args.pairwise_tie_breaker,
        analysis_omit_figure_dimensions=args.omit_figure_dimensions,
        analysis_no_figure_titles=args.no_figure_titles,
        analysis_figure_font_scale=args.figure_font_scale,
        analysis_figure_label_size=args.figure_label_size,
        analysis_figure_tick_size=args.figure_tick_size,
        analysis_pairwise_dimension_weighted_winner=args.pairwise_dimension_weighted_winner,
        analysis_pairwise_dimension_weighted_configs=args.pairwise_dimension_weighted_configs,
        analysis_pairwise_correctness_mode=args.pairwise_correctness_mode,
        analysis_pairwise_include_plus_correctness=args.pairwise_include_plus_correctness,
        enable_llm_code_extraction=args.enable_llm_code_extraction,
        llm_code_extraction_prompt=args.llm_code_extraction_prompt,
        llm_code_extraction_prompt_file=args.llm_code_extraction_prompt_file,
        llm_code_extraction_generation_kwargs=args.llm_code_extraction_generation_kwargs,
        llm_code_extraction_retry_on_timeout=args.llm_code_extraction_retry_on_timeout,
        llm_code_extraction_retry_timeout=args.llm_code_extraction_retry_timeout,
    )
    results = runner.run_tasks(tasks)

    # Print final summary
    print()
    print("=" * 50)
    print(f"Completed: {results['completed']}")
    print(f"Failed: {results['failed']}")
    if results["failed"] > 0:
        print("\nSome tasks failed. Check logs for details.")
        return 1

    print("\nExperiment complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
