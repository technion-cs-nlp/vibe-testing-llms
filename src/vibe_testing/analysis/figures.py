"""Visualization helpers for Stage 6 analysis.

This module provides publication-quality figures suitable for ACL and similar venues.
All figures are saved in both PDF (vector) and PNG (raster) formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from src.vibe_testing.analysis.aggregations import AnalysisDataError
from src.vibe_testing.analysis.io import SUBJECTIVE_DIMENSIONS
from src.vibe_testing.model_names import display_model_name

# Module logger (configured by the caller via the project's logging setup).
logger = logging.getLogger(__name__)

# Heatmap diverging colors (low -> mid -> high)
HEATMAP_LOW_COLOR = "#FF1493"  # Hot pink
HEATMAP_HIGH_COLOR = "#00BFFF"  # Deep Sky Blue
HEATMAP_MID_COLOR = "#FFFFFF"


def _heatmap_cmap() -> LinearSegmentedColormap:
    """Create a diverging colormap using the configured heatmap colors."""
    return LinearSegmentedColormap.from_list(
        "vibe_diverging",
        [HEATMAP_LOW_COLOR, HEATMAP_MID_COLOR, HEATMAP_HIGH_COLOR],
    )


def _joint_heatmap_cmap_with_alpha() -> LinearSegmentedColormap:
    """
    Joint-matrix colormap with alpha ramp on the low end.

    Goal: values in 0.00–0.10 should be *more transparent* than values around 0.20,
    while keeping the same hot-pink/white/deep-sky-blue hue mapping.
    """
    pink = np.array(plt.matplotlib.colors.to_rgba(HEATMAP_LOW_COLOR))
    white = np.array(plt.matplotlib.colors.to_rgba(HEATMAP_MID_COLOR))
    blue = np.array(plt.matplotlib.colors.to_rgba(HEATMAP_HIGH_COLOR))

    def _with_alpha(rgba: np.ndarray, a: float) -> tuple[float, float, float, float]:
        out = rgba.copy()
        out[3] = float(np.clip(a, 0.0, 1.0))
        return tuple(out.tolist())

    # Low side alpha ramp:
    # 0.00: very transparent
    # 0.10: still very transparent
    # 0.20: noticeably less transparent
    # 0.40: mostly opaque
    # 0.50+: opaque
    return LinearSegmentedColormap.from_list(
        "vibe_joint_alpha",
        [
            (0.00, _with_alpha(pink, 0.12)),
            (0.10, _with_alpha(pink, 0.20)),
            (0.20, _with_alpha(pink, 0.45)),
            (0.40, _with_alpha(pink, 0.85)),
            (0.50, _with_alpha(white, 1.00)),
            (1.00, _with_alpha(blue, 1.00)),
        ],
    )


# =============================================================================
# CONFIGURABLE CONSTANTS - Easy to modify for different publication styles
# =============================================================================

# Color palettes
VARIANT_PALETTE: Dict[str, str] = {
    "original": "#4C72B0",  # Blue
    "personalized": "#55A868",  # Green
}

# Vibe dimension mappings
# Deduplicate to avoid duplicates from backward compatibility mappings
VIBE_DIMENSION_COLUMNS: List[str] = list(dict.fromkeys(SUBJECTIVE_DIMENSIONS.values()))
DIMENSION_LABELS: Dict[str, str] = {
    "subj_clarity": "Clarity",
    "subj_tone_style_fit": "Tone/Style",
    "subj_workflow_fit": "Workflow Fit",
    "subj_cognitive_load": "Cog. Load",
    "subj_context_awareness": "Context",
    "subj_persona_consistency": "Persona",
    "subj_friction_loss_of_control": "Friction",
    "subj_reliability_user_trust": "Reliability (Trust)",
    "subj_anthropomorphism": "Anthropomorphism",
    # Backward compatibility
    "subj_efficiency": "Workflow Fit",
    "subj_frustration_indicator": "Friction",
}

# =============================================================================
# FIGURE STYLE CONFIGURATION - Tuned for ACL publication standards
# =============================================================================

FIGURE_CONFIG = {
    # Typography - LARGE sizes for publication readability
    "font_family": "serif",
    "font_serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font_scale": 1.8,
    # Font sizes (increased ~50% from previous values)
    "title_size": 20,
    "suptitle_size": 22,
    "label_size": 16,
    "tick_size": 14,
    "legend_size": 14,
    "legend_title_size": 15,
    # Figure dimensions
    "default_width": 14,
    "default_height": 6,
    "catplot_height": 6,
    "catplot_aspect": 1.8,
    "scatter_height": 6,
    "scatter_aspect": 1.4,
    # Output settings
    "dpi": 300,
    "save_pdf": True,
    "save_png": False,
    # Grid and styling
    "grid_alpha": 0.5,
    "grid_linestyle": "--",
    "bar_edgecolor": "black",
    "bar_linewidth": 0.8,
    # Error bars
    "errorbar_capsize": 3,
    "errorbar_linewidth": 1.5,
    # Layout
    "col_wrap": 3,
    "x_label_rotation": 35,
    # Title control
    "show_titles": True,
    # Tight layout / saving
    "tight_layout_pad": 0.05,
    "tight_layout_w_pad": 0.15,
    "tight_layout_h_pad": 0.15,
    "save_pad_inches": 0.02,
    "tight_layout_on_save": True,
    # Joint matrix annotation sizing (None -> derive from tick_size)
    "joint_annot_fontsize": None,
    # Dimension omission (applies across relevant figures)
    "omit_pairwise_dimensions": set(),
    "omit_subjective_dimensions": set(),
}


def apply_figure_config_overrides(overrides: Dict[str, Any]) -> None:
    """
    Apply runtime overrides to FIGURE_CONFIG.

    This is used by Stage 6 scripts to adjust typography/layout without editing code.

    Args:
        overrides: Dict of FIGURE_CONFIG keys to override.
    """
    if not overrides:
        return
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in FIGURE_CONFIG:
            raise KeyError(f"Unknown FIGURE_CONFIG key: {key}")
        FIGURE_CONFIG[key] = value


def apply_figure_dimension_omits(dimensions: List[str]) -> None:
    """
    Configure omission of specific dimensions from figures.

    This affects:
    - Subjective vibe-dimension figures (drops corresponding `subj_*` columns)
    - Pairwise dimension figures (drops corresponding `dimension` values)

    Args:
        dimensions: List of dimension tokens to omit. Supports canonical names and
            common aliases, e.g.:
            - "reliability_user_trust"
            - "friction_loss_of_control"
            - "frustration" (alias -> friction_loss_of_control)
            - "efficiency" (alias -> workflow_fit)
            - "subj_reliability_user_trust" (subjective-only explicit)
    """
    pairwise, subjective = _normalize_omit_dimensions(dimensions)
    FIGURE_CONFIG["omit_pairwise_dimensions"] = pairwise
    FIGURE_CONFIG["omit_subjective_dimensions"] = subjective


def _normalize_omit_dimensions(dimensions: List[str]) -> tuple[set[str], set[str]]:
    """
    Normalize user-supplied omit tokens into (pairwise_dim_keys, subjective_col_keys).
    """
    if not dimensions:
        return set(), set()

    # Aliases / synonyms
    alias_to_pairwise = {
        "frustration": "friction_loss_of_control",
        "friction": "friction_loss_of_control",
        "efficiency": "workflow_fit",
    }

    pairwise_omits: set[str] = set()
    subjective_omits: set[str] = set()

    for raw in dimensions:
        if raw is None:
            continue
        token = str(raw).strip().lower().replace("-", "_")
        if not token:
            continue

        # Explicit subjective column
        if token.startswith("subj_"):
            subjective_omits.add(token)
            # Also attempt to omit the corresponding pairwise key if it exists
            base = token.replace("subj_", "", 1)
            base = alias_to_pairwise.get(base, base)
            if base in PAIRWISE_DIMENSION_LABELS:
                pairwise_omits.add(base)
            continue

        # Handle common aliases first
        token = alias_to_pairwise.get(token, token)

        # Pairwise key direct
        if token in PAIRWISE_DIMENSION_LABELS:
            pairwise_omits.add(token)

        # Also map to subjective column if it exists
        subj_col = f"subj_{token}"
        if subj_col in DIMENSION_LABELS:
            subjective_omits.add(subj_col)

    return pairwise_omits, subjective_omits


# =============================================================================
# STYLE APPLICATION
# =============================================================================


def _apply_acl_style() -> None:
    """Configure publication-friendly seaborn style for ACL papers."""
    cfg = FIGURE_CONFIG

    sns.set_theme(style="whitegrid", context="paper", font_scale=cfg["font_scale"])

    plt.rcParams.update(
        {
            # Typography
            # "font.family": cfg["font_family"],
            # "font.serif": cfg["font_serif"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": cfg["title_size"],
            "axes.labelsize": cfg["label_size"],
            "xtick.labelsize": cfg["tick_size"],
            "ytick.labelsize": cfg["tick_size"],
            "legend.fontsize": cfg["legend_size"],
            "legend.title_fontsize": cfg["legend_title_size"],
            # Figure dimensions
            "figure.figsize": (cfg["default_width"], cfg["default_height"]),
            "figure.dpi": cfg["dpi"],
            # PDF/PS compatibility (editable fonts)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Legend styling
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.8",
            # Grid styling
            "axes.grid": True,
            "grid.alpha": cfg["grid_alpha"],
            "grid.linestyle": cfg["grid_linestyle"],
        }
    )


def _save_figure(fig, output_path: Path) -> Path:
    """
    Save figure in configured formats (PDF and/or PNG).

    Args:
        fig: Matplotlib figure object.
        output_path: Base path for output (extension will be adjusted).

    Returns:
        Path to the primary output file (PDF if enabled, else PNG).
    """
    cfg = FIGURE_CONFIG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally strip titles (axes titles + suptitle) to maximize usable space.
    if not cfg.get("show_titles", True):
        for ax in getattr(fig, "axes", []):
            try:
                ax.set_title("")
            except Exception:
                continue
        if getattr(fig, "_suptitle", None) is not None:
            try:
                fig._suptitle.remove()
            except Exception:
                try:
                    fig._suptitle.set_text("")
                except Exception:
                    pass
            fig._suptitle = None

    # Optionally re-tighten layout right before saving (after title stripping).
    if cfg.get("tight_layout_on_save", False):
        try:
            fig.tight_layout(
                pad=cfg.get("tight_layout_pad", 0.05),
                w_pad=cfg.get("tight_layout_w_pad", 0.15),
                h_pad=cfg.get("tight_layout_h_pad", 0.15),
            )
        except Exception:
            pass

    # Always save to the user-specified path (respecting its extension).
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=cfg.get("save_pad_inches", 0.02),
        dpi=cfg["dpi"],
    )
    primary_path = output_path
    logger.debug("Saved figure to %s", output_path)

    if cfg["save_pdf"]:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=cfg.get("save_pad_inches", 0.02),
            dpi=cfg["dpi"],
        )
        primary_path = pdf_path
        logger.debug("Saved figure PDF to %s", pdf_path)

    if cfg["save_png"]:
        png_path = output_path.with_suffix(".png")
        fig.savefig(
            png_path,
            format="png",
            bbox_inches="tight",
            pad_inches=cfg.get("save_pad_inches", 0.02),
            dpi=cfg["dpi"],
        )
        if not cfg["save_pdf"]:
            primary_path = png_path
        logger.debug("Saved figure PNG to %s", png_path)

    return primary_path


def _build_model_palette(df: pd.DataFrame) -> Dict[str, Tuple]:
    """Create a consistent color palette per model."""
    models = sorted(df["model_name"].dropna().unique())
    colors = sns.color_palette("colorblind", n_colors=max(len(models), 1))
    return {model: colors[idx] for idx, model in enumerate(models)}


# =============================================================================
# MAIN PLOT FUNCTIONS
# =============================================================================


def plot_persona_metric_bars(
    persona_df: pd.DataFrame,
    metric: str,
    output_path: str,
    metric_label: Optional[str] = None,
    title: Optional[str] = None,
    variant_palette: Optional[Dict[str, str]] = None,
    sample_df: Optional[pd.DataFrame] = None,
    show_error_bars: bool = True,
) -> Path:
    """
    Plot grouped bar charts by user_id, model, and variant for a single metric.

    Args:
        persona_df: Output of ``build_persona_summary`` (aggregated data).
        metric: Column to visualize (e.g., ``obj_overall_pass_at_1_mean``).
        output_path: Destination image path.
        metric_label: Friendly y-axis label.
        title: Figure title.
        variant_palette: Custom colors per variant.
        sample_df: Optional sample-level data for computing error bars.
        show_error_bars: Whether to show error bars (requires sample_df).

    Returns:
        Path to the saved figure.
    """
    # Determine which data source to use
    use_sample_data = sample_df is not None and show_error_bars

    # For sample data, derive the raw metric name from the aggregated name
    raw_metric = metric.replace("_mean", "") if metric.endswith("_mean") else metric

    if use_sample_data and raw_metric in sample_df.columns:
        data = sample_df.dropna(subset=[raw_metric]).copy()
        plot_metric = raw_metric
        errorbar_setting = ("sd", 1)  # Show 1 standard deviation
    elif metric in persona_df.columns:
        data = persona_df.dropna(subset=[metric]).copy()
        plot_metric = metric
        errorbar_setting = None  # No error bars for pre-aggregated data
    else:
        raise AnalysisDataError(f"Metric '{metric}' not found in data.")

    if data.empty:
        raise AnalysisDataError("Data is empty; cannot render bars.")

    if "user_id" not in data.columns:
        data = data.copy()
        fallback_user = (
            data.get("user_profile_type")
            if "user_profile_type" in data.columns
            else "all_users"
        )
        data["user_id"] = fallback_user

    _apply_acl_style()
    cfg = FIGURE_CONFIG
    palette = variant_palette or VARIANT_PALETTE
    metric_label = (
        metric_label or metric.replace("_", " ").replace("mean", "").title().strip()
    )
    title = title or f"{metric_label} by User"

    n_users = data["user_id"].nunique() if "user_id" in data.columns else 1
    col_wrap = min(cfg["col_wrap"], n_users)

    g = sns.catplot(
        data=data,
        x="model_name",
        y=plot_metric,
        hue="variant_label",
        col="user_id",
        col_wrap=col_wrap,
        kind="bar",
        palette=palette,
        hue_order=list(palette.keys()),
        height=cfg["catplot_height"],
        aspect=cfg["catplot_aspect"],
        sharey=False,
        edgecolor=cfg["bar_edgecolor"],
        linewidth=cfg["bar_linewidth"],
        errorbar=errorbar_setting,
        capsize=cfg["errorbar_capsize"] / 100 if errorbar_setting else 0,
        err_kws={"linewidth": cfg["errorbar_linewidth"]} if errorbar_setting else {},
    )

    g.set_axis_labels("Model", metric_label)
    g.set_titles("User: {col_name}")

    for ax in g.axes.flatten():
        ax.axhline(0, color="#999999", linewidth=0.8)
        # Set ticks first, then labels (best practice to avoid warnings)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=cfg["x_label_rotation"],
            ha="right",
            fontsize=cfg["tick_size"],
        )
        ax.grid(axis="y", linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    g.figure.subplots_adjust(top=0.88)
    g.figure.suptitle(title, fontsize=cfg["suptitle_size"], weight="bold")

    path = _save_figure(g.figure, Path(output_path))
    plt.close(g.figure)
    return path


def plot_personalization_deltas(
    delta_df: pd.DataFrame,
    metrics: Dict[str, str],
    output_path: str,
    title: str = "Personalization Deltas",
) -> Path:
    """
    Plot delta metrics (personalized minus original) per user and model.

    Note: Error bars are not shown for deltas as they represent single
    computed differences per user/model pair.

    Args:
        delta_df: Output of ``compute_user_model_deltas``.
        metrics: Mapping of column -> display label.
        output_path: Destination image path.
        title: Title for the figure.

    Returns:
        Path to the saved figure.
    """
    if delta_df.empty:
        raise AnalysisDataError("Delta table is empty; cannot render delta plot.")
    missing = [col for col in metrics if col not in delta_df.columns]
    if missing:
        raise AnalysisDataError(f"Missing delta columns: {missing}")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    id_vars = ["user_id", "model_name"]

    melted = delta_df.melt(
        id_vars=id_vars,
        value_vars=list(metrics.keys()),
        var_name="metric",
        value_name="delta_value",
    )
    melted["metric_label"] = melted["metric"].map(metrics)

    n_users = melted["user_id"].nunique() if "user_id" in melted.columns else 1
    col_wrap = min(cfg["col_wrap"], n_users)

    g = sns.catplot(
        data=melted,
        x="model_name",
        y="delta_value",
        hue="metric_label",
        col="user_id",
        col_wrap=col_wrap,
        kind="bar",
        height=cfg["catplot_height"],
        aspect=cfg["catplot_aspect"],
        sharey=False,
        palette="deep",
        edgecolor=cfg["bar_edgecolor"],
        linewidth=cfg["bar_linewidth"],
    )

    g.set_axis_labels("Model", "Δ Score (Personalized − Original)")
    g.set_titles("User: {col_name}")

    for ax in g.axes.flatten():
        ax.axhline(0, color="#444444", linewidth=1.5, linestyle="-")
        # Set ticks first, then labels (best practice to avoid warnings)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=cfg["x_label_rotation"],
            ha="right",
            fontsize=cfg["tick_size"],
        )
        ax.grid(axis="y", linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    g.figure.subplots_adjust(top=0.88)
    g.figure.suptitle(title, fontsize=cfg["suptitle_size"], weight="bold")

    path = _save_figure(g.figure, Path(output_path))
    plt.close(g.figure)
    return path


def plot_vibe_dimension_by_variant(
    sample_df: pd.DataFrame,
    variant_label: str,
    output_path: str,
    title: Optional[str] = None,
    show_error_bars: bool = True,
) -> Path:
    """
    Plot vibe dimensions comparing models for a specific variant (original or personalized).

    Creates a figure where:
    - X-axis: Vibe dimension names
    - Y-axis: Score (0-1)
    - Hue: Model names (different colors for each model)
    - Faceted by: user_id

    Args:
        sample_df: Sample-level data with raw vibe dimension columns.
        variant_label: Which variant to plot ("original" or "personalized").
        output_path: Destination image path.
        title: Optional custom title.
        show_error_bars: Whether to show standard deviation error bars.

    Returns:
        Path to the saved figure.
    """
    # Filter to the requested variant
    data = sample_df[sample_df["variant_label"] == variant_label].copy()
    if data.empty:
        raise AnalysisDataError(f"No data for variant '{variant_label}'.")

    # Find available vibe dimension columns (raw, not _mean), excluding omitted dims
    omit_subjective = FIGURE_CONFIG.get("omit_subjective_dimensions", set())
    dimension_cols = [
        col
        for col in VIBE_DIMENSION_COLUMNS
        if col in data.columns and col not in omit_subjective
    ]
    if not dimension_cols:
        raise AnalysisDataError("Sample data lacks vibe dimension columns.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    # Melt to long format
    id_vars = ["user_id", "model_name", "variant_label"]
    id_vars = [col for col in id_vars if col in data.columns]

    melted = data.melt(
        id_vars=id_vars,
        value_vars=dimension_cols,
        var_name="dimension",
        value_name="score",
    )
    if "user_id" in melted.columns:
        melted["persona_display"] = melted["user_id"].apply(translate_persona_name)
    melted["dimension_label"] = melted["dimension"].apply(
        lambda col: DIMENSION_LABELS.get(col, col.replace("subj_", "").title())
    )

    # Build model palette
    model_palette = _build_model_palette(melted)

    n_users = melted["user_id"].nunique() if "user_id" in melted.columns else 1
    col_wrap = min(cfg["col_wrap"], n_users)

    # Determine error bar setting
    errorbar_setting = ("sd", 1) if show_error_bars else None

    g = sns.catplot(
        data=melted,
        x="dimension_label",
        y="score",
        hue="model_name",
        col="persona_display" if "persona_display" in melted.columns else "user_id",
        col_wrap=col_wrap,
        kind="bar",
        palette=model_palette,
        height=cfg["catplot_height"],
        aspect=cfg["catplot_aspect"] + 0.3,  # Extra wide for dimension labels
        sharey=True,
        edgecolor=cfg["bar_edgecolor"],
        linewidth=cfg["bar_linewidth"],
        errorbar=errorbar_setting,
        capsize=cfg["errorbar_capsize"] / 100 if errorbar_setting else 0,
        err_kws={"linewidth": cfg["errorbar_linewidth"]} if errorbar_setting else {},
    )

    g.set_axis_labels("Vibe Dimension", "Score")
    g.set_titles("Persona: {col_name}")

    for ax in g.axes.flatten():
        # Set ticks first, then labels (best practice to avoid warnings)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=cfg["tick_size"]
        )
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    # Title with variant indicated
    variant_display = variant_label.capitalize()
    fig_title = title or f"Vibe Dimensions — {variant_display} Prompts"
    g.figure.subplots_adjust(top=0.88)
    g.figure.suptitle(fig_title, fontsize=cfg["suptitle_size"], weight="bold")

    path = _save_figure(g.figure, Path(output_path))
    plt.close(g.figure)
    return path


def plot_vibe_dimension_bars(
    variant_df: pd.DataFrame,
    output_path: str,
    title: str = "Vibe Dimension Comparison",
    sample_df: Optional[pd.DataFrame] = None,
    show_error_bars: bool = True,
) -> Path:
    """
    Plot bar charts of vibe dimensions per model, user, and variant.

    This is the legacy combined view. Consider using plot_vibe_dimension_by_variant
    for clearer separated views.

    Args:
        variant_df: Output of ``build_user_model_variant_summary`` (aggregated).
        output_path: Destination image path.
        title: Figure title.
        sample_df: Optional sample-level data for error bars.
        show_error_bars: Whether to show error bars (requires sample_df).

    Returns:
        Path to the saved figure.
    """
    # Determine data source
    use_sample_data = sample_df is not None and show_error_bars

    _apply_acl_style()
    cfg = FIGURE_CONFIG
    omit_subjective = cfg.get("omit_subjective_dimensions", set())

    if use_sample_data:
        dimension_cols = [
            col
            for col in VIBE_DIMENSION_COLUMNS
            if col in sample_df.columns and col not in omit_subjective
        ]
        data = sample_df
        errorbar_setting = ("sd", 1)
    else:
        dimension_cols = [
            f"{col}_mean"
            for col in VIBE_DIMENSION_COLUMNS
            if f"{col}_mean" in variant_df.columns and col not in omit_subjective
        ]
        data = variant_df
        errorbar_setting = None

    if not dimension_cols:
        raise AnalysisDataError("Data lacks vibe dimension columns.")

    id_vars = ["user_id", "model_name", "variant_label"]
    id_vars = [col for col in id_vars if col in data.columns]

    melted = data.melt(
        id_vars=id_vars,
        value_vars=dimension_cols,
        var_name="dimension",
        value_name="score",
    )
    if "user_id" in melted.columns:
        melted["persona_display"] = melted["user_id"].apply(translate_persona_name)
    melted["dimension_label"] = melted["dimension"].apply(
        lambda col: DIMENSION_LABELS.get(
            col.replace("_mean", ""),
            col.replace("_mean", "").replace("subj_", "").title(),
        )
    )

    # Facet by Model (columns) and User (rows)
    g = sns.catplot(
        data=melted,
        x="dimension_label",
        y="score",
        hue="variant_label",
        col="model_name",
        row="persona_display" if "persona_display" in melted.columns else "user_id",
        kind="bar",
        palette=VARIANT_PALETTE,
        hue_order=list(VARIANT_PALETTE.keys()),
        height=5,
        aspect=2.2,
        sharey=True,
        edgecolor=cfg["bar_edgecolor"],
        linewidth=cfg["bar_linewidth"],
        errorbar=errorbar_setting,
        capsize=cfg["errorbar_capsize"] / 100 if errorbar_setting else 0,
        err_kws={"linewidth": cfg["errorbar_linewidth"]} if errorbar_setting else {},
    )

    g.set_axis_labels("", "Score")
    g.set_titles("Persona: {row_name} | Model: {col_name}")

    for ax in g.axes.flatten():
        # Set ticks first, then labels (best practice to avoid warnings)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=cfg["tick_size"]
        )
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    g.figure.subplots_adjust(top=0.90)
    g.figure.suptitle(title, fontsize=cfg["suptitle_size"], weight="bold")

    path = _save_figure(g.figure, Path(output_path))
    plt.close(g.figure)
    return path


def plot_objective_vs_subjective_scatter(
    variant_df: pd.DataFrame,
    objective_metric: str,
    output_path: str,
    title: str = "Objective vs Subjective",
) -> Path:
    """
    Plot scatter of objective vs subjective performance, colored by model.

    Note: Scatter plots show individual data points, so error bars are not applicable.

    Args:
        variant_df: User/model/variant summary.
        objective_metric: Column to use on the x-axis.
        output_path: Destination image path.
        title: Figure title.

    Returns:
        Path to the saved scatter plot.
    """
    if objective_metric not in variant_df.columns:
        raise AnalysisDataError(f"Objective metric '{objective_metric}' missing.")
    if "subj_overall_mean" not in variant_df.columns:
        raise AnalysisDataError("variant_df must contain 'subj_overall_mean'.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG
    palette = _build_model_palette(variant_df)

    if "user_id" not in variant_df.columns:
        variant_df = variant_df.copy()
        fallback_user = (
            variant_df.get("user_profile_type")
            if "user_profile_type" in variant_df.columns
            else "all_users"
        )
        variant_df["user_id"] = fallback_user

    n_users = variant_df["user_id"].nunique() if "user_id" in variant_df.columns else 1
    col_wrap = min(cfg["col_wrap"], n_users)

    g = sns.relplot(
        data=variant_df,
        x=objective_metric,
        y="subj_overall_mean",
        hue="model_name",
        style="variant_label",
        col="user_id",
        col_wrap=col_wrap,
        kind="scatter",
        palette=palette,
        height=cfg["scatter_height"],
        aspect=cfg["scatter_aspect"],
        s=150,  # Larger markers
        edgecolor="white",
        linewidth=0.8,
    )

    g.set_axis_labels("Objective Score", "Subjective Score")
    g.set_titles("User: {col_name}")

    for ax in g.axes.flatten():
        ax.grid(True, linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    g.figure.subplots_adjust(top=0.88)
    g.figure.suptitle(title, fontsize=cfg["suptitle_size"], weight="bold")

    path = _save_figure(g.figure, Path(output_path))
    plt.close(g.figure)
    return path


# =============================================================================
# PAIRWISE COMPARISON PLOT FUNCTIONS
# =============================================================================

# Model name translations for display
MODEL_NAME_TRANSLATIONS: Dict[str, str] = {
    "gpt5": "GPT-5.1",
    "gpt-5.1-2025-11-13": "GPT-5.1",
    "gpt-5.1": "GPT-5.1",
    "gpt-oss-low-effort": "GPT-OSS-20B",
    "gpt-oss": "GPT-OSS-20B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "meta-llama-Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3-70B-Instruct": "Llama-3-70B",
    "meta-llama-Llama-3-70B-Instruct": "Llama-3-70B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "claude-3-5-sonnet-20241022": "Claude-3.5-Sonnet",
    "claude-3-opus-20240229": "Claude-3-Opus",
    "claude-3-5-sonnet": "Claude-3.5-Sonnet",
    "claude-3-opus": "Claude-3-Opus",
    "gpt-4-turbo-preview": "GPT-4-Turbo",
    "gpt-4-turbo": "GPT-4-Turbo",
    "gpt-4": "GPT-4",
    "gpt-3.5-turbo": "GPT-3.5-Turbo",
}

PERSONA_NAME_TRANSLATIONS: Dict[str, str] = {
    "novice_user": "Novice",
    "researcher_user": "Researcher",
    "intermediate_learner": "Intermediate Learner",
    "advanced_developer": "Advanced Developer",
}

# Prefix/pattern-based persona translations for nicer display in figures.
# These are matched against the raw `user_id` (often used as persona directory names).
PERSONA_PREFIX_TRANSLATIONS: Dict[str, str] = {
    "python_novice": "Beginner",
    "python_beginner": "Beginner",
    "python_intermediate": "Intermediate",
    "python_advanced": "Advanced",
    "python_expert": "Expert",
}


def translate_model_name(model_name: str) -> str:
    """
    Translate a model name to its display format.

    Args:
        model_name: Raw model name from data.

    Returns:
        Translated model name or original if no translation exists.
    """
    return display_model_name(model_name)


def translate_persona_name(persona_name: str) -> str:
    """
    Translate a persona name to its display format.

    Args:
        persona_name: Raw persona name from data.

    Returns:
        Translated persona name or original if no translation exists.
    """
    if not persona_name:
        return persona_name

    raw = str(persona_name)
    normalized = raw.strip().lower().replace("-", "_")

    # Prefer prefix matches (longest-first) for generated persona IDs.
    for prefix, display in sorted(
        PERSONA_PREFIX_TRANSLATIONS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if normalized.startswith(prefix):
            return display

    # Fallback: substring matches (longest-first) for other stable tokens.
    for token, display in sorted(
        PERSONA_NAME_TRANSLATIONS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if token in normalized:
            return display

    return raw


def extract_judge_model_info(
    df: pd.DataFrame, sample_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Extract and format judge model information from dataframes.

    Args:
        df: Primary dataframe (pair_summary, dimension_summary, etc.).
        sample_df: Optional sample-level dataframe with judge_model_name.

    Returns:
        Formatted string describing judge model(s), or empty string if none found.
    """
    judge_models = set()

    # Check primary dataframe
    if "judge_model_name" in df.columns:
        judge_models.update(df["judge_model_name"].dropna().unique())

    # Check sample dataframe if provided
    if sample_df is not None and "judge_model_name" in sample_df.columns:
        judge_models.update(sample_df["judge_model_name"].dropna().unique())

    if not judge_models:
        return ""

    # Translate model names and sort
    translated = sorted([translate_model_name(j) for j in judge_models])

    if len(translated) == 1:
        return f"Judge: {translated[0]}"
    elif len(translated) <= 3:
        return f"Judges: {', '.join(translated)}"
    else:
        # For many judges, show count and first few
        return f"Judges: {', '.join(translated[:3])} (+{len(translated) - 3} more)"


def extract_persona_info(
    df: pd.DataFrame, sample_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Extract and format persona/user information from dataframes.

    Uses user_id (shorter) instead of user_profile_type (long descriptions)
    to match the naming convention used in figure paths.

    Args:
        df: Primary dataframe (pair_summary, dimension_summary, etc.).
        sample_df: Optional sample-level dataframe with user_id/user_profile_type.

    Returns:
        Formatted string describing persona(s), or empty string if none found.
    """
    personas = set()

    # Prefer user_id (shorter, matches figure paths) over user_profile_type (long descriptions)
    if "user_id" in df.columns:
        personas.update(df["user_id"].dropna().unique())
    elif "user_profile_type" in df.columns:
        personas.update(df["user_profile_type"].dropna().unique())

    # Check sample dataframe if provided
    if sample_df is not None:
        if "user_id" in sample_df.columns:
            personas.update(sample_df["user_id"].dropna().unique())
        elif "user_profile_type" in sample_df.columns:
            personas.update(sample_df["user_profile_type"].dropna().unique())

    if not personas:
        return ""

    # Translate, sort and format
    sorted_personas = sorted(translate_persona_name(p) for p in personas)

    if len(sorted_personas) == 1:
        return f"Persona: {sorted_personas[0]}"
    elif len(sorted_personas) <= 3:
        return f"Personas: {', '.join(sorted_personas)}"
    else:
        # For many personas, show count and first few
        return f"Personas: {', '.join(sorted_personas[:3])} (+{len(sorted_personas) - 3} more)"


def extract_variant_info(
    df: pd.DataFrame, sample_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Extract and format variant information (original/personalized/all) from dataframes.

    Args:
        df: Primary dataframe (pair_summary, dimension_summary, etc.).
        sample_df: Optional sample-level dataframe with variant_label.

    Returns:
        Formatted string describing variant(s), or empty string if none found.
    """
    variants = set()

    # Check primary dataframe
    if "variant_label" in df.columns:
        variants.update(df["variant_label"].dropna().unique())

    # Check sample dataframe if provided
    if sample_df is not None and "variant_label" in sample_df.columns:
        variants.update(sample_df["variant_label"].dropna().unique())

    if not variants:
        return ""

    # Normalize variant labels
    normalized = set()
    for v in variants:
        v_lower = str(v).lower()
        if v_lower in ("original", "base"):
            normalized.add("Original")
        elif v_lower in ("personalized", "personalization", "variation"):
            normalized.add("Personalized")
        else:
            normalized.add(str(v).title())

    sorted_variants = sorted(normalized)

    if len(sorted_variants) == 1:
        return f"Prompts: {sorted_variants[0]}"
    elif (
        len(sorted_variants) == 2
        and "Original" in sorted_variants
        and "Personalized" in sorted_variants
    ):
        return "Prompts: All (Original + Personalized)"
    else:
        return f"Prompts: {', '.join(sorted_variants)}"


def build_figure_metadata_subtitle(
    df: pd.DataFrame, sample_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Build a combined subtitle string with judge, persona, and variant information.

    Args:
        df: Primary dataframe.
        sample_df: Optional sample-level dataframe.

    Returns:
        Formatted subtitle string with all metadata, or empty string if none found.
    """
    parts = []

    # Extract all metadata
    judge_info = extract_judge_model_info(df, sample_df)
    persona_info = extract_persona_info(df, sample_df)
    variant_info = extract_variant_info(df, sample_df)

    # Combine non-empty parts
    if judge_info:
        parts.append(judge_info)
    if persona_info:
        parts.append(persona_info)
    if variant_info:
        parts.append(variant_info)

    # Join with separator
    if parts:
        return " \n ".join(parts)
    return ""


# Pairwise dimension labels (without subj_ prefix)
PAIRWISE_DIMENSION_LABELS: Dict[str, str] = {
    "clarity": "Clarity",
    "tone_style_fit": "Tone & Style",
    "workflow_fit": "Workflow Fit",
    "cognitive_load": "Cognitive Load",
    "context_awareness": "Context Awareness",
    "persona_consistency": "Persona Consistency",
    "friction_loss_of_control": "Friction",
    "reliability_user_trust": "Reliability (User Trust)",
    "anthropomorphism": "Anthropomorphism",
    "objective_pass_at_1": "Pass@1",
    "objective_pass_at_5": "Pass@5",
    "objective_plus_pass_at_1": "Pass@1 (Plus)",
    # Backward compatibility
    "efficiency": "Workflow Fit",
    "frustration": "Friction",
}

# Modern color palette for pairwise results - inspired by big tech design
PAIRWISE_PALETTE: Dict[str, str] = {
    "model_a": "#007BFF",  # “brand blue” # "#00BFFF",  # Deep Sky Blue (Model A wins)
    "model_b": "#FF1493",  # Hot Pink (Model B wins)
    "tie": "#6B7280",  # Neutral Gray (Ties)
    "model_a_light": "#BFEFFF",  # Light blue for backgrounds
    "model_b_light": "#FFB6D9",  # Light pink for backgrounds
    "tie_light": "#D1D5DB",  # Light Gray for backgrounds
}

# Modern gradient-friendly colors with alpha support
MODERN_COLORS = {
    "primary": "#00BFFF",  # Deep Sky Blue
    "secondary": "#FF1493",  # Hot Pink
    "accent": "#059669",  # Emerald
    "neutral": "#6B7280",  # Gray
    "background": "#F9FAFB",  # Light background
    "grid": "#E5E7EB",  # Grid lines
    "text": "#1F2937",  # Dark text
    "text_light": "#6B7280",  # Light text
}

LIMA_STACKED_PALETTE: Dict[str, str] = {
    # LIMA-style monochrome palette (dark -> medium -> light).
    "left": "#1565C0",
    "mid": "#5C9DFF",
    "right": "#CFE2FF",
}


def _apply_lima_bar_style(ax: Any, cfg: dict) -> None:
    """
    Apply styling that mimics the LIMA stacked bar figure aesthetic.

    Args:
        ax: Matplotlib axes.
        cfg: FIGURE_CONFIG dict (used for font sizes where relevant).
    """
    ax.set_facecolor("white")
    ax.grid(False)

    # Keep a thin black box around the axes.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(axis="both", which="both", direction="out", length=3, width=0.8)

    # Percent axis formatting.
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=cfg.get("tick_size", 9))
    ax.tick_params(axis="y", labelsize=cfg.get("tick_size", 9))


def _text_color_for_hex(hex_color: str) -> str:
    """
    Choose a readable text color for a bar segment.

    Args:
        hex_color: Segment fill color.

    Returns:
        A contrasting text color ("black" for light fills; otherwise "white").
    """
    rgb = np.array(plt.matplotlib.colors.to_rgb(hex_color))
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.70 else "white"


def plot_pairwise_win_rates(
    pair_summary: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    show_model_names: bool = True,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot horizontal stacked bar chart showing win/tie/loss proportions per model pair.

    Modern, clean visualization with beautiful colors and clear labeling.

    Args:
        pair_summary: Output from compute_pair_summary().
        output_path: Destination image path.
        title: Optional custom title.
        show_model_names: If True, use actual model names in legend.
        sample_df: Optional sample-level data for extracting judge model info.

    Returns:
        Path to the saved figure.
    """
    if pair_summary.empty:
        raise AnalysisDataError("Pair summary is empty; cannot render win rate plot.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    # Prepare data for stacked bar
    data = pair_summary.copy()

    # Extract and translate model names for legend
    model_a_name = (
        translate_model_name(data["model_a_name"].iloc[0])
        if show_model_names and "model_a_name" in data.columns
        else "Model A"
    )
    model_b_name = (
        translate_model_name(data["model_b_name"].iloc[0])
        if show_model_names and "model_b_name" in data.columns
        else "Model B"
    )

    # Create pair labels with translated names
    data["pair_label"] = data.apply(
        lambda r: f"{translate_model_name(r['model_a_name'])} vs {translate_model_name(r['model_b_name'])}",
        axis=1,
    )

    # Sort by model_a win rate descending
    data = data.sort_values("model_a_win_rate", ascending=True)

    # Create figure with modern styling
    fig, ax = plt.subplots(figsize=(12, max(4, len(data) * 1.2)))
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(data))
    bar_height = 0.65

    # Stack: Model A wins, then Ties, then Model B wins
    a_wins = data["model_a_win_rate"].values
    ties = data["tie_rate"].values
    b_wins = data["model_b_win_rate"].values

    # Create stacked bars with modern colors and transparency
    ax.barh(
        y_pos,
        a_wins,
        bar_height,
        label=model_a_name,
        color=PAIRWISE_PALETTE["model_a"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.barh(
        y_pos,
        ties,
        bar_height,
        left=a_wins,
        label="Tie",
        color=PAIRWISE_PALETTE["tie"],
        alpha=0.65,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.barh(
        y_pos,
        b_wins,
        bar_height,
        left=a_wins + ties,
        label=model_b_name,
        color=PAIRWISE_PALETTE["model_b"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    # Add percentage labels on bars
    for i, (a, t, b) in enumerate(zip(a_wins, ties, b_wins)):
        if a >= 0.08:
            ax.text(
                a / 2,
                i,
                f"{a:.0%}",
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                fontweight="bold",
            )
        if t >= 0.08:
            ax.text(
                a + t / 2,
                i,
                f"{t:.0%}",
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="medium",
            )
        if b >= 0.08:
            ax.text(
                a + t + b / 2,
                i,
                f"{b:.0%}",
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        data["pair_label"],
        fontsize=cfg["tick_size"],
        fontweight="medium",
        color=MODERN_COLORS["text"],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        "Win Rate Distribution",
        fontsize=cfg["label_size"],
        color=MODERN_COLORS["text"],
    )

    # Center reference line
    ax.axvline(
        0.5, color=MODERN_COLORS["text_light"], linestyle="--", linewidth=1.5, alpha=0.5
    )

    # Apply modern styling
    _apply_modern_style(ax, cfg)

    # Elegant legend
    legend = ax.legend(
        loc="lower right",
        fontsize=cfg["legend_size"] - 1,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor=MODERN_COLORS["grid"],
        ncol=3,
    )
    legend.get_frame().set_linewidth(0.5)

    # Title with metadata subtitle
    fig_title = title or "Overall Pairwise Win Rate Comparison"
    metadata_subtitle = build_figure_metadata_subtitle(data, sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(
        fig_title,
        fontsize=cfg["title_size"],
        fontweight="bold",
        color=MODERN_COLORS["text"],
        pad=20,
    )

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def _apply_modern_style(ax, cfg: dict) -> None:
    """Apply modern, clean styling to an axes object."""
    # Clean background
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MODERN_COLORS["grid"])
    ax.spines["bottom"].set_color(MODERN_COLORS["grid"])
    ax.tick_params(colors=MODERN_COLORS["text"], which="both")
    ax.grid(axis="both", linestyle="-", alpha=0.3, color=MODERN_COLORS["grid"])


def plot_pairwise_dimension_comparison(
    dimension_summary: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    model_pair: Optional[str] = None,
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
    style: str = "modern",
) -> Path:
    """
    Plot horizontal stacked bar chart showing win rates per dimension.

    Creates a modern, clean visualization with dimensions on Y-axis and
    stacked bars showing Model A wins, Ties, and Model B wins proportions.

    Args:
        dimension_summary: Output from compute_dimension_win_rates().
        output_path: Destination image path.
        title: Optional custom title.
        model_pair: If specified, filter to this model pair.
        model_a_name: Display name for Model A (extracted from data if not provided).
        model_b_name: Display name for Model B (extracted from data if not provided).
        sample_df: Optional sample-level data for extracting judge model info.
        style: Plot styling preset. "modern" (default) retains the existing look.
            "lima" mimics the compact, monochrome stacked-bar style used in the LIMA paper.

    Returns:
        Path to the saved figure.
    """
    if dimension_summary.empty:
        raise AnalysisDataError("Dimension summary is empty; cannot render plot.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    data = dimension_summary.copy()
    if model_pair and "model_pair" in data.columns:
        data = data[data["model_pair"] == model_pair]

    omit_pairwise = cfg.get("omit_pairwise_dimensions", set())
    if omit_pairwise and "dimension" in data.columns:
        data = data[~data["dimension"].isin(omit_pairwise)].copy()

    if data.empty:
        raise AnalysisDataError(f"No data for model pair '{model_pair}'.")

    # Extract and translate model names from data if not provided
    if model_a_name is None and "model_a_name" in data.columns:
        model_a_name = translate_model_name(data["model_a_name"].iloc[0])
    if model_b_name is None and "model_b_name" in data.columns:
        model_b_name = translate_model_name(data["model_b_name"].iloc[0])

    # Fallback names
    model_a_name = model_a_name or "Model A"
    model_b_name = model_b_name or "Model B"

    # Add dimension labels
    data["dimension_label"] = data["dimension"].map(
        lambda d: PAIRWISE_DIMENSION_LABELS.get(d, d.replace("_", " ").title())
    )

    # Aggregate by dimension (in case there are multiple variants)
    agg_cols = {
        "dimension_label": "first",
        "model_a_win_rate": "mean",
        "model_b_win_rate": "mean",
        "tie_rate": "mean",
        "total_comparisons": "sum",
    }
    data = data.groupby("dimension").agg(agg_cols).reset_index()

    # Sort dimensions by a consistent order (reversed for horizontal bar)
    # Ensure objective pass@k dimensions render at the top of the stacked bars
    objective_dims = [
        d for d in PAIRWISE_DIMENSION_LABELS if d.startswith("objective_pass_at_")
    ]
    base_dims = [
        d
        for d in PAIRWISE_DIMENSION_LABELS
        if d not in objective_dims and d not in omit_pairwise
    ]
    objective_dims = [d for d in objective_dims if d not in omit_pairwise]
    dim_order = base_dims + objective_dims

    data["dim_order"] = data["dimension"].apply(
        lambda d: dim_order.index(d) if d in dim_order else len(dim_order)
    )
    data = data.sort_values("dim_order", ascending=False)

    style_normalized = str(style or "modern").strip().lower()
    if style_normalized not in {"modern", "lima"}:
        raise AnalysisDataError(
            f"Unknown style '{style}'. Expected 'modern' or 'lima'."
        )

    # Create figure
    if style_normalized == "lima":
        # ACL single-column friendly defaults; height scales with number of dimensions.
        # Narrower + taller so bars are less visually "long and thin" and labels fit.
        fig_width = 9.0  # 5.0
        fig_height = max(3.2, len(data) * 0.42)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig, ax = plt.subplots(figsize=(12, max(5, len(data) * 0.7)))
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(data))
    # LIMA-style uses thicker bars so the percentage text fits comfortably.
    bar_height = 0.90 if style_normalized == "lima" else 0.65

    # LIMA mode prefers compact, non-bold typography even if global defaults are larger.
    # User request: increase the LIMA font sizes by +6pt relative to the "compact" baseline.
    base_lima_tick_size = min(int(cfg.get("tick_size", 9)), 9)
    base_lima_legend_size = min(int(cfg.get("legend_size", 9)), 9)
    lima_tick_size = base_lima_tick_size + 6
    lima_legend_size = base_lima_legend_size + 6

    # Get values
    a_wins = data["model_a_win_rate"].values
    ties = data["tie_rate"].values
    b_wins = data["model_b_win_rate"].values

    if style_normalized == "lima":
        # LIMA/ACL stacked bars: monochrome, no alpha, no edges.
        ax.barh(
            y_pos,
            a_wins,
            bar_height,
            label=model_a_name,
            color=LIMA_STACKED_PALETTE["left"],
        )
        ax.barh(
            y_pos,
            ties,
            bar_height,
            left=a_wins,
            label="Tie",
            color=LIMA_STACKED_PALETTE["mid"],
        )
        ax.barh(
            y_pos,
            b_wins,
            bar_height,
            left=a_wins + ties,
            label=model_b_name,
            color=LIMA_STACKED_PALETTE["right"],
        )
    else:
        # Modern colors with transparency and white separators.
        ax.barh(
            y_pos,
            a_wins,
            bar_height,
            label=model_a_name,
            color=PAIRWISE_PALETTE["model_a"],
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )
        ax.barh(
            y_pos,
            ties,
            bar_height,
            left=a_wins,
            label="Tie",
            color=PAIRWISE_PALETTE["tie"],
            alpha=0.65,
            edgecolor="white",
            linewidth=1,
        )
        ax.barh(
            y_pos,
            b_wins,
            bar_height,
            left=a_wins + ties,
            label=model_b_name,
            color=PAIRWISE_PALETTE["model_b"],
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )

    # Add percentage labels inside bars (only if segment is wide enough)
    for i, (a, t, b) in enumerate(zip(a_wins, ties, b_wins)):
        if style_normalized == "lima":
            a_color = _text_color_for_hex(LIMA_STACKED_PALETTE["left"])
            t_color = _text_color_for_hex(LIMA_STACKED_PALETTE["mid"])
            b_color = _text_color_for_hex(LIMA_STACKED_PALETTE["right"])
            a_fs = lima_tick_size
            t_fs = max(6, lima_tick_size - 1)
        else:
            a_color = "white"
            t_color = "white"
            b_color = "white"
            a_fs = 11
            t_fs = 10

        # Model A label
        if a >= 0.12:
            ax.text(
                a / 2,
                i,
                f"{a:.0%}",
                ha="center",
                va="center",
                fontsize=a_fs,
                color=a_color,
                fontweight="normal",
            )
        # Tie label
        if t >= 0.12:
            ax.text(
                a + t / 2,
                i,
                f"{t:.0%}",
                ha="center",
                va="center",
                fontsize=t_fs,
                color=t_color,
                fontweight="normal",
            )
        # Model B label
        if b >= 0.12:
            ax.text(
                a + t + b / 2,
                i,
                f"{b:.0%}",
                ha="center",
                va="center",
                fontsize=a_fs,
                color=b_color,
                fontweight="normal",
            )

    # Style the axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        data["dimension_label"].values,
        fontsize=lima_tick_size if style_normalized == "lima" else cfg["tick_size"],
        fontweight="normal" if style_normalized == "lima" else "medium",
        color=MODERN_COLORS["text"],
    )
    ax.set_xlim(0, 1)
    if style_normalized == "lima":
        ax.set_xlabel("")
        _apply_lima_bar_style(ax, {"tick_size": lima_tick_size})

        # Ensure ALL dimension tick labels (including Pass@K) share the same font settings.
        default_family = plt.rcParams.get("font.family")
        family = (
            default_family[0]
            if isinstance(default_family, (list, tuple)) and default_family
            else str(default_family or "sans-serif")
        )
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(lima_tick_size)
            lbl.set_fontweight("normal")
            lbl.set_fontfamily(family)

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=lima_legend_size,
        )
        # LIMA-style: omit in-plot title/subtitle; paper caption carries it.
        ax.set_title("")
    else:
        ax.set_xlabel(
            "Win Rate Distribution",
            fontsize=cfg["label_size"],
            color=MODERN_COLORS["text"],
        )

        # Add center reference line
        ax.axvline(
            0.5,
            color=MODERN_COLORS["text_light"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
        )

        # Apply modern styling
        _apply_modern_style(ax, cfg)

        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),  # x: left of axes; y: just above axes
            fontsize=cfg["legend_size"] - 1,
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.95,
            edgecolor=MODERN_COLORS["grid"],
            ncol=3,
            borderaxespad=0.3,
        )
        legend.get_frame().set_linewidth(0.5)

        # Build title with metadata subtitle
        if title:
            fig_title = title
        else:
            fig_title = f"Win Rate by Dimension: {model_a_name} vs {model_b_name}"

        metadata_subtitle = build_figure_metadata_subtitle(data, sample_df)
        if metadata_subtitle:
            fig_title = f"{fig_title}\n{metadata_subtitle}"

        ax.set_title(
            fig_title,
            fontsize=cfg["title_size"],
            fontweight="bold",
            color=MODERN_COLORS["text"],
            pad=40,  # make sure title is above the legend
        )

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_pairwise_dimension_heatmap(
    dimension_summary: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> Path:
    """
    Plot a heatmap showing win rates at the dimension level.

    Creates a beautiful heatmap visualization with dimensions on rows
    and model pairs on columns, using a diverging color scale.

    Args:
        dimension_summary: Output from compute_dimension_win_rates().
        output_path: Destination image path.
        title: Optional custom title.
        sample_df: Optional sample-level data for extracting judge model info.

    Returns:
        Path to the saved figure.
    """
    if dimension_summary.empty:
        raise AnalysisDataError("Dimension summary is empty; cannot render heatmap.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    data = dimension_summary.copy()
    omit_pairwise = cfg.get("omit_pairwise_dimensions", set())
    if omit_pairwise and "dimension" in data.columns:
        data = data[~data["dimension"].isin(omit_pairwise)].copy()

    # Add dimension labels
    data["dimension_label"] = data["dimension"].map(
        lambda d: PAIRWISE_DIMENSION_LABELS.get(d, d.replace("_", " ").title())
    )

    if not (0.0 < float(alpha) < 1.0):
        raise AnalysisDataError(f"alpha must be in (0,1). Got: {alpha}")

    # Aggregate by dimension (mean across variants) while preserving counts
    agg_cols = {
        "dimension_label": "first",
        "model_a_win_rate": "mean",
        "model_b_win_rate": "mean",
        "tie_rate": "mean",
        "model_a_name": "first",
        "model_b_name": "first",
        "model_a_wins": "sum",
        "model_b_wins": "sum",
        "ties": "sum",
    }
    data = data.groupby("dimension").agg(agg_cols).reset_index()

    # Sort dimensions
    dim_order = [
        d for d in list(PAIRWISE_DIMENSION_LABELS.keys()) if d not in omit_pairwise
    ]
    data["dim_order"] = data["dimension"].apply(
        lambda d: dim_order.index(d) if d in dim_order else len(dim_order)
    )
    data = data.sort_values("dim_order")

    # Get and translate model names
    model_a_name = (
        translate_model_name(data["model_a_name"].iloc[0])
        if "model_a_name" in data.columns
        else "Model A"
    )
    model_b_name = (
        translate_model_name(data["model_b_name"].iloc[0])
        if "model_b_name" in data.columns
        else "Model B"
    )

    # Create matrix for heatmap: dimensions x [Model A, Tie, Model B]
    matrix_data = np.column_stack(
        [
            data["model_a_win_rate"].values,
            data["tie_rate"].values,
            data["model_b_win_rate"].values,
        ]
    )

    # Determine winner for each dimension (higher win rate)
    model_a_wins = data["model_a_win_rate"].values > data["model_b_win_rate"].values

    # Significance per dimension (two-sided binomial test on non-ties)
    sig_flags: List[bool] = []
    for a_w, b_w in zip(data["model_a_wins"].values, data["model_b_wins"].values):
        n = int(a_w) + int(b_w)
        if n <= 0:
            sig_flags.append(False)
            continue
        p_value = float(stats.binomtest(int(a_w), n, p=0.5).pvalue)
        sig_flags.append(p_value < alpha)

    # ------------------------------------------------------------------
    # Alpha mapping (bucketed)
    #
    # ≥ 0.90: very opaque
    # 0.70–0.90: clearly visible
    # 0.50–0.70: slightly transparent
    # 0.20–0.50: transparent
    # 0.00–0.20: very transparent
    # ------------------------------------------------------------------

    def _alpha_from_win_rate_bucketed(value: float) -> float:
        v = float(np.clip(value, 0.0, 1.0))
        if v >= 0.90:
            return 0.92  # very opaque
        if v >= 0.70:
            return 0.72  # clearly visible
        if v >= 0.50:
            return 0.64  # slightly transparent (more visible)
        if v >= 0.20:
            return 0.32  # transparent
        return 0.16  # very transparent

    ALPHA_TIE = 0.55  # Medium alpha for tie column
    ALPHA_NEUTRAL = 0.35  # Used for both sides when not significant

    # Create figure
    n_dims = len(data)
    fig, ax = plt.subplots(figsize=(8, max(6, n_dims * 0.6)))
    fig.patch.set_facecolor("#FFFFFF")

    # Create custom colormap for each column
    # We'll plot each column separately with its own color
    y_pos = np.arange(n_dims)

    # Create a combined heatmap effect with three columns
    for col_idx, (col_name, base_color) in enumerate(
        [
            (model_a_name, PAIRWISE_PALETTE["model_a"]),
            ("Tie", PAIRWISE_PALETTE["tie"]),
            (model_b_name, PAIRWISE_PALETTE["model_b"]),
        ]
    ):
        values = matrix_data[:, col_idx]

        # Create bars with fixed alpha based on winner/loser status
        for i, val in enumerate(values):
            if col_idx == 1:  # Tie column
                alpha_val = ALPHA_TIE
            else:
                if not sig_flags[i]:
                    alpha_val = ALPHA_NEUTRAL
                else:
                    # Significant: use alpha tied to the win-rate bucket for this cell.
                    alpha_val = _alpha_from_win_rate_bucketed(val)

            rect = plt.Rectangle(
                (col_idx, i - 0.4),
                1,
                0.8,
                facecolor=base_color,
                alpha=alpha_val,
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Add value label
            label = f"{val:.0%}"
            if col_idx != 1 and sig_flags[i]:
                label = f"{label}*"
            ax.text(
                col_idx + 0.5,
                i,
                label,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if alpha_val > 0.5 else MODERN_COLORS["text"],
            )

    # Set axis limits and labels
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, n_dims - 0.5)

    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(
        [model_a_name, "Tie", model_b_name],
        fontsize=cfg["tick_size"],
        fontweight="medium",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        data["dimension_label"].values,
        fontsize=cfg["tick_size"],
        fontweight="medium",
    )

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Title with metadata subtitle
    fig_title = title or f"Dimension-Level Win Rates: {model_a_name} vs {model_b_name}"
    metadata_subtitle = build_figure_metadata_subtitle(dimension_summary, sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(
        fig_title,
        fontsize=cfg["title_size"],
        fontweight="bold",
        color=MODERN_COLORS["text"],
        pad=20,
    )

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_position_bias_rates(
    dimension_summary: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot horizontal bar chart showing position bias rate per dimension.

    Modern, clean visualization highlighting potential evaluation biases.

    Args:
        dimension_summary: Output from compute_dimension_win_rates().
        output_path: Destination image path.
        title: Optional custom title.
        sample_df: Optional sample-level data for extracting judge model info.

    Returns:
        Path to the saved figure.
    """
    if dimension_summary.empty:
        raise AnalysisDataError("Dimension summary is empty; cannot render bias plot.")

    if "position_bias_rate" not in dimension_summary.columns:
        raise AnalysisDataError(
            "Missing 'position_bias_rate' column in dimension summary."
        )

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    data = dimension_summary.copy()
    omit_pairwise = cfg.get("omit_pairwise_dimensions", set())
    if omit_pairwise and "dimension" in data.columns:
        data = data[~data["dimension"].isin(omit_pairwise)].copy()

    # Add dimension labels
    data["dimension_label"] = data["dimension"].map(
        lambda d: PAIRWISE_DIMENSION_LABELS.get(d, d.replace("_", " ").title())
    )

    # Aggregate across model pairs if multiple exist
    if "model_pair" in data.columns and data["model_pair"].nunique() > 1:
        data = (
            data.groupby("dimension")
            .agg(
                {
                    "dimension_label": "first",
                    "position_bias_rate": "mean",
                    "position_bias_count": "sum",
                    "total_comparisons": "sum",
                }
            )
            .reset_index()
        )

    # Sort by dimension order (reversed for horizontal)
    dim_order = [
        d for d in list(PAIRWISE_DIMENSION_LABELS.keys()) if d not in omit_pairwise
    ]
    data["dim_order"] = data["dimension"].apply(
        lambda d: dim_order.index(d) if d in dim_order else len(dim_order)
    )
    data = data.sort_values("dim_order", ascending=False)

    # Modern figure
    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.7)))
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(data))
    bias_rates = data["position_bias_rate"].values

    # Create gradient colors based on bias rate (more bias = more red)
    colors = []
    for rate in bias_rates:
        if rate < 0.2:
            colors.append("#10B981")  # Green - low bias
        elif rate < 0.4:
            colors.append("#F59E0B")  # Amber - moderate bias
        else:
            colors.append("#EF4444")  # Red - high bias

    bars = ax.barh(
        y_pos,
        bias_rates,
        height=0.6,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    # Add value labels
    for i, (rect, rate) in enumerate(zip(bars, bias_rates)):
        width = rect.get_width()
        label_x = width + 0.02 if width < 0.7 else width - 0.08
        label_color = MODERN_COLORS["text"] if width < 0.7 else "white"
        ax.text(
            label_x,
            rect.get_y() + rect.get_height() / 2,
            f"{rate:.0%}",
            ha="left" if width < 0.7 else "right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=label_color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        data["dimension_label"].values,
        fontsize=cfg["tick_size"],
        fontweight="medium",
        color=MODERN_COLORS["text"],
    )
    ax.set_xlim(0, min(1.0, bias_rates.max() + 0.15))
    ax.set_xlabel(
        "Position Bias Rate",
        fontsize=cfg["label_size"],
        color=MODERN_COLORS["text"],
    )

    # Apply modern styling
    _apply_modern_style(ax, cfg)

    # Add reference lines
    ax.axvline(
        0.2,
        color="#10B981",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Low threshold",
    )
    ax.axvline(
        0.4,
        color="#EF4444",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="High threshold",
    )

    fig_title = title or "Position Bias Detection by Dimension"
    metadata_subtitle = build_figure_metadata_subtitle(dimension_summary, sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(
        fig_title,
        fontsize=cfg["title_size"],
        fontweight="bold",
        color=MODERN_COLORS["text"],
        pad=20,
    )

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_pairwise_by_user(
    user_pair_summary: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot faceted bar charts showing pairwise preferences by user.

    Args:
        user_pair_summary: Output from compute_user_pair_summary().
        output_path: Destination image path.
        title: Optional custom title.
        sample_df: Optional sample-level data for extracting judge model info.

    Returns:
        Path to the saved figure.
    """
    if user_pair_summary.empty:
        raise AnalysisDataError("User pair summary is empty; cannot render plot.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    data = user_pair_summary.copy()
    if "user_id" in data.columns:
        data["persona_display"] = data["user_id"].apply(translate_persona_name)

    # Translate model names for labels
    model_a_name = translate_model_name(data["model_a_name"].iloc[0])
    model_b_name = translate_model_name(data["model_b_name"].iloc[0])

    # Melt for grouped bar plot
    melted = data.melt(
        id_vars=[
            col
            for col in ["user_id", "persona_display", "model_pair"]
            if col in data.columns
        ],
        value_vars=["model_a_win_rate", "model_b_win_rate", "tie_rate"],
        var_name="outcome",
        value_name="rate",
    )
    melted["outcome_label"] = melted["outcome"].map(
        {
            "model_a_win_rate": model_a_name,  # "Model A",
            "model_b_win_rate": model_b_name,
            "tie_rate": "Tie",
        }
    )

    n_users = melted["user_id"].nunique()
    col_wrap = min(cfg["col_wrap"], n_users)

    g = sns.catplot(
        data=melted,
        x="model_pair",
        y="rate",
        hue="outcome_label",
        col="persona_display" if "persona_display" in melted.columns else "user_id",
        col_wrap=col_wrap,
        kind="bar",
        # with std lines
        errorbar=("sd", 1),
        capsize=cfg["errorbar_capsize"] / 100,
        err_kws={"linewidth": cfg["errorbar_linewidth"]},
        palette=[
            PAIRWISE_PALETTE["model_a"],
            PAIRWISE_PALETTE["model_b"],
            PAIRWISE_PALETTE["tie"],
        ],
        height=cfg["catplot_height"],
        aspect=cfg["catplot_aspect"],
        sharey=True,
        edgecolor=cfg["bar_edgecolor"],
        linewidth=cfg["bar_linewidth"],
    )

    g.set_axis_labels("Model Pair", "Win Rate")
    g.set_titles("Persona: {col_name}")

    for ax in g.axes.flatten():
        # Set ticks first, then labels (best practice to avoid warnings)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=cfg["x_label_rotation"],
            ha="right",
            fontsize=cfg["tick_size"],
        )
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    fig_title = title or "Pairwise Preferences by User"
    metadata_subtitle = build_figure_metadata_subtitle(data, sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    g.figure.subplots_adjust(top=0.88)
    g.figure.suptitle(fig_title, fontsize=cfg["suptitle_size"], weight="bold")

    path = _save_figure(g.figure, Path(output_path))
    plt.close(g.figure)
    return path


# =============================================================================
# MULTI-PAIR COMPARISON PLOT FUNCTIONS
# =============================================================================


def plot_preference_matrix_heatmap(
    preference_matrix: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    annotate: bool = True,
    sample_df: Optional[pd.DataFrame] = None,
    statistical_tests: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> Path:
    """
    Plot N x N heatmap of pairwise win rates between models.

    The cell (i, j) shows the win rate of model i against model j.
    Darker blue indicates higher win rate, darker red indicates lower.
    Diagonal is neutral (0.5).

    Args:
        preference_matrix: Output from build_preference_matrix().
        output_path: Destination image path.
        title: Optional custom title.
        annotate: If True, show percentage values in cells.
        sample_df: Optional sample-level data for extracting judge model info.
        statistical_tests: Optional DataFrame with per-pair p-values from
            compute_statistical_significance(). If provided, non-significant cells
            are colored as neutral (0.5) while retaining numeric annotations.
        alpha: Significance threshold for coloring and '*' markers.

    Returns:
        Path to the saved figure.
    """
    if preference_matrix.empty:
        raise AnalysisDataError("Preference matrix is empty; cannot render heatmap.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    # Translate model names in index and columns
    preference_matrix = preference_matrix.copy()
    preference_matrix.index = [translate_model_name(m) for m in preference_matrix.index]
    preference_matrix.columns = [
        translate_model_name(m) for m in preference_matrix.columns
    ]

    # If statistical tests are provided, only show "strong" colors for significant pairs.
    # Non-significant pairs are colored as neutral (0.5), but annotations retain the
    # underlying numeric values and mark significant cells with '*'.
    heat_values = preference_matrix.copy()
    annot_matrix = None
    if statistical_tests is not None and not statistical_tests.empty:
        if not (0.0 < float(alpha) < 1.0):
            raise AnalysisDataError(f"alpha must be in (0,1). Got: {alpha}")

        # Build a lookup on untranslated model names; translate for lookup symmetry.
        tests = statistical_tests.copy()
        required_cols = {"model_a_name", "model_b_name", "binomial_pvalue"}
        if not required_cols.issubset(set(tests.columns)):
            raise AnalysisDataError(
                "statistical_tests is missing required columns: "
                + ", ".join(sorted(required_cols - set(tests.columns)))
            )
        tests["model_a_name"] = tests["model_a_name"].apply(translate_model_name)
        tests["model_b_name"] = tests["model_b_name"].apply(translate_model_name)
        tests_lookup = {}
        for _, row in tests.iterrows():
            a = row["model_a_name"]
            b = row["model_b_name"]
            p = row["binomial_pvalue"]
            if pd.notna(a) and pd.notna(b) and pd.notna(p):
                tests_lookup[(a, b)] = float(p)

        def _pvalue_for_pair(i: str, j: str) -> Optional[float]:
            if (i, j) in tests_lookup:
                return tests_lookup[(i, j)]
            if (j, i) in tests_lookup:
                return tests_lookup[(j, i)]
            return None

        annot_matrix = pd.DataFrame(
            "", index=heat_values.index, columns=heat_values.columns
        )
        for i in heat_values.index:
            for j in heat_values.columns:
                if i == j:
                    annot_matrix.loc[i, j] = "50%"
                    continue
                p = _pvalue_for_pair(i, j)
                sig = bool(p is not None and p < alpha)
                val = float(preference_matrix.loc[i, j])
                annot_matrix.loc[i, j] = f"{val:.0%}" + ("*" if sig else "")
                if not sig:
                    heat_values.loc[i, j] = 0.5

    n_models = len(preference_matrix)
    fig_size = max(8, n_models * 1.2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Create diverging colormap centered at 0.5
    cmap = _joint_heatmap_cmap_with_alpha()

    # Plot heatmap
    sns.heatmap(
        heat_values,
        ax=ax,
        annot=annot_matrix if annot_matrix is not None else annotate,
        fmt="" if annot_matrix is not None else ".0%",
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Win Rate", "shrink": 0.8},
        annot_kws={"fontsize": cfg["tick_size"] - 2, "fontweight": "bold"},
    )

    # Labels
    ax.set_xlabel("Opponent Model", fontsize=cfg["label_size"])
    ax.set_ylabel("Model", fontsize=cfg["label_size"])
    ax.tick_params(axis="both", labelsize=cfg["tick_size"])

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig_title = title or "Pairwise Preference Matrix"
    # Preference matrix doesn't contain metadata, so only check sample_df
    metadata_subtitle = build_figure_metadata_subtitle(pd.DataFrame(), sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(fig_title, fontsize=cfg["title_size"], fontweight="bold", pad=15)

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_joint_preference_matrix_heatmap(
    win_rate_matrix: pd.DataFrame,
    annotation_matrix: Optional[pd.DataFrame],
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot a joint preference matrix heatmap with optional string annotations.

    This is similar to plot_preference_matrix_heatmap, but supports arbitrary
    per-cell annotations (e.g., '0.62*') and masks missing values (NaN) rather
    than treating them as neutral.

    Args:
        win_rate_matrix: Square matrix of win rates in [0,1] with models as index/columns.
        annotation_matrix: Optional square matrix of strings for cell annotations.
            Must align with win_rate_matrix.
        output_path: Destination image path (PDF primary; PNG also saved).
        title: Optional custom title.
        sample_df: Optional sample-level data for metadata subtitle.

    Returns:
        Path to the saved figure.
    """
    if win_rate_matrix is None or win_rate_matrix.empty:
        raise AnalysisDataError("Win-rate matrix is empty; cannot render heatmap.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    data = win_rate_matrix.copy()
    ann = annotation_matrix.copy() if annotation_matrix is not None else None

    # Translate model names in index/columns for display
    translated_index = [translate_model_name(m) for m in data.index]
    translated_cols = [translate_model_name(m) for m in data.columns]
    data.index = translated_index
    data.columns = translated_cols
    if ann is not None:
        ann = ann.copy()
        ann.index = translated_index
        ann.columns = translated_cols

    n_models = len(data)
    fig_size = max(8, n_models * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    cmap = _joint_heatmap_cmap_with_alpha()
    mask = data.isna()

    # Render with NaN masking; show masked area as light grey
    ax.set_facecolor("#f0f0f0")
    annot_fontsize = cfg.get("joint_annot_fontsize") or max(
        6, int(cfg["tick_size"] * 2)
    )
    sns.heatmap(
        data,
        ax=ax,
        annot=ann if ann is not None else False,
        fmt="",
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        cbar_kws={"label": "Win Rate", "shrink": 0.8},
        annot_kws={"fontsize": annot_fontsize, "fontweight": "bold"},
    )

    ax.set_xlabel("Opponent", fontsize=cfg["label_size"])
    ax.set_ylabel("Model", fontsize=cfg["label_size"])
    ax.tick_params(axis="both", labelsize=cfg["tick_size"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig_title = title or "Joint Pairwise Preference Matrix"
    fig_title = (
        f"{fig_title}\nCell (row, col) = win-rate of the row model vs the column model"
    )
    metadata_subtitle = build_figure_metadata_subtitle(pd.DataFrame(), sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(fig_title, fontsize=cfg["title_size"], fontweight="bold", pad=8)

    plt.tight_layout(
        pad=cfg.get("tight_layout_pad", 0.05),
        w_pad=cfg.get("tight_layout_w_pad", 0.15),
        h_pad=cfg.get("tight_layout_h_pad", 0.15),
    )
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_joint_preference_persona_panels(
    matrices: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    prompt_types: List[str],
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot a persona-level figure with one matrix per prompt type (side-by-side).

    Args:
        matrices: Mapping prompt_type -> (win_rate_matrix, annotation_matrix).
        prompt_types: Prompt type ordering to render.
        output_path: Destination path for the figure.
        title: Optional overall title.
        sample_df: Optional sample-level data for metadata subtitle.

    Returns:
        Path to saved figure.
    """
    if not prompt_types:
        raise AnalysisDataError("prompt_types is empty; cannot render persona panels.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    n_cols = len(prompt_types)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(max(10, n_cols * 6), 6),
        squeeze=False,
    )

    for idx, prompt_type in enumerate(prompt_types):
        ax = axes[0, idx]
        pair = matrices.get(prompt_type)
        if pair is None:
            ax.axis("off")
            ax.set_title(prompt_type.title(), fontsize=cfg["title_size"])
            continue
        win_rate_matrix, annotation_matrix = pair
        _render_joint_preference_heatmap_ax(
            ax=ax,
            win_rate_matrix=win_rate_matrix,
            annotation_matrix=annotation_matrix,
            cfg=cfg,
        )
        ax.set_title(prompt_type.title(), fontsize=cfg["title_size"], fontweight="bold")

    fig_title = title or "Joint Pairwise Preference Matrices"
    fig_title = (
        f"{fig_title}\nCell (row, col) = win-rate of the row model vs the column model"
    )
    metadata_subtitle = build_figure_metadata_subtitle(pd.DataFrame(), sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    fig.suptitle(fig_title, fontsize=cfg["suptitle_size"], weight="bold", y=1.02)
    fig.tight_layout(
        pad=cfg.get("tight_layout_pad", 0.05),
        w_pad=cfg.get("tight_layout_w_pad", 0.15),
        h_pad=cfg.get("tight_layout_h_pad", 0.15),
    )

    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_joint_preference_overall_grid(
    matrices: Dict[Tuple[str, str], Tuple[pd.DataFrame, pd.DataFrame]],
    personas: List[str],
    prompt_types: List[str],
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot the overall grid (prompt type rows x persona columns).

    Args:
        matrices: Mapping (persona, prompt_type) -> (win_rate_matrix, annotation_matrix).
        personas: Persona ordering (columns).
        prompt_types: Prompt type ordering (rows).
        output_path: Destination path.
        title: Optional overall title.
        sample_df: Optional sample-level data for metadata subtitle.

    Returns:
        Path to saved figure.
    """
    if not personas or not prompt_types:
        raise AnalysisDataError(
            "personas and prompt_types must be non-empty to render overall grid."
        )

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    # Transposed layout: 3 prompt types stacked per persona column.
    n_rows = len(prompt_types)
    n_cols = len(personas)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(12, n_cols * 6), max(8, n_rows * 5.5)),
        squeeze=False,
    )

    for i, prompt_type in enumerate(prompt_types):
        for j, persona in enumerate(personas):
            ax = axes[i, j]
            pair = matrices.get((persona, prompt_type))
            if pair is None:
                ax.axis("off")
                continue
            win_rate_matrix, annotation_matrix = pair
            _render_joint_preference_heatmap_ax(
                ax=ax,
                win_rate_matrix=win_rate_matrix,
                annotation_matrix=annotation_matrix,
                cfg=cfg,
            )
            # Column headers: personas
            if i == 0:
                ax.set_title(
                    translate_persona_name(str(persona)),
                    fontsize=cfg["title_size"],
                    fontweight="bold",
                )
            # Row headers: prompt types
            if j == 0:
                ax.set_ylabel(
                    prompt_type.title(),
                    fontsize=cfg["label_size"],
                    fontweight="bold",
                )

    fig_title = (
        title or "Joint Pairwise Preference Matrices (All Personas × Prompt Types)"
    )
    fig_title = (
        f"{fig_title}\nCell (row, col) = win-rate of the row model vs the column model"
    )
    metadata_subtitle = build_figure_metadata_subtitle(pd.DataFrame(), sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    fig.suptitle(fig_title, fontsize=cfg["suptitle_size"], weight="bold", y=1.01)
    fig.tight_layout(
        pad=cfg.get("tight_layout_pad", 0.05),
        w_pad=cfg.get("tight_layout_w_pad", 0.15),
        h_pad=cfg.get("tight_layout_h_pad", 0.15),
    )

    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def _render_joint_preference_heatmap_ax(
    ax: Any,
    win_rate_matrix: pd.DataFrame,
    annotation_matrix: pd.DataFrame,
    cfg: Dict[str, Any],
) -> None:
    """Render a joint preference heatmap into an existing axes."""
    data = win_rate_matrix.copy()
    ann = annotation_matrix.copy()

    translated_index = [translate_model_name(m) for m in data.index]
    translated_cols = [translate_model_name(m) for m in data.columns]
    data.index = translated_index
    data.columns = translated_cols
    ann.index = translated_index
    ann.columns = translated_cols

    cmap = _heatmap_cmap()
    mask = data.isna()
    ax.set_facecolor("#f0f0f0")
    annot_fontsize = cfg.get("joint_annot_fontsize") or max(
        6, int(cfg["tick_size"] * 2)
    )
    sns.heatmap(
        data,
        ax=ax,
        annot=ann,
        fmt="",
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        cbar=False,
        annot_kws={"fontsize": annot_fontsize, "fontweight": "bold"},
    )
    ax.set_xlabel("Opponent", fontsize=max(8, cfg["tick_size"]))
    ax.set_ylabel("Model", fontsize=max(8, cfg["tick_size"]))
    ax.tick_params(axis="both", labelsize=cfg["tick_size"] - 1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


def plot_pairwise_forest(
    pair_summary: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    statistical_tests: Optional[pd.DataFrame] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot forest plot showing win rates with confidence intervals per model pair.

    This is a publication-standard visualization for comparing multiple pairs
    with uncertainty quantification.

    Args:
        pair_summary: Output from compute_pair_summary().
        output_path: Destination image path.
        title: Optional custom title.
        statistical_tests: Optional DataFrame with CI bounds from
            compute_statistical_significance().
        sample_df: Optional sample-level data for extracting judge model info.

    Returns:
        Path to the saved figure.
    """
    if pair_summary.empty:
        raise AnalysisDataError("Pair summary is empty; cannot render forest plot.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    data = pair_summary.copy()
    data["pair_label"] = data.apply(
        lambda r: f"{translate_model_name(r['model_a_name'])} vs {translate_model_name(r['model_b_name'])}",
        axis=1,
    )

    # Merge CI bounds if available
    if statistical_tests is not None and not statistical_tests.empty:
        merge_cols = ["model_a_name", "model_b_name"]
        if all(col in statistical_tests.columns for col in merge_cols):
            data = data.merge(
                statistical_tests[
                    merge_cols + ["ci_lower", "ci_upper", "binomial_pvalue"]
                ],
                on=merge_cols,
                how="left",
            )

    # Sort by win probability
    data = data.sort_values("model_a_win_prob", ascending=True)

    n_pairs = len(data)
    fig, ax = plt.subplots(figsize=(cfg["default_width"], max(4, n_pairs * 0.6)))

    y_pos = np.arange(n_pairs)

    # Plot points
    win_probs = data["model_a_win_prob"].values

    # Plot confidence intervals if available
    if "ci_lower" in data.columns and "ci_upper" in data.columns:
        ci_lower = data["ci_lower"].fillna(0).values
        ci_upper = data["ci_upper"].fillna(1).values
        # Ensure xerr values are non-negative (clamp to 0)
        xerr_lower = np.maximum(0, win_probs - ci_lower)
        xerr_upper = np.maximum(0, ci_upper - win_probs)
        xerr = np.array([xerr_lower, xerr_upper])

        ax.errorbar(
            win_probs,
            y_pos,
            xerr=xerr,
            fmt="o",
            color=PAIRWISE_PALETTE["model_a"],
            markersize=10,
            capsize=5,
            capthick=2,
            elinewidth=2,
            markeredgecolor="white",
            markeredgewidth=1,
        )
    else:
        ax.scatter(
            win_probs,
            y_pos,
            s=100,
            color=PAIRWISE_PALETTE["model_a"],
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

    # Add significance markers if available
    if "binomial_pvalue" in data.columns:
        for i, (y, p) in enumerate(zip(y_pos, data["binomial_pvalue"].values)):
            if pd.notna(p):
                if p < 0.01:
                    marker = "**"
                elif p < 0.05:
                    marker = "*"
                else:
                    marker = ""
                if marker:
                    ax.text(
                        win_probs[i] + 0.02,
                        y,
                        marker,
                        fontsize=cfg["tick_size"],
                        fontweight="bold",
                        va="center",
                    )

    # Reference line at 0.5
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["pair_label"], fontsize=cfg["tick_size"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Model A Win Probability (excl. ties)", fontsize=cfg["label_size"])

    ax.grid(axis="x", linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    fig_title = title or "Pairwise Comparison Forest Plot"
    metadata_subtitle = build_figure_metadata_subtitle(data, sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(fig_title, fontsize=cfg["title_size"], fontweight="bold", pad=15)

    # Add legend for significance
    if "binomial_pvalue" in data.columns:
        ax.text(
            0.02,
            0.98,
            "* p < 0.05, ** p < 0.01",
            transform=ax.transAxes,
            fontsize=cfg["tick_size"] - 2,
            verticalalignment="top",
            style="italic",
        )

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_pairwise_objective_passk(
    pair_summary: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot objective pass@k win/tie rates for a model pair.

    Expects columns generated by compute_pair_summary when objective metrics are supplied:
    obj_pass_at_{k}_model_a_win_rate, obj_pass_at_{k}_tie_rate, obj_pass_at_{k}_model_b_win_rate.
    """
    if pair_summary.empty:
        raise AnalysisDataError(
            "Pair summary is empty; cannot render objective pass@k plot."
        )

    # Limit to a single model pair
    model_pairs = (
        pair_summary["model_pair"].unique()
        if "model_pair" in pair_summary.columns
        else []
    )
    if len(model_pairs) > 1:
        raise AnalysisDataError(
            "plot_pairwise_objective_passk expects a single model pair."
        )

    row = pair_summary.iloc[0]
    model_a = translate_model_name(row.get("model_a_name", "Model A"))
    model_b = translate_model_name(row.get("model_b_name", "Model B"))

    records: List[Dict[str, Any]] = []
    for k in (1, 5):
        comparisons = row.get(f"obj_pass_at_{k}_comparisons")
        a_rate = row.get(f"obj_pass_at_{k}_model_a_win_rate")
        tie_rate = row.get(f"obj_pass_at_{k}_tie_rate")
        b_rate = row.get(f"obj_pass_at_{k}_model_b_win_rate")

        if comparisons is None or pd.isna(comparisons) or comparisons <= 0:
            continue
        if any(pd.isna(val) for val in (a_rate, tie_rate, b_rate)):
            continue

        records.append(
            {
                "metric": f"Pass@{k}",
                "model_a_win_rate": float(a_rate),
                "tie_rate": float(tie_rate),
                "model_b_win_rate": float(b_rate),
            }
        )

    if not records:
        raise AnalysisDataError("No objective pass@k metrics available to plot.")

    data = pd.DataFrame(records)

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    fig, ax = plt.subplots(figsize=(12, max(3.5, len(data) * 1.2)))
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(data))
    bar_height = 0.6

    a_wins = data["model_a_win_rate"].values
    ties = data["tie_rate"].values
    b_wins = data["model_b_win_rate"].values

    ax.barh(
        y_pos,
        a_wins,
        bar_height,
        label=model_a,
        color=PAIRWISE_PALETTE["model_a"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.barh(
        y_pos,
        ties,
        bar_height,
        left=a_wins,
        label="Tie",
        color=PAIRWISE_PALETTE["tie"],
        alpha=0.70,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.barh(
        y_pos,
        b_wins,
        bar_height,
        left=a_wins + ties,
        label=model_b,
        color=PAIRWISE_PALETTE["model_b"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )

    for i, (a, t, b) in enumerate(zip(a_wins, ties, b_wins)):
        if a >= 0.08:
            ax.text(
                a / 2,
                i,
                f"{a:.0%}",
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                fontweight="bold",
            )
        if t >= 0.08:
            ax.text(
                a + t / 2,
                i,
                f"{t:.0%}",
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="medium",
            )
        if b >= 0.08:
            ax.text(
                a + t + b / 2,
                i,
                f"{b:.0%}",
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        data["metric"].values,
        fontsize=cfg["tick_size"],
        fontweight="medium",
        color=MODERN_COLORS["text"],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        "Win Rate Distribution", fontsize=cfg["label_size"], color=MODERN_COLORS["text"]
    )

    ax.axvline(
        0.5, color=MODERN_COLORS["text_light"], linestyle="--", linewidth=1.5, alpha=0.5
    )
    _apply_modern_style(ax, cfg)

    legend = ax.legend(
        loc="lower right",
        fontsize=cfg["legend_size"] - 1,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor=MODERN_COLORS["grid"],
        ncol=3,
    )
    legend.get_frame().set_linewidth(0.6)

    fig_title = title or "Objective Pass@k Win Rates"
    metadata_subtitle = build_figure_metadata_subtitle(pair_summary, sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(
        fig_title,
        fontsize=cfg["title_size"],
        fontweight="bold",
        color=MODERN_COLORS["text"],
        pad=16,
    )

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path


def plot_model_ranking(
    model_rankings: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Plot horizontal bar chart showing aggregate model rankings from pairwise comparisons.

    Args:
        model_rankings: Output from compute_model_rankings().
        output_path: Destination image path.
        title: Optional custom title.
        sample_df: Optional sample-level data for extracting judge model info.

    Returns:
        Path to the saved figure.
    """
    if model_rankings.empty:
        raise AnalysisDataError("Model rankings is empty; cannot render plot.")

    _apply_acl_style()
    cfg = FIGURE_CONFIG

    data = model_rankings.copy()
    # Translate model names
    data["model_name"] = data["model_name"].apply(translate_model_name)
    data = data.sort_values("avg_win_rate", ascending=True)

    n_models = len(data)
    fig, ax = plt.subplots(figsize=(cfg["default_width"], max(4, n_models * 0.6)))

    y_pos = np.arange(n_models)
    colors = sns.color_palette("viridis", n_colors=n_models)

    bars = ax.barh(
        y_pos,
        data["avg_win_rate"].values,
        color=colors,
        edgecolor=cfg["bar_edgecolor"],
        linewidth=cfg["bar_linewidth"],
    )

    # Add min/max range as error bars
    if "min_win_rate" in data.columns and "max_win_rate" in data.columns:
        min_rates = data["min_win_rate"].values
        max_rates = data["max_win_rate"].values
        avg_rates = data["avg_win_rate"].values

        for i, (y, avg, mn, mx) in enumerate(
            zip(y_pos, avg_rates, min_rates, max_rates)
        ):
            ax.plot([mn, mx], [y, y], color="#333333", linewidth=2, zorder=1)
            ax.scatter([mn, mx], [y, y], color="#333333", s=30, zorder=2)

    # Add value labels
    for i, (rect, rate) in enumerate(zip(bars, data["avg_win_rate"].values)):
        width = rect.get_width()
        ax.text(
            width + 0.02,
            rect.get_y() + rect.get_height() / 2,
            f"{rate:.1%}",
            ha="left",
            va="center",
            fontsize=cfg["tick_size"] - 2,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["model_name"], fontsize=cfg["tick_size"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average Win Rate", fontsize=cfg["label_size"])

    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    ax.grid(axis="x", linestyle=cfg["grid_linestyle"], alpha=cfg["grid_alpha"])

    fig_title = title or "Model Rankings from Pairwise Comparisons"
    # Model rankings don't contain metadata, so only check sample_df
    metadata_subtitle = build_figure_metadata_subtitle(pd.DataFrame(), sample_df)
    if metadata_subtitle:
        fig_title = f"{fig_title}\n{metadata_subtitle}"
    ax.set_title(fig_title, fontsize=cfg["title_size"], fontweight="bold", pad=15)

    plt.tight_layout()
    path = _save_figure(fig, Path(output_path))
    plt.close(fig)
    return path
