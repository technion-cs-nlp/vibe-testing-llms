import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# ACL PUBLICATION-QUALITY STYLING CONFIGURATION
# =============================================================================

# Configure ACL-style with EXTRA LARGE fonts for publication readability
FIGURE_CONFIG = {
    "font_scale": 4,
    "title_size": 26,
    "label_size": 24,
    "tick_size": 20,
    "legend_size": 20,
    "legend_title_size": 20,
    "dpi": 300,
    "grid_alpha": 0.5,
    "grid_linestyle": "--",
    "bar_edgecolor": "black",
    "bar_linewidth": 1.0,
}

def apply_acl_style() -> None:
    """
    Configure publication-friendly seaborn style for ACL papers with extra large fonts.
    """
    cfg = FIGURE_CONFIG
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=cfg["font_scale"])
    
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": cfg["title_size"],
        "axes.labelsize": cfg["label_size"],
        "xtick.labelsize": cfg["tick_size"],
        "ytick.labelsize": cfg["tick_size"],
        "legend.fontsize": cfg["legend_size"],
        "legend.title_fontsize": cfg["legend_title_size"],
        "figure.dpi": cfg["dpi"],
        "pdf.fonttype": 42,  # Editable fonts in PDF
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "axes.grid": True,
        "grid.alpha": cfg["grid_alpha"],
        "grid.linestyle": cfg["grid_linestyle"],
    })

def save_figure(fig: plt.Figure, output_path: str) -> None:
    """
    Save figure as PDF file.

    Args:
        fig (plt.Figure): The matplotlib figure object.
        output_path (str): Path to save the figure (without extension).
    """
    output_path = Path(output_path)
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", dpi=FIGURE_CONFIG["dpi"])
    logger.info(f"Saved PDF: {pdf_path}")

def get_counts(df: pd.DataFrame, col: str) -> tuple[pd.Series, int]:
    """
    Extract and count multi-select responses.

    Args:
        df (pd.DataFrame): The survey data.
        col (str): The column name to process.

    Returns:
        tuple[pd.Series, int]: A tuple containing the counts of each item and 
                              the total number of respondents for this column.
    """
    logger.info(f"Processing multi-select responses for column: {col}")
    series = df[col].dropna()
    total_respondents = len(series)
    all_items = []
    for entry in series:
        items = [item.strip() for item in entry.split(';')]
        all_items.extend(items)
    counts = pd.Series(all_items).value_counts().sort_values(ascending=True)
    return counts, total_respondents

def normalize_label(label: str, short_labels: dict[str, str]) -> str:
    """
    Normalize label by handling quote variations and mapping to short labels.

    Args:
        label (str): The original label.
        short_labels (dict[str, str]): Mapping from original/normalized labels to short ones.

    Returns:
        str: The normalized and mapped label.
    """
    # Replace curly quotes (Unicode) with straight quotes for matching
    # " (U+201C) -> " and " (U+201D) -> "
    normalized = label.replace('\u201c', '"').replace('\u201d', '"')
    # Try normalized version first, then original, then return original if no match
    return short_labels.get(normalized, short_labels.get(label, label))

