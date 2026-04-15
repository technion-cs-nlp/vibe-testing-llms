import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# =============================================================================
# ACL PUBLICATION-QUALITY STYLING CONFIGURATION
# =============================================================================

# Configure ACL-style with EXTRA LARGE fonts for publication readability
FIGURE_CONFIG = {
    "font_family": "serif",
    "font_serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font_scale": 2.2,  # Extra large font scale
    "title_size": 24,  # Extra large title
    "label_size": 20,  # Extra large labels
    "tick_size": 18,  # Extra large ticks
    "legend_size": 18,  # Extra large legend
    "legend_title_size": 19,
    "dpi": 300,
    "grid_alpha": 0.5,
    "grid_linestyle": "--",
    "bar_edgecolor": "black",
    "bar_linewidth": 1.0,
}

def apply_acl_style():
    """Configure publication-friendly seaborn style for ACL papers with extra large fonts."""
    cfg = FIGURE_CONFIG
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=cfg["font_scale"])
    
    plt.rcParams.update({
        "font.family": cfg["font_family"],
        "font.serif": cfg["font_serif"],
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

def save_figure(fig, output_path, save_pdf=True, save_png=True):
    """Save figure in both PDF (vector) and PNG (raster) formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=FIGURE_CONFIG["dpi"])
        print(f"Saved PDF: {pdf_path}")
    
    if save_png:
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=FIGURE_CONFIG["dpi"])
        print(f"Saved PNG: {png_path}")

# Apply ACL styling globally
apply_acl_style()

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

# Reload and process data
df = pd.read_csv('examples_annotations/survey/Vibe-Testing AI Systems.csv')

new_columns = {
    'How often do you use AI tools (e.g., ChatGPT, Claude, Gemini, Copilot)?': 'usage_freq',
    'How would you describe your technical background?': 'tech_background',
    'When using a new AI model, how often do you experiment with prompts or tasks just to see how it behaves?': 'experiment_freq',
    'Which of the following do you typically do when vibe-testing?': 'vibe_methods',
    'Which aspects do you feel benchmarks fail to measure well? (Select all that apply)': 'benchmark_failures',
}
df.rename(columns=new_columns, inplace=True)

# Helper to process multi-select responses
def get_counts(df, col):
    """Extract and count multi-select responses."""
    series = df[col].dropna()
    all_items = []
    for entry in series:
        items = [item.strip() for item in entry.split(';')]
        all_items.extend(items)
    return pd.Series(all_items).value_counts().sort_values(ascending=True)

# =============================================================================
# FIGURE 1: What Benchmarks Fail to Measure
# =============================================================================

fig1, ax1 = plt.subplots(figsize=(14, 8))
failures_counts = get_counts(df, 'benchmark_failures')

# Clean and shorten labels for clarity
label_mapping = {
    'Handling ambiguity / underspecification': 'Handling Ambiguity',
    'Fit for my workflow': 'Fit for Personal Workflow',
    'Style, tone, or personality': 'Style/Tone/Personality',
    'Clarity and readability': 'Clarity',
    'Trustworthiness / safety': 'Safety/Trust',
    'Stability and consistency': 'Stability'
}
failures_counts.index = [label_mapping.get(x, x) for x in failures_counts.index]

# Create horizontal bar chart with modern styling
bars = ax1.barh(
    failures_counts.index, 
    failures_counts.values, 
    color='#2563EB',  # Modern vibrant blue
    alpha=0.85,
    edgecolor='white',
    linewidth=FIGURE_CONFIG["bar_linewidth"]
)

# Add value labels on bars
for i, (idx, val) in enumerate(zip(failures_counts.index, failures_counts.values)):
    ax1.text(
        val + 0.3, i, str(int(val)), 
        va='center', 
        fontsize=FIGURE_CONFIG["tick_size"],
        fontweight='bold',
        color='#1F2937'
    )

ax1.set_xlabel('Number of Respondents', fontsize=FIGURE_CONFIG["label_size"], fontweight='bold')
ax1.set_title('What Benchmarks Fail to Measure', fontsize=FIGURE_CONFIG["title_size"], fontweight='bold', pad=25)
ax1.grid(axis='x', linestyle=FIGURE_CONFIG["grid_linestyle"], alpha=FIGURE_CONFIG["grid_alpha"], zorder=0)
ax1.set_axisbelow(True)

# Clean up spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#E5E7EB')
ax1.spines['bottom'].set_color('#E5E7EB')

plt.tight_layout()
save_figure(fig1, 'figure_1_benchmark_failures')
plt.close(fig1)

# =============================================================================
# FIGURE 2: Common Vibe-Testing Methods
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(14, 8))
methods_counts = get_counts(df, 'vibe_methods')

# Shorten labels for readability
short_labels = {
    'Compare outputs from different models': 'Compare Models',
    'Try tasks from my own workflow': 'Test Workflow Tasks',
    'Check tone, style, or personality': 'Check Tone/Style',
    'Use a small set of personal "test prompts"': 'Use Personal "Test Prompts"',
    'Give vague or underspecified instructions to see how it handles ambiguity': 'Test Ambiguity Handling',
    'Re-run the same prompt to check stability or consistency': 'Check Stability'
}
methods_counts.index = [short_labels.get(x, x) for x in methods_counts.index]
methods_counts = methods_counts[methods_counts >= 2]  # Filter rare responses

# Create horizontal bar chart
bars2 = ax2.barh(
    methods_counts.index, 
    methods_counts.values, 
    color='#059669',  # Modern emerald green
    alpha=0.85,
    edgecolor='white',
    linewidth=FIGURE_CONFIG["bar_linewidth"]
)

# Add value labels
for i, (idx, val) in enumerate(zip(methods_counts.index, methods_counts.values)):
    ax2.text(
        val + 0.3, i, str(int(val)), 
        va='center', 
        fontsize=FIGURE_CONFIG["tick_size"],
        fontweight='bold',
        color='#1F2937'
    )

ax2.set_xlabel('Number of Respondents', fontsize=FIGURE_CONFIG["label_size"], fontweight='bold')
ax2.set_title('Common Vibe-Testing Methods', fontsize=FIGURE_CONFIG["title_size"], fontweight='bold', pad=25)
ax2.grid(axis='x', linestyle=FIGURE_CONFIG["grid_linestyle"], alpha=FIGURE_CONFIG["grid_alpha"], zorder=0)
ax2.set_axisbelow(True)

# Clean up spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#E5E7EB')
ax2.spines['bottom'].set_color('#E5E7EB')

plt.tight_layout()
save_figure(fig2, 'figure_2_vibe_methods')
plt.close(fig2)

# =============================================================================
# FIGURE 3: Experimentation Frequency by Expertise
# =============================================================================

# Simplify tech background labels
df['simple_tech'] = df['tech_background'].apply(
    lambda x: 'AI/ML Expert' if 'Expert' in str(x) 
    else ('Technical (Gen)' if 'Technical' in str(x) else 'Other')
)

# Filter to main groups
plot_df = df[df['simple_tech'].isin(['AI/ML Expert', 'Technical (Gen)'])].copy()

fig3, ax3 = plt.subplots(figsize=(10, 8))

# Create boxplot with swarm overlay using seaborn
sns.boxplot(
    x='simple_tech', 
    y='experiment_freq', 
    data=plot_df, 
    palette=['#DC2626', '#7C3AED'],  # Modern red and purple
    width=0.6,
    linewidth=FIGURE_CONFIG["bar_linewidth"],
    ax=ax3
)

sns.swarmplot(
    x='simple_tech', 
    y='experiment_freq', 
    data=plot_df, 
    color='#1F2937',  # Dark gray for points
    size=8, 
    alpha=0.6,
    ax=ax3
)

ax3.set_ylabel('Frequency (1=Never, 7=Always)', fontsize=FIGURE_CONFIG["label_size"], fontweight='bold')
ax3.set_xlabel('', fontsize=FIGURE_CONFIG["label_size"])
ax3.set_title('Experimentation Frequency by Expertise', fontsize=FIGURE_CONFIG["title_size"], fontweight='bold', pad=25)
ax3.set_ylim(1, 7.5)
ax3.grid(axis='y', linestyle=FIGURE_CONFIG["grid_linestyle"], alpha=FIGURE_CONFIG["grid_alpha"], zorder=0)
ax3.set_axisbelow(True)

# Clean up spines
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color('#E5E7EB')
ax3.spines['bottom'].set_color('#E5E7EB')

# Set tick labels to be extra large
ax3.tick_params(axis='both', labelsize=FIGURE_CONFIG["tick_size"])

plt.tight_layout()
save_figure(fig3, 'figure_3_expertise_experimentation')
plt.close(fig3)

print("\n✓ All figures generated successfully with ACL publication-quality styling!")
print("  - Extra large fonts for readability")
print("  - Professional serif typography")
print("  - High-resolution output (300 DPI)")
print("  - Saved in both PDF (vector) and PNG (raster) formats")

