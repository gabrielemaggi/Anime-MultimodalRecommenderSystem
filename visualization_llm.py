import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

RESULTS_DIR = Path("./evaluation_results")
PLOTS_DIR = RESULTS_DIR / "plots"

# Color scheme
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "success": "#06A77D",
    "warning": "#F77F00",
    "danger": "#D62828",
    "neutral": "#6C757D",
}

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging():
    """Configure logging for visualization"""
    logger = logging.getLogger("VisualizationScript")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()

# ============================================================================
# DATA LOADING
# ============================================================================


def find_latest_files():
    """Find the most recent statistics and detailed results files"""
    stats_files = list(RESULTS_DIR.glob("statistics_*.json"))
    detailed_files = list(RESULTS_DIR.glob("detailed_results_*.json"))

    if not stats_files:
        raise FileNotFoundError("No statistics files found in evaluation_results/")

    latest_stats = max(stats_files, key=lambda p: p.stat().st_mtime)
    latest_detailed = (
        max(detailed_files, key=lambda p: p.stat().st_mtime) if detailed_files else None
    )

    logger.info(f"Found statistics file: {latest_stats.name}")
    if latest_detailed:
        logger.info(f"Found detailed results file: {latest_detailed.name}")

    return latest_stats, latest_detailed


def load_data():
    """Load statistics and detailed results"""
    stats_file, detailed_file = find_latest_files()

    with open(stats_file, "r") as f:
        statistics = json.load(f)

    detailed_results = None
    if detailed_file:
        with open(detailed_file, "r") as f:
            detailed_results = json.load(f)

    return statistics, detailed_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_model_comparison(statistics, save_path):
    """Bar chart comparing mean scores across models with error bars"""
    logger.info("Creating model comparison plot...")

    models = list(statistics.keys())
    means = [statistics[m]["score_mean"] for m in models]
    stds = [statistics[m]["score_std"] for m in models]

    # Sort by mean score
    sorted_indices = np.argsort(means)[::-1]
    models = [models[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars with gradient colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.bar(
        models,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.05,
            f"{mean:.2f}±{std:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_ylabel("Mean Score (out of 5)", fontweight="bold", fontsize=12)
    ax.set_title(
        "Model Performance Comparison\n(Mean Score ± Standard Deviation)",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color="red", linestyle="--", alpha=0.5, label="Baseline (3.0)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


def plot_score_distribution_violin(statistics, save_path):
    """Violin plot showing score distributions for each model"""
    logger.info("Creating score distribution violin plot...")

    # Prepare data
    data = []
    labels = []

    for model_name, stats in statistics.items():
        scores = stats.get("all_scores", [])
        data.extend(scores)
        labels.extend([model_name] * len(scores))

    if not data:
        logger.warning("  ⚠ No score data available for violin plot")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create violin plot
    models = list(statistics.keys())
    score_data = [statistics[m].get("all_scores", []) for m in models]

    parts = ax.violinplot(
        score_data,
        positions=range(len(models)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    # Customize violin colors
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plt.cm.viridis(i / len(models)))
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Customize mean and median lines
    parts["cmeans"].set_edgecolor("red")
    parts["cmeans"].set_linewidth(2)
    parts["cmedians"].set_edgecolor("blue")
    parts["cmedians"].set_linewidth(2)

    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_ylabel("Score Distribution", fontweight="bold", fontsize=12)
    ax.set_title(
        "Score Distribution Across Models\n(Violin Plot)",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", alpha=0.3)

    # Add legend
    red_patch = mpatches.Patch(color="red", label="Mean")
    blue_patch = mpatches.Patch(color="blue", label="Median")
    ax.legend(handles=[red_patch, blue_patch])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


def plot_score_distribution_histogram(statistics, save_path):
    """Histogram showing score distribution for each model"""
    logger.info("Creating score distribution histograms...")

    models = list(statistics.keys())
    n_models = len(models)

    # Calculate grid dimensions
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, stats) in enumerate(statistics.items()):
        ax = axes[idx]
        scores = stats.get("all_scores", [])

        if scores:
            # Create histogram
            counts, bins, patches = ax.hist(
                scores,
                bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                edgecolor="black",
                alpha=0.7,
                color=plt.cm.viridis(idx / n_models),
            )

            # Color bars by score
            colors_map = {
                1: "#D62828",
                2: "#F77F00",
                3: "#FCBF49",
                4: "#06A77D",
                5: "#06A77D",
            }
            for i, patch in enumerate(patches):
                score_value = int(bins[i] + 0.5)
                patch.set_facecolor(colors_map.get(score_value, "#6C757D"))

            # Add value labels on bars
            for i, count in enumerate(counts):
                if count > 0:
                    ax.text(
                        bins[i] + 0.5,
                        count,
                        str(int(count)),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            ax.set_title(
                f"{model_name}\n(Mean: {stats['score_mean']:.2f}, Std: {stats['score_std']:.2f})",
                fontweight="bold",
            )
            ax.set_xlabel("Score", fontweight="bold")
            ax.set_ylabel("Frequency", fontweight="bold")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(model_name, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Score Distribution by Model", fontweight="bold", fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


def plot_consistency_analysis(statistics, save_path):
    """Scatter plot: Mean vs Std Dev (consistency analysis)"""
    logger.info("Creating consistency analysis plot...")

    models = list(statistics.keys())
    means = [statistics[m]["score_mean"] for m in models]
    stds = [statistics[m]["score_std"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    scatter = ax.scatter(
        means, stds, s=300, c=colors, alpha=0.7, edgecolors="black", linewidth=2
    )

    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (means[i], stds[i]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                alpha=0.7,
            ),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0", color="black", lw=1
            ),
        )

    # Add quadrant lines
    mean_threshold = np.mean(means)
    std_threshold = np.mean(stds)

    ax.axvline(x=mean_threshold, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=std_threshold, color="gray", linestyle="--", alpha=0.5)

    # Add quadrant labels
    ax.text(
        mean_threshold + 0.1,
        max(stds) - 0.05,
        "High Mean\nHigh Variance",
        ha="left",
        va="top",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )
    ax.text(
        mean_threshold + 0.1,
        min(stds) + 0.05,
        "High Mean\nLow Variance\n(IDEAL)",
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic",
        alpha=0.6,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3),
    )
    ax.text(
        mean_threshold - 0.1,
        max(stds) - 0.05,
        "Low Mean\nHigh Variance",
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )
    ax.text(
        mean_threshold - 0.1,
        min(stds) + 0.05,
        "Low Mean\nLow Variance",
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )

    ax.set_xlabel("Mean Score", fontweight="bold", fontsize=12)
    ax.set_ylabel("Standard Deviation (Consistency)", fontweight="bold", fontsize=12)
    ax.set_title(
        "Model Performance vs Consistency\n(Lower std = more consistent)",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


def plot_success_rate(statistics, save_path):
    """Bar chart showing success rate and errors for each model"""
    logger.info("Creating success rate plot...")

    models = list(statistics.keys())
    successful = [statistics[m]["successful_evaluations"] for m in models]
    errors = [statistics[m]["errors"] for m in models]
    total = [statistics[m]["total_evaluations"] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    # Create stacked bars
    bars1 = ax.bar(
        x,
        successful,
        width,
        label="Successful",
        color=COLORS["success"],
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x,
        errors,
        width,
        bottom=successful,
        label="Errors",
        color=COLORS["danger"],
        edgecolor="black",
        linewidth=1.5,
    )

    # Add percentage labels
    for i, (s, e, t) in enumerate(zip(successful, errors, total)):
        success_pct = (s / t * 100) if t > 0 else 0
        ax.text(
            i,
            s / 2,
            f"{success_pct:.1f}%",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )
        if e > 0:
            error_pct = (e / t * 100) if t > 0 else 0
            ax.text(
                i,
                s + e / 2,
                f"{error_pct:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
            )

    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_ylabel("Number of Evaluations", fontweight="bold", fontsize=12)
    ax.set_title(
        "Evaluation Success Rate by Model", fontweight="bold", fontsize=14, pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


def plot_genre_goal_comparison(statistics, save_path):
    """Compare overall scores vs genre goal scores"""
    logger.info("Creating genre goal comparison plot...")

    models = []
    overall_scores = []
    genre_scores = []

    for model_name, stats in statistics.items():
        if stats.get("genre_goal_score_mean") is not None:
            models.append(model_name)
            overall_scores.append(stats["score_mean"])
            genre_scores.append(stats["genre_goal_score_mean"])

    if not models:
        logger.warning("  ⚠ No genre goal data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        overall_scores,
        width,
        label="Overall Score",
        color=COLORS["primary"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        genre_scores,
        width,
        label="Genre Goal Score",
        color=COLORS["tertiary"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_ylabel("Mean Score", fontweight="bold", fontsize=12)
    ax.set_title(
        "Overall Performance vs Genre Goal Alignment",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


def plot_detailed_heatmap(detailed_results, save_path):
    """Heatmap showing scores by user and model"""
    logger.info("Creating detailed heatmap...")

    if not detailed_results:
        logger.warning("  ⚠ No detailed results available for heatmap")
        return

    # Extract data
    user_model_scores = {}

    for result in detailed_results:
        if "error" in result:
            continue

        username = result["username"]
        model = result["model"]
        score = result.get("evaluation", {}).get("score", 0)

        if username not in user_model_scores:
            user_model_scores[username] = {}

        if model not in user_model_scores[username]:
            user_model_scores[username][model] = []

        user_model_scores[username][model].append(score)

    # Create matrix
    users = sorted(user_model_scores.keys())
    models = sorted(set(m for u in user_model_scores.values() for m in u.keys()))

    matrix = np.zeros((len(users), len(models)))

    for i, user in enumerate(users):
        for j, model in enumerate(models):
            scores = user_model_scores.get(user, {}).get(model, [])
            matrix[i, j] = np.mean(scores) if scores else 0

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(users) * 0.5)))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=5)

    # Set ticks
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(users)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(users)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Score", rotation=270, labelpad=20, fontweight="bold")

    # Add text annotations
    for i in range(len(users)):
        for j in range(len(models)):
            value = matrix[i, j]
            if value > 0:
                text = ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="black" if value < 2.5 or value > 4.5 else "white",
                    fontweight="bold",
                    fontsize=9,
                )

    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_ylabel("User", fontweight="bold", fontsize=12)
    ax.set_title(
        "Mean Scores: Users × Models Heatmap", fontweight="bold", fontsize=14, pad=20
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


def plot_comprehensive_dashboard(statistics, save_path):
    """Create a comprehensive dashboard with multiple metrics"""
    logger.info("Creating comprehensive dashboard...")

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    models = list(statistics.keys())

    # 1. Mean scores bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    means = [statistics[m]["score_mean"] for m in models]
    stds = [statistics[m]["score_std"] for m in models]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    ax1.bar(
        models,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax1.set_title("Mean Scores by Model", fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 5.5)
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 2. Statistics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("tight")
    ax2.axis("off")

    table_data = [["Model", "Mean", "Std", "Min", "Max"]]
    for model in models:
        stats = statistics[model]
        table_data.append(
            [
                model[:15] + "..." if len(model) > 15 else model,
                f"{stats['score_mean']:.2f}",
                f"{stats['score_std']:.2f}",
                f"{stats['score_min']:.1f}",
                f"{stats['score_max']:.1f}",
            ]
        )

    table = ax2.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax2.set_title("Performance Summary", fontweight="bold", pad=20)

    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, :])
    score_data = [statistics[m].get("all_scores", []) for m in models]
    bp = ax3.boxplot(score_data, labels=models, patch_artist=True)

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(plt.cm.viridis(i / len(models)))
        box.set_alpha(0.7)

    ax3.set_title("Score Distribution (Box Plot)", fontweight="bold")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 5.5)
    ax3.grid(axis="y", alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 4. Success rate
    ax4 = fig.add_subplot(gs[2, 0])
    successful = [statistics[m]["successful_evaluations"] for m in models]
    total = [statistics[m]["total_evaluations"] for m in models]
    success_rates = [s / t * 100 if t > 0 else 0 for s, t in zip(successful, total)]

    ax4.bar(
        models,
        success_rates,
        color=COLORS["success"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax4.set_title("Success Rate", fontweight="bold")
    ax4.set_ylabel("Success %")
    ax4.set_ylim(0, 110)
    ax4.grid(axis="y", alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 5. Consistency (std dev)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(
        models,
        stds,
        color=COLORS["warning"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax5.set_title("Consistency (Lower = Better)", fontweight="bold")
    ax5.set_ylabel("Std Deviation")
    ax5.grid(axis="y", alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 6. Min/Max range
    ax6 = fig.add_subplot(gs[2, 2])
    mins = [statistics[m]["score_min"] for m in models]
    maxs = [statistics[m]["score_max"] for m in models]
    ranges = [ma - mi for mi, ma in zip(mins, maxs)]

    ax6.bar(
        models,
        ranges,
        color=COLORS["secondary"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax6.set_title("Score Range (Max - Min)", fontweight="bold")
    ax6.set_ylabel("Range")
    ax6.grid(axis="y", alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle(
        "Evaluation Dashboard - Comprehensive Overview",
        fontweight="bold",
        fontsize=18,
        y=0.98,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def generate_all_plots():
    """Generate all visualization plots"""
    logger.info("=" * 80)
    logger.info("STARTING VISUALIZATION GENERATION")
    logger.info("=" * 80)

    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    logger.info("\nLoading data...")
    try:
        statistics, detailed_results = load_data()
        logger.info(f"✓ Loaded statistics for {len(statistics)} models")
        if detailed_results:
            logger.info(f"✓ Loaded {len(detailed_results)} detailed results")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Generate plots
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 80 + "\n")

    plots = [
        (
            "model_comparison.png",
            lambda: plot_model_comparison(
                statistics, PLOTS_DIR / f"model_comparison_{timestamp}.png"
            ),
        ),
        (
            "score_distribution_violin.png",
            lambda: plot_score_distribution_violin(
                statistics, PLOTS_DIR / f"score_distribution_violin_{timestamp}.png"
            ),
        ),
        (
            "score_distribution_histogram.png",
            lambda: plot_score_distribution_histogram(
                statistics, PLOTS_DIR / f"score_distribution_histogram_{timestamp}.png"
            ),
        ),
        (
            "consistency_analysis.png",
            lambda: plot_consistency_analysis(
                statistics, PLOTS_DIR / f"consistency_analysis_{timestamp}.png"
            ),
        ),
        (
            "success_rate.png",
            lambda: plot_success_rate(
                statistics, PLOTS_DIR / f"success_rate_{timestamp}.png"
            ),
        ),
        (
            "genre_goal_comparison.png",
            lambda: plot_genre_goal_comparison(
                statistics, PLOTS_DIR / f"genre_goal_comparison_{timestamp}.png"
            ),
        ),
        (
            "comprehensive_dashboard.png",
            lambda: plot_comprehensive_dashboard(
                statistics, PLOTS_DIR / f"comprehensive_dashboard_{timestamp}.png"
            ),
        ),
    ]

    for plot_name, plot_func in plots:
        try:
            plot_func()
        except Exception as e:
            logger.error(f"  ✗ Failed to generate {plot_name}: {e}")

    # Generate heatmap if detailed results available
    if detailed_results:
        try:
            plot_detailed_heatmap(
                detailed_results, PLOTS_DIR / f"user_model_heatmap_{timestamp}.png"
            )
        except Exception as e:
            logger.error(f"  ✗ Failed to generate heatmap: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✓ All plots saved to: {PLOTS_DIR}")
    logger.info(f"  Total plots generated: Look in the plots directory")

    return PLOTS_DIR


if __name__ == "__main__":
    print("\n" + "📊" * 40)
    print("EVALUATION STATISTICS VISUALIZATION")
    print("📊" * 40 + "\n")

    try:
        plots_dir = generate_all_plots()
        print(f"\n✅ Visualization completed successfully!")
        print(f"📁 Plots saved in: {plots_dir}")
        print("\nGenerated plots:")
        print("  1. Model Comparison (Bar chart with error bars)")
        print("  2. Score Distribution (Violin plot)")
        print("  3. Score Histograms (Per-model distributions)")
        print("  4. Consistency Analysis (Mean vs Std scatter)")
        print("  5. Success Rate (Stacked bar chart)")
        print("  6. Genre Goal Comparison (If available)")
        print("  7. User×Model Heatmap (If detailed results available)")
        print("  8. Comprehensive Dashboard (All-in-one overview)")

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        print(f"\n❌ Visualization failed! Check the error above.")
