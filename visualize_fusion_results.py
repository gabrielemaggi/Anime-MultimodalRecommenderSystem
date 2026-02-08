"""
Visualization script for fusion strategy evaluation results.

Generates comparison charts and analysis plots from evaluation JSON.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class FusionResultsVisualizer:
    """Visualize fusion strategy evaluation results."""

    def __init__(self, results_path: str = "fusion_strategy_evaluation.json"):
        self.results_path = results_path
        self.results = self._load_results()
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

    def _load_results(self) -> Dict:
        if not Path(self.results_path).exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        with open(self.results_path, "r") as f:
            return json.load(f)

    def _remap_strategy_name(self, name: str) -> str:
        """
        Remaps weight-based strategy names to descriptive labels.
        Example: '1.0, 0, 0' -> 'Only Synopsis'
        """
        mapping = {
            "weighted (Classical) [1.0, 0, 0]": "Only Synopsis",
            "weighted (Classical) [0, 1.0, 0]": "Only Visual",
            "weighted (Classical) [0, 0, 1.0]": "Only Tabular",
        }

        # Strip brackets or parentheses if present in your JSON keys
        clean_name = name.strip()
        print(clean_name)

        return mapping.get(clean_name, name)

    def plot_metric_comparison(self, metrics: List[str] = None, save: bool = True):
        if metrics is None:
            # Removed ndcg@5 per request
            metrics = [
                "genre_overlap_avg",
                "ndcg@10",
                "diversity",
                "precision@5",
            ]

        plot_data = []
        for strategy_raw, results in self.results.items():
            print(strategy_raw)
            strategy_name = self._remap_strategy_name(strategy_raw)
            for metric in metrics:
                val = results["metrics"].get(metric, {}).get("mean", 0)
                # Per user instructions: 0 is not bad, just not scored.
                # We skip plotting 0 to keep the log scale from breaking.
                if val <= 0:
                    continue

                plot_data.append(
                    {
                        "Strategy": strategy_name,
                        "Metric": metric.replace("_", " ").title().replace("@", " @"),
                        "Score": val,
                    }
                )

        df = pd.DataFrame(plot_data)
        plt.figure(figsize=(14, 8))

        ax = sns.barplot(
            data=df,
            x="Metric",
            y="Score",
            hue="Strategy",
            palette="husl",
            edgecolor="black",
            alpha=0.8,
        )

        # Set Log Scale for NDCG @10
        # Note: If multiple metrics are on one plot, log scale applies to the Y-axis globally.
        # It is highly recommended to use log scale only if the metrics share similar ranges.
        # ax.set_yscale("log")

        # Add rotated value labels
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    rotation=90,  # Rotated for better visibility
                )

        ax.set_title(
            "Fusion Strategy Performance",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Score (Log)", fontsize=12, fontweight="bold")

        plt.legend(title="Fusion Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if save:
            plt.savefig(
                "fusion_metrics_comparison_refined.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

    def plot_radar_chart(self, save: bool = True):
        """
        Create radar chart comparing all strategies (excluding MRR).
        """
        metrics = ["genre_overlap_avg", "ndcg@5", "diversity", "precision@5"]
        strategies = list(self.results.keys())

        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
        colors = sns.color_palette("husl", len(strategies))

        for idx, strategy in enumerate(strategies):
            values = []
            for metric in metrics:
                val = self.results[strategy]["metrics"].get(metric, {}).get("mean", 0)
                values.append(val)
            values += values[:1]

            ax.plot(
                angles, values, "o-", linewidth=2, label=strategy, color=colors[idx]
            )
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(
            "Fusion Strategy Comparison - Radar Chart",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        if save:
            plt.savefig("fusion_radar_comparison.png", dpi=300, bbox_inches="tight")
            print("✅ Saved: fusion_radar_comparison.png")
        plt.show()

    def plot_metric_distribution(self, metric: str = "ndcg@5", save: bool = True):
        """Plot distribution of a specific metric across queries."""
        fig, ax = plt.subplots(figsize=(12, 6))
        data_for_box = []
        labels = []

        for strategy_name, results in self.results.items():
            if "query_results" in results:
                values = [
                    qr["metrics"][metric]
                    for qr in results["query_results"]
                    if metric in qr["metrics"]
                ]
                data_for_box.append(values)
                labels.append(strategy_name)

        bp = ax.boxplot(
            data_for_box, labels=labels, patch_artist=True, notch=True, showmeans=True
        )
        colors = sns.color_palette("husl", len(labels))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(
            f"Distribution of {metric.replace('_', ' ').title()} Across Queries",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticklabels(labels, rotation=45, ha="right")
        plt.tight_layout()

        if save:
            plt.savefig(
                f"fusion_{metric}_distribution.png", dpi=300, bbox_inches="tight"
            )
            print(f"✅ Saved: fusion_{metric}_distribution.png")
        plt.show()

    def plot_heatmap(self, save: bool = True):
        """Create heatmap of all metrics (excluding MRR) vs strategies."""
        metrics = [
            "genre_overlap_avg",
            "genre_overlap_max",
            "score_similarity_avg",
            "ndcg@5",
            "ndcg@10",
            "diversity",
            "precision@5",
            "precision@10",
        ]
        strategies = list(self.results.keys())

        data_matrix = []
        for strategy in strategies:
            row = [
                self.results[strategy]["metrics"].get(m, {}).get("mean", 0)
                for m in metrics
            ]
            data_matrix.append(row)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_xticklabels(
            [m.replace("_", " ").replace("@", " @").title() for m in metrics],
            rotation=45,
            ha="right",
        )
        ax.set_yticklabels(strategies)

        plt.colorbar(im, ax=ax).set_label(
            "Score", rotation=270, labelpad=20, fontweight="bold"
        )

        for i in range(len(strategies)):
            for j in range(len(metrics)):
                ax.text(
                    j,
                    i,
                    f"{data_matrix[i][j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.set_title(
            "Fusion Strategy Performance Heatmap",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        if save:
            plt.savefig("fusion_heatmap.png", dpi=300, bbox_inches="tight")
            print("✅ Saved: fusion_heatmap.png")
        plt.show()

    def plot_ranking_comparison(self, save: bool = True):
        """Show how strategies rank for each metric (excluding MRR)."""
        metrics = ["genre_overlap_avg", "ndcg@5", "diversity", "precision@5"]
        strategies = list(self.results.keys())

        rankings = {strategy: [] for strategy in strategies}
        for metric in metrics:
            scores = [
                (s, self.results[s]["metrics"].get(metric, {}).get("mean", 0))
                for s in strategies
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (strategy, _) in enumerate(scores, 1):
                rankings[strategy].append(rank)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.15
        colors = sns.color_palette("husl", len(strategies))

        for idx, strategy in enumerate(strategies):
            ax.bar(
                x + (width * idx),
                rankings[strategy],
                width,
                label=strategy,
                color=colors[idx],
                alpha=0.8,
            )

        ax.set_ylabel("Rank (1 = Best)", fontsize=12, fontweight="bold")
        ax.set_title("Strategy Rankings Across Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in metrics], rotation=45, ha="right"
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.invert_yaxis()
        ax.set_yticks(range(1, len(strategies) + 1))

        plt.tight_layout()
        if save:
            plt.savefig("fusion_rankings.png", dpi=300, bbox_inches="tight")
            print("✅ Saved: fusion_rankings.png")
        plt.show()

    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATION PLOTS (MRR REMOVED)")
        print("=" * 60 + "\n")

        print("[1/5] Creating grouped metric comparison...")
        self.plot_metric_comparison()
        print("\n[2/5] Creating radar chart...")
        self.plot_radar_chart()
        print("\n[3/5] Creating distribution plot (NDCG@5)...")
        self.plot_metric_distribution("ndcg@5")
        print("\n[4/5] Creating heatmap...")
        self.plot_heatmap()
        print("\n[5/5] Creating ranking comparison...")
        self.plot_ranking_comparison()

        print("\n" + "=" * 60)
        print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
        print("=" * 60)


def main():
    try:
        visualizer = FusionResultsVisualizer()
        visualizer.generate_all_plots()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
