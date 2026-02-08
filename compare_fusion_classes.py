"""
Dedicated comparison script for Classical Fusion vs Neural Fusion.

This script specifically compares:
- Classical methods (Fusion class): mean, weighted, concatenate
- Neural methods (FusionTrainer class): trainable attention-based fusion
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from Indexing import Indexing

from test_fusion_strategies import FusionStrategyTester


class FusionClassComparator:
    """Compare Classical (Fusion) vs Neural (FusionTrainer) fusion approaches."""

    def __init__(self, indexing_system):
        self.indexing = indexing_system
        self.tester = FusionStrategyTester(indexing_system)

    def compare_fusion_classes(
        self, query_ids: List[int], top_k: int = 10, test_variants: bool = True
    ) -> Dict:
        """
        Comprehensive comparison of Fusion vs FusionTrainer.

        Args:
            query_ids: Test query IDs
            top_k: Number of results to retrieve
            test_variants: If True, test multiple weight configurations for classical fusion

        Returns:
            Detailed comparison results
        """
        print("\n" + "=" * 80)
        print("CLASSICAL (Fusion) vs NEURAL (FusionTrainer) COMPARISON")
        print("=" * 80)

        results = {
            "classical": {},
            "neural": {},
            "metadata": {
                "query_ids": query_ids,
                "top_k": top_k,
                "num_queries": len(query_ids),
            },
        }

        # Test Classical Fusion methods
        print("\n📊 Testing CLASSICAL FUSION (Fusion class)")
        print("-" * 80)

        classical_configs = [
            ("mean", None),
            ("concatenate", None),
        ]

        if test_variants:
            classical_configs.extend(
                [
                    ("weighted", [0.7, 0.1, 0.2]),  # Synopsis-heavy
                    ("weighted", [0.4, 0.4, 0.2]),  # Balanced
                    ("weighted", [0.2, 0.6, 0.2]),  # Visual-heavy
                    ("weighted", [0.33, 0.33, 0.34]),  # Equal
                ]
            )
        else:
            classical_configs.append(("weighted", [0.4, 0.4, 0.2]))

        for method, weights in classical_configs:
            config_name = f"{method}" + (f"_{weights}" if weights else "")
            print(f"\n  Testing: {config_name}")

            result = self.tester.test_fusion_strategy(
                fusion_method=method,
                query_ids=query_ids,
                top_k=top_k,
                fusion_weights=weights,
            )

            if result:
                results["classical"][config_name] = result

        # Test Neural Fusion
        print("\n" + "=" * 80)
        print("📊 Testing NEURAL FUSION (FusionTrainer class)")
        print("-" * 80)

        if Path(self.indexing.fusion_model).exists():
            print("\n  Testing: trainable (attention-based)")
            result = self.tester.test_fusion_strategy(
                fusion_method="trainable",
                query_ids=query_ids,
                top_k=top_k,
                fusion_weights=None,
            )

            if result:
                results["neural"]["trainable"] = result
        else:
            print(
                f"\n  ⚠️  Neural fusion model not found at: {self.indexing.fusion_model}"
            )
            print("  Please train the model first using:")
            print("    indexing.build_vector_database()")

        # Generate comparison report
        self._print_detailed_comparison(results)

        # Save results
        self._save_comparison_results(results)

        return results

    def _print_detailed_comparison(self, results: Dict):
        """Print detailed comparison of classical vs neural approaches."""
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON REPORT")
        print("=" * 80)

        classical = results["classical"]
        neural = results["neural"]

        if not classical and not neural:
            print("No results to compare")
            return

        metrics = [
            "genre_overlap_avg",
            "ndcg@5",
            "ndcg@10",
            "diversity",
            "precision@5",
            "precision@10",
            "mrr",
        ]

        # 1. Best from each class
        print("\n1️⃣  BEST PERFORMER FROM EACH CLASS")
        print("-" * 80)

        if classical:
            best_classical = self._find_best_strategy(classical, "ndcg@5")
            print(f"\n🏆 Best Classical: {best_classical['name']}")
            print(f"   Method: {best_classical['method']}")
            print(f"   NDCG@5: {best_classical['score']:.3f}")
            self._print_metric_summary(
                classical[best_classical["name"]]["metrics"], indent=3
            )

        if neural:
            best_neural = self._find_best_strategy(neural, "ndcg@5")
            print(f"\n🏆 Best Neural: {best_neural['name']}")
            print(f"   Method: Attention-based fusion with CoMM loss")
            print(f"   NDCG@5: {best_neural['score']:.3f}")
            self._print_metric_summary(neural[best_neural["name"]]["metrics"], indent=3)

        # 2. Head-to-head comparison
        print("\n" + "=" * 80)
        print("2️⃣  HEAD-TO-HEAD METRIC COMPARISON")
        print("-" * 80)

        print(
            f"\n{'Metric':<25} {'Classical Best':>15} {'Neural Best':>15} {'Winner':>15} {'Advantage':>15}"
        )
        print("-" * 85)

        for metric in metrics:
            # Get best scores from each class
            classical_best = max(
                [
                    r["metrics"][metric]["mean"]
                    for r in classical.values()
                    if metric in r["metrics"]
                ],
                default=0,
            )
            neural_best = max(
                [
                    r["metrics"][metric]["mean"]
                    for r in neural.values()
                    if metric in r["metrics"]
                ],
                default=0,
            )

            # Determine winner
            if classical_best > neural_best:
                winner = "Classical ✓"
                advantage = f"+{(classical_best - neural_best):.3f}"
            elif neural_best > classical_best:
                winner = "Neural ✓"
                advantage = f"+{(neural_best - classical_best):.3f}"
            else:
                winner = "Tie"
                advantage = "0.000"

            print(
                f"{metric:<25} {classical_best:>15.3f} {neural_best:>15.3f} {winner:>15} {advantage:>15}"
            )

        # 3. Statistical comparison
        print("\n" + "=" * 80)
        print("3️⃣  STATISTICAL SUMMARY")
        print("-" * 80)

        for metric in ["ndcg@5", "genre_overlap_avg", "diversity", "precision@5"]:
            print(f"\n📊 {metric.upper()}:")

            # Classical statistics
            classical_scores = [
                r["metrics"][metric]["mean"]
                for r in classical.values()
                if metric in r["metrics"]
            ]
            if classical_scores:
                print(
                    f"   Classical: μ={np.mean(classical_scores):.3f}, "
                    f"σ={np.std(classical_scores):.3f}, "
                    f"range=[{np.min(classical_scores):.3f}, {np.max(classical_scores):.3f}]"
                )

            # Neural statistics
            neural_scores = [
                r["metrics"][metric]["mean"]
                for r in neural.values()
                if metric in r["metrics"]
            ]
            if neural_scores:
                print(
                    f"   Neural:    μ={np.mean(neural_scores):.3f}, "
                    f"σ={np.std(neural_scores):.3f}, "
                    f"range=[{np.min(neural_scores):.3f}, {np.max(neural_scores):.3f}]"
                )

        # 4. Recommendations
        print("\n" + "=" * 80)
        print("4️⃣  RECOMMENDATIONS")
        print("-" * 80)

        if classical and neural:
            # Calculate overall scores
            classical_avg = np.mean(
                [
                    r["metrics"]["ndcg@5"]["mean"]
                    for r in classical.values()
                    if "ndcg@5" in r["metrics"]
                ]
            )
            neural_avg = np.mean(
                [
                    r["metrics"]["ndcg@5"]["mean"]
                    for r in neural.values()
                    if "ndcg@5" in r["metrics"]
                ]
            )

            print("\n💡 Based on NDCG@5 (primary metric):")
            if neural_avg > classical_avg + 0.02:  # 2% threshold
                print("   ✅ NEURAL FUSION is recommended")
                print(
                    f"      - {(neural_avg - classical_avg) * 100:.1f}% better ranking quality"
                )
                print("      - Learned representations capture complex patterns")
                print("      - Trade-off: Requires pre-training and more computation")
            elif classical_avg > neural_avg + 0.02:
                print("   ✅ CLASSICAL FUSION is recommended")
                print(
                    f"      - {(classical_avg - neural_avg) * 100:.1f}% better ranking quality"
                )
                print("      - Simpler, more interpretable")
                print("      - No training required")
            else:
                print("   ⚖️  PERFORMANCE IS COMPARABLE")
                print("      - Neural: Better learned representations")
                print("      - Classical: Simpler and faster")
                print(
                    "      - Choose based on computational budget and interpretability needs"
                )

            # Specific use case recommendations
            print("\n📌 Use Case Recommendations:")

            # Diversity winner
            diversity_classical = np.mean(
                [r["metrics"]["diversity"]["mean"] for r in classical.values()]
            )
            diversity_neural = np.mean(
                [r["metrics"]["diversity"]["mean"] for r in neural.values()]
            )

            if diversity_classical > diversity_neural:
                print("   • Content Discovery → Classical (better diversity)")
            else:
                print("   • Content Discovery → Neural (better diversity)")

            # Genre matching winner
            genre_classical = np.mean(
                [r["metrics"]["genre_overlap_avg"]["mean"] for r in classical.values()]
            )
            genre_neural = np.mean(
                [r["metrics"]["genre_overlap_avg"]["mean"] for r in neural.values()]
            )

            if genre_classical > genre_neural:
                print("   • Genre-based Search → Classical (better genre matching)")
            else:
                print("   • Genre-based Search → Neural (better genre matching)")

            print("   • Real-time Systems → Classical (faster inference)")
            print("   • Offline Batch Processing → Neural (quality over speed)")

    def _find_best_strategy(self, results: Dict, metric: str = "ndcg@5") -> Dict:
        """Find the best performing strategy based on a metric."""
        best_name = None
        best_score = -1

        for name, result in results.items():
            if metric in result["metrics"]:
                score = result["metrics"][metric]["mean"]
                if score > best_score:
                    best_score = score
                    best_name = name

        return {
            "name": best_name,
            "method": results[best_name]["fusion_method"],
            "score": best_score,
        }

    def _print_metric_summary(self, metrics: Dict, indent: int = 0):
        """Print a summary of metrics."""
        spaces = "   " * indent

        key_metrics = [
            "genre_overlap_avg",
            "ndcg@10",
            "diversity",
            "precision@5",
            "mrr",
        ]

        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]["mean"]
                print(f"{spaces}{metric}: {value:.3f}")

    def _save_comparison_results(self, results: Dict):
        """Save comparison results to JSON."""
        output_path = Path("./classical_vs_neural_comparison.json")

        # Make serializable
        def make_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = make_serializable(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Detailed comparison saved to {output_path}")


def main():
    """Run classical vs neural fusion comparison."""
    print("=" * 80)
    print("CLASSICAL FUSION vs NEURAL FUSION COMPARISON")
    print("=" * 80)

    # Initialize
    print("\n[1/3] Initializing indexing system...")
    indexing = Indexing()
    indexing.load_vector_database()

    # Define test queries
    test_queries = [
        1,  # Cowboy Bebop
        5,  # Cowboy Bebop Movie
        20,  # Naruto
        30,  # Neon Genesis Evangelion
        199,  # Sen to Chihiro
        467,  # Samurai Champloo
        1535,  # Death Note
        5114,  # FMA: Brotherhood
        9253,  # Steins;Gate
        11757,  # Sword Art Online
    ]

    print(f"\n[2/3] Running comparison on {len(test_queries)} queries...")

    # Run comparison
    comparator = FusionClassComparator(indexing)
    results = comparator.compare_fusion_classes(
        query_ids=test_queries,
        top_k=10,
        test_variants=True,  # Test multiple weight configurations
    )

    print("\n[3/3] Comparison complete!")
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - classical_vs_neural_comparison.json")

    return results


if __name__ == "__main__":
    main()
