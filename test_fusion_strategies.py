import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar


class FusionStrategyTester:
    """
    Test and evaluate different fusion strategies for anime recommendation.
    """

    def __init__(self, indexing_system):
        """
        Args:
            indexing_system: An instance of the Indexing class
        """
        self.indexing = indexing_system
        self.results = {}

    def calculate_genre_overlap(self, query_genres: str, result_genres: str) -> float:
        """Calculate Jaccard similarity between genre sets."""
        if pd.isna(query_genres) or pd.isna(result_genres):
            return 0.0

        query_set = set([g.strip() for g in str(query_genres).split(",")])
        result_set = set([g.strip() for g in str(result_genres).split(",")])

        if not query_set or not result_set:
            return 0.0

        intersection = len(query_set & result_set)
        union = len(query_set | result_set)

        return intersection / union if union > 0 else 0.0

    def calculate_score_similarity(
        self, query_score: float, result_score: float
    ) -> float:
        """Calculate normalized score similarity."""
        if pd.isna(query_score) or pd.isna(result_score):
            return 0.0

        # Max difference is 10 (scores range 0-10)
        diff = abs(float(query_score) - float(result_score))
        return 1 - (diff / 10.0)

    def calculate_ndcg(self, relevance_scores: List[float], k: int = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not relevance_scores:
            return 0.0

        if k is not None:
            relevance_scores = relevance_scores[:k]

        # DCG
        dcg = sum(
            [
                (2**rel - 1) / np.log2(idx + 2)  # +2 because idx starts at 0
                for idx, rel in enumerate(relevance_scores)
            ]
        )

        # IDCG (ideal DCG with scores sorted)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(
            [(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores)]
        )

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_diversity(self, results: List[Dict]) -> float:
        """Calculate diversity of results based on genre distribution."""
        all_genres = []
        for result in results:
            genres = result.get("genre", "")
            if not pd.isna(genres):
                all_genres.extend([g.strip() for g in str(genres).split(",")])

        if not all_genres:
            return 0.0

        # Calculate entropy of genre distribution
        genre_counts = Counter(all_genres)
        total = len(all_genres)

        entropy = -sum(
            [
                (count / total) * np.log2(count / total)
                for count in genre_counts.values()
            ]
        )

        # Normalize by max possible entropy
        max_entropy = np.log2(len(genre_counts)) if len(genre_counts) > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def calculate_mrr(
        self, relevance_scores: List[float], threshold: float = 0.5
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for idx, score in enumerate(relevance_scores):
            if score >= threshold:
                return 1.0 / (idx + 1)
        return 0.0

    def evaluate_results(
        self, query_id: int, results: List[Dict], query_info: Dict
    ) -> Dict:
        """Evaluate a set of search results against the query."""
        if not results:
            return {
                "genre_overlap_avg": 0.0,
                "score_similarity_avg": 0.0,
                "ndcg@5": 0.0,
                "ndcg@10": 0.0,
                "mrr": 0.0,
                "diversity": 0.0,
                "precision@5": 0.0,
            }

        query_genres = query_info.get("genre", "")
        query_score = query_info.get("score", 0.0)

        # Calculate relevance scores for each result
        genre_overlaps = []
        score_similarities = []
        combined_relevance = []

        for result in results:
            # Skip the query itself if it appears in results
            if str(result.get("id")) == str(query_id):
                continue

            genre_overlap = self.calculate_genre_overlap(
                query_genres, result.get("genre", "")
            )
            score_sim = self.calculate_score_similarity(
                query_score, result.get("score", 0.0)
            )

            genre_overlaps.append(genre_overlap)
            score_similarities.append(score_sim)

            # Combined relevance (weighted)
            combined = 0.7 * genre_overlap + 0.3 * score_sim
            combined_relevance.append(combined)

        # Calculate metrics
        metrics = {
            "genre_overlap_avg": np.mean(genre_overlaps) if genre_overlaps else 0.0,
            "genre_overlap_max": np.max(genre_overlaps) if genre_overlaps else 0.0,
            "score_similarity_avg": np.mean(score_similarities)
            if score_similarities
            else 0.0,
            "ndcg@5": self.calculate_ndcg(combined_relevance, k=5),
            "ndcg@10": self.calculate_ndcg(combined_relevance, k=10),
            "mrr": self.calculate_mrr(combined_relevance, threshold=0.5),
            "diversity": self.calculate_diversity(results),
            "precision@5": sum(1 for r in combined_relevance[:5] if r >= 0.5) / 5.0,
            "precision@10": sum(1 for r in combined_relevance[:10] if r >= 0.5) / 10.0,
            "avg_distance": np.mean([r.get("distance", 0) for r in results]),
        }

        return metrics

    def test_fusion_strategy(
        self,
        fusion_method: str,
        query_ids: List[int],
        top_k: int = 10,
        fusion_weights: List[float] = None,
    ) -> Dict:
        """
        Test a single fusion strategy on multiple queries.
        """
        print(f"\nTesting Fusion Strategy: {fusion_method.upper()}")
        if fusion_weights:
            print(f"Weights: {fusion_weights}")

        # Temporarily change fusion method
        original_method = self.indexing.fusion_method
        original_weights = self.indexing.fusion_weights

        self.indexing.fusion_method = fusion_method
        if fusion_weights:
            self.indexing.fusion_weights = fusion_weights

        all_metrics = []
        query_results = []

        # Use tqdm for progress bar
        pbar = tqdm(query_ids, desc=f"Evaluating {fusion_method}", unit="query")

        for query_id in pbar:
            # Removed per-query console logs

            # Get query anime info
            query_info = self.indexing.get_anime_info_by_id(query_id)
            if query_info is None:
                # Use tqdm.write to print without breaking the progress bar
                # tqdm.write(f"⚠️ Anime {query_id} not found, skipping...")
                continue

            try:
                # Search using current fusion method
                results = self.indexing.search_by_id(query_id, top_k=top_k + 1)

                # Remove query itself from results if present
                results = [r for r in results if str(r.get("id")) != str(query_id)][
                    :top_k
                ]

                # Evaluate results
                metrics = self.evaluate_results(query_id, results, query_info)
                all_metrics.append(metrics)

                # Store for detailed analysis
                query_results.append(
                    {
                        "query_id": query_id,
                        "query_title": query_info.get("title"),
                        "results": results,
                        "metrics": metrics,
                    }
                )

                # Optional: Update pbar postfix with running average of a key metric
                current_ndcg = metrics["ndcg@5"]
                pbar.set_postfix({"last_ndcg": f"{current_ndcg:.2f}"})

            except Exception as e:
                tqdm.write(f"❌ Error processing query {query_id}: {e}")
                continue

        # Restore original settings
        self.indexing.fusion_method = original_method
        self.indexing.fusion_weights = original_weights

        # Aggregate metrics
        if not all_metrics:
            print("\n⚠️ No successful queries processed")
            return {}

        aggregated = {
            "fusion_method": fusion_method,
            "fusion_weights": fusion_weights,
            "num_queries": len(all_metrics),
            "metrics": {
                key: {
                    "mean": np.mean([m[key] for m in all_metrics]),
                    "std": np.std([m[key] for m in all_metrics]),
                    "min": np.min([m[key] for m in all_metrics]),
                    "max": np.max([m[key] for m in all_metrics]),
                }
                for key in all_metrics[0].keys()
            },
            "query_results": query_results,
        }

        return aggregated

    def compare_all_strategies(
        self, query_ids: List[int], top_k: int = 10, save_results: bool = True
    ) -> Dict:
        """
        Compare all fusion strategies on the same set of queries.
        """
        print("\n" + "=" * 80)
        print("FUSION STRATEGY COMPARISON")
        print("=" * 80)
        print("📊 Testing Classical Fusion Methods (Fusion class)")
        print("📊 Testing Neural Fusion Methods (FusionTrainer class)")
        print("=" * 80)

        # Classical Fusion strategies (using Fusion class)
        classical_strategies = [
            {"method": "mean", "weights": None, "class": "Fusion"},
            {
                "method": "weighted",
                "weights": [1.0, 0, 0],
                "class": "Fusion",
            },
            {
                "method": "weighted",
                "weights": [0, 1.0, 0],
                "class": "Fusion",
            },
            {
                "method": "weighted",
                "weights": [0, 0, 1.0],
                "class": "Fusion",
            },
            {
                "method": "weighted",
                "weights": [0.7, 0.1, 0.2],
                "class": "Fusion",
            },  # Synopsis-heavy
            {
                "method": "weighted",
                "weights": [0.4, 0.4, 0.2],
                "class": "Fusion",
            },  # Balanced
            {
                "method": "weighted",
                "weights": [0.5, 0.3, 0.2],
                "class": "Fusion",
            },  # Mid-range
            {
                "method": "weighted",
                "weights": [0.2, 0.6, 0.2],
                "class": "Fusion",
            },  # Visual-heavy
            {"method": "concatenate", "weights": None, "class": "Fusion"},
        ]

        # Neural Fusion strategy (using FusionTrainer class)
        neural_strategies = []
        if (
            hasattr(self.indexing, "fusion_model")
            and Path(self.indexing.fusion_model).exists()
        ):
            neural_strategies.append(
                {"method": "trainable", "weights": None, "class": "FusionTrainer"}
            )
        else:
            # Check just in case variable is string path or checking location logic
            pass
            # Note: Logic slightly adjusted to prevent error if fusion_model attr doesn't exist on simple indexing class

        all_strategies = classical_strategies + neural_strategies
        all_results = {}

        for strategy in all_strategies:
            method = strategy["method"]
            weights = strategy["weights"]
            fusion_class = strategy["class"]

            # Create descriptive name
            if method == "trainable":
                strategy_name = f"{method} (Neural)"
            else:
                strategy_name = f"{method} (Classical)"
                if weights:
                    strategy_name += f" {weights}"

            results = self.test_fusion_strategy(
                fusion_method=method,
                query_ids=query_ids,
                top_k=top_k,
                fusion_weights=weights,
            )

            if results:
                results["fusion_class"] = fusion_class  # Track which class was used
                all_results[strategy_name] = results

        # Print comparison summary
        self._print_comparison_summary(all_results)

        # Print class-specific insights
        self._print_class_comparison(all_results)

        # Save results
        if save_results:
            self._save_results(all_results)

        return all_results

    def _print_comparison_summary(self, all_results: Dict):
        """Print a summary comparison table of all strategies."""
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)

        if not all_results:
            print("No results to compare")
            return

        # Extract key metrics
        metrics_to_compare = [
            "genre_overlap_avg",
            "ndcg@5",
            "ndcg@10",
            "diversity",
            "precision@5",
            "mrr",
        ]

        print(
            f"\n{'Strategy':<40} {'Genre Overlap':>12} {'NDCG@5':>10} "
            f"{'NDCG@10':>10} {'Diversity':>10} {'Prec@5':>10} {'MRR':>10}"
        )
        print("-" * 102)

        for strategy_name, results in all_results.items():
            metrics = results["metrics"]

            row = f"{strategy_name:<40}"
            for metric in metrics_to_compare:
                if metric in metrics:
                    mean_val = metrics[metric]["mean"]
                    row += f" {mean_val:>10.3f}"
                else:
                    row += f" {'N/A':>10}"

            print(row)

        # Find best strategy for each metric
        print("\n" + "=" * 80)
        print("BEST STRATEGY PER METRIC")
        print("=" * 80)

        for metric in metrics_to_compare:
            best_strategy = None
            best_score = -1

            for strategy_name, results in all_results.items():
                if metric in results["metrics"]:
                    score = results["metrics"][metric]["mean"]
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy_name

            if best_strategy:
                print(f"{metric:<25}: {best_strategy:<40} ({best_score:.3f})")

    def _print_class_comparison(self, all_results: Dict):
        """Compare Classical (Fusion) vs Neural (FusionTrainer) approaches."""
        print("\n" + "=" * 80)
        print("CLASSICAL vs NEURAL FUSION COMPARISON")
        print("=" * 80)

        # Separate results by fusion class
        classical_results = {
            k: v for k, v in all_results.items() if v.get("fusion_class") == "Fusion"
        }
        neural_results = {
            k: v
            for k, v in all_results.items()
            if v.get("fusion_class") == "FusionTrainer"
        }

        if not classical_results and not neural_results:
            print("No results to compare")
            return

        metrics = [
            "genre_overlap_avg",
            "ndcg@5",
            "ndcg@10",
            "diversity",
            "precision@5",
            "mrr",
        ]

        print("\n📊 Average Performance by Fusion Type:")
        print("-" * 80)

        for metric in metrics:
            # Classical average
            classical_scores = [
                r["metrics"][metric]["mean"]
                for r in classical_results.values()
                if metric in r["metrics"]
            ]
            classical_avg = np.mean(classical_scores) if classical_scores else 0

            # Neural average
            neural_scores = [
                r["metrics"][metric]["mean"]
                for r in neural_results.values()
                if metric in r["metrics"]
            ]
            neural_avg = np.mean(neural_scores) if neural_scores else 0

            # Determine winner
            if classical_avg > neural_avg:
                winner = "Classical ✓"
                diff = classical_avg - neural_avg
            elif neural_avg > classical_avg:
                winner = "Neural ✓"
                diff = neural_avg - classical_avg
            else:
                winner = "Tie"
                diff = 0

            print(
                f"{metric:<25} | Classical: {classical_avg:.3f} | Neural: {neural_avg:.3f} "
                f"| Winner: {winner:<15} (Δ {diff:.3f})"
            )

        # Best from each class
        if classical_results:
            print("\n🏆 Best Classical Strategy:")
            best_classical = max(
                classical_results.items(),
                key=lambda x: x[1]["metrics"]["ndcg@5"]["mean"],
            )
            print(
                f"   {best_classical[0]} (NDCG@5: {best_classical[1]['metrics']['ndcg@5']['mean']:.3f})"
            )

        if neural_results:
            print("\n🏆 Best Neural Strategy:")
            best_neural = max(
                neural_results.items(), key=lambda x: x[1]["metrics"]["ndcg@5"]["mean"]
            )
            print(
                f"   {best_neural[0]} (NDCG@5: {best_neural[1]['metrics']['ndcg@5']['mean']:.3f})"
            )

        # Overall recommendation
        print("\n💡 Recommendation:")
        if neural_scores and classical_scores:
            if np.mean(neural_scores) > np.mean(classical_scores):
                print("   → Use Neural Fusion (trainable) for best overall performance")
            else:
                print(
                    "   → Classical fusion methods are competitive and computationally cheaper"
                )
        elif neural_scores:
            print("   → Neural fusion is available and performing well")
        else:
            print(
                "   → Only classical fusion tested. Consider training a neural fusion model."
            )

    def _save_results(self, all_results: Dict):
        """Save results to JSON file."""
        output_path = Path("./fusion_strategy_evaluation.json")

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(all_results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Results saved to {output_path}")


def run_fusion_evaluation(indexing_system, test_query_ids: List[int] = None):
    """
    Main function to run fusion strategy evaluation.
    """
    # Default test queries if none provided
    if test_query_ids is None:
        sample_size = 1000
        # Mix of different genres and types
        all_available_ids = indexing_system.get_all_ids()
        # Ensure we don't try to sample more than what exists
        if sample_size > len(all_available_ids):
            print(
                f"Warning: Requested {sample_size} elements, but only {len(all_available_ids)} available."
            )
            test_query_ids = all_available_ids
        else:
            test_query_ids = random.sample(all_available_ids, sample_size)

    # Initialize tester
    tester = FusionStrategyTester(indexing_system)

    # Run comparison
    results = tester.compare_all_strategies(
        query_ids=test_query_ids, top_k=10, save_results=True
    )

    return results


# Example usage
if __name__ == "__main__":
    # This is a usage example - adjust the import path as needed
    from Libs import Indexing

    print("Initializing indexing system...")
    indexing = Indexing()
    indexing.load_vector_database()

    print("\nStarting fusion strategy evaluation...")
    results = run_fusion_evaluation(indexing)

    print("\n✅ Evaluation complete!")
