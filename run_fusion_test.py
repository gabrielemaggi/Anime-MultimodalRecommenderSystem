"""
Simple script to run fusion strategy evaluation on your anime dataset.

Usage:
    python run_fusion_test.py
"""

import random

from Libs import *
from test_fusion_strategies import FusionStrategyTester, run_fusion_evaluation


def main(sample_size=1000):  # Set default to 1 thousand
    """Run the fusion strategy evaluation with random samples."""

    print("=" * 80)
    print("ANIME RECOMMENDATION FUSION STRATEGY EVALUATION")
    print("=" * 80)

    # Step 1: Initialize the indexing system
    print("\n[1/3] Initializing indexing system...")
    indexing = Indexing()

    # Step 2: Load or build the vector database
    print("\n[2/3] Loading vector database...")
    indexing.load_vector_database()

    # Step 3: Select random test queries
    # Assuming indexing has a way to get all IDs, e.g., indexing.all_ids
    all_available_ids = indexing.get_all_ids()

    # Ensure we don't try to sample more than what exists
    if sample_size > len(all_available_ids):
        print(
            f"Warning: Requested {sample_size} elements, but only {len(all_available_ids)} available."
        )
        test_query_ids = all_available_ids
    else:
        test_query_ids = random.sample(all_available_ids, sample_size)

    print(f"\n[3/3] Running evaluation on {len(test_query_ids)} random test queries...")

    # Run the evaluation
    results = run_fusion_evaluation(indexing, test_query_ids)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved. Processed {len(test_query_ids)} items.")

    return results


def run_quick_test():
    """Quick test on a single query to verify setup."""
    print("Running quick verification test...")

    indexing = Indexing()
    indexing.load_vector_database()

    tester = FusionStrategyTester(indexing)

    # Test on a single anime
    test_id = 1  # Cowboy Bebop

    results = tester.test_fusion_strategy(
        fusion_method="mean", query_ids=[test_id], top_k=5
    )

    if results:
        print("\n✅ Quick test passed! System is working correctly.")
    else:
        print("\n❌ Quick test failed. Please check your setup.")

    return results


def compare_specific_strategies():
    """Compare specific fusion strategies in detail."""

    indexing = Indexing()
    indexing.load_vector_database()

    tester = FusionStrategyTester(indexing)

    # Define strategies to compare
    strategies_to_test = [
        ("mean", None),
        ("weighted", [0.7, 0.1, 0.2]),  # Heavy on synopsis
        ("weighted", [0.1, 0.7, 0.2]),  # Heavy on visual
        ("weighted", [0.33, 0.33, 0.34]),  # Balanced
    ]

    test_queries = [1, 5, 20, 1535, 5114]

    all_results = {}

    for method, weights in strategies_to_test:
        strategy_name = f"{method}" + (f"_{weights}" if weights else "")
        print(f"\n{'=' * 60}")
        print(f"Testing: {strategy_name}")
        print(f"{'=' * 60}")

        results = tester.test_fusion_strategy(
            fusion_method=method,
            query_ids=test_queries,
            top_k=10,
            fusion_weights=weights,
        )

        all_results[strategy_name] = results

    # Print comparison
    tester._print_comparison_summary(all_results)

    return all_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            run_quick_test()
        elif sys.argv[1] == "compare":
            compare_specific_strategies()
        else:
            print(
                "Unknown command. Use 'quick' or 'compare', or run without args for full evaluation."
            )
    else:
        main()
