import gc
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from Libs.indexing_db import Indexing
from Libs.User import User

# ============================================================================
# CONFIGURATION
# ============================================================================
RECOMMENDATIONS_FILE = "./Embeddings/recommendations_output.jsonl"
ERROR_LOG = "./Embeddings/evaluation_errors.log"

TRAIN_SPLIT = 0.8  # 80% train, 20% test
TOP_K_RECOMMENDATIONS = 100
EVAL_K = 50  # Evaluate Recall@50 and Hit@50
MIN_ITEMS_FOR_SPLIT = 10  # Minimum items needed to create train/test split
MIN_TEST_SCORE = 6  # Only include test items with score >= 6

CHUNK_SIZE = 1000
GC_FREQUENCY = 100


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def log_error(user_id, error):
    """Log errors to file"""
    os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(f"User {user_id}: {str(error)}\n")


def get_processed_users():
    """Load set of already processed user IDs"""
    processed = set()
    if os.path.exists(RECOMMENDATIONS_FILE):
        try:
            with open(RECOMMENDATIONS_FILE, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed.add(str(data["user_id"]))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading file: {e}")
    return processed


def split_watchlist(watchlist, train_ratio=0.8, min_items=10, min_test_score=6):
    """
    Split watchlist into train and test sets.

    :param watchlist: List of [anime_id, score] pairs
    :param train_ratio: Ratio for training (default 0.8)
    :param min_items: Minimum items required to split
    :param min_test_score: Minimum score for test items
    :return: (train_list, test_list) or (watchlist, []) if too few items
    """
    if len(watchlist) < min_items:
        return watchlist, []

    # Shuffle for randomness
    np.random.seed(42)
    shuffled = watchlist.copy()
    np.random.shuffle(shuffled)

    # Split
    split_point = int(len(shuffled) * train_ratio)
    train_list = shuffled[:split_point]
    test_list = shuffled[split_point:]

    # Filter test set by minimum score
    test_list = [item for item in test_list if item[1] >= min_test_score]

    # If test set too small, return all as training
    if len(test_list) < 5:
        return watchlist, []

    return train_list, test_list


# ============================================================================
# PHASE 1: GENERATE RECOMMENDATIONS (NO METRICS)
# ============================================================================
def generate_recommendations():
    """
    Generate recommendations for all users and save to JSONL file.

    ONLY saves:
    - user_id
    - train_watchlist
    - test_watchlist
    - recommendations_from_train
    - recommendations_from_full

    NO METRICS are calculated here!
    This function is RESUMABLE - it skips already processed users.
    """

    print("\n" + "=" * 70)
    print("GENERATING RECOMMENDATIONS (NO METRICS)")
    print("=" * 70)
    print(f"Output: {RECOMMENDATIONS_FILE}")
    print(
        f"Train Split: {TRAIN_SPLIT * 100:.0f}% | Test Split: {(1 - TRAIN_SPLIT) * 100:.0f}%"
    )
    print(f"Recommendations: Top-{TOP_K_RECOMMENDATIONS}")

    # Create output directory
    os.makedirs(os.path.dirname(RECOMMENDATIONS_FILE), exist_ok=True)

    # Check resume point
    print("\n[1/5] Checking resume point...")
    processed_users = get_processed_users()
    print(f"       Already processed: {len(processed_users)} users")

    # Load vector database
    print("\n[2/5] Loading Vector Database...")
    try:
        index = Indexing()
        index.load_vector_database()
        print("       ✓ Vector DB loaded")
    except Exception as e:
        print(f"       ✗ ERROR: {e}")
        return

    # Load user data
    print("\n[3/5] Loading user data...")
    parquet_path = "./Dataset/UserAnimeList.parquet"
    df = pd.read_parquet(parquet_path)
    unique_users = df["user_id"].unique().tolist()
    print(f"       Found {len(unique_users)} total users")

    # Filter already processed
    users_to_process = [u for u in unique_users if str(u) not in processed_users]
    print(f"       To process: {len(users_to_process)} users")

    if len(users_to_process) == 0:
        print("\n       ✓ All users already processed!")
        return

    # Process users
    print("\n[4/5] Processing users...")
    successful = 0
    skipped = 0
    errors = 0

    with open(RECOMMENDATIONS_FILE, "a", buffering=1) as f_out:
        for i, user_id in enumerate(tqdm(users_to_process, desc="Generating")):
            try:
                # Get user data
                user_data = df[df["user_id"] == user_id]

                # Prepare full watchlist
                if "my_score" in user_data.columns:
                    full_watchlist = user_data[["anime_id", "my_score"]].values.tolist()
                else:
                    full_watchlist = [
                        [int(aid), 0] for aid in user_data["anime_id"].values
                    ]

                if not full_watchlist:
                    skipped += 1
                    continue

                # Split into train and test
                train_watchlist, test_watchlist = split_watchlist(
                    full_watchlist,
                    train_ratio=TRAIN_SPLIT,
                    min_items=MIN_ITEMS_FOR_SPLIT,
                    min_test_score=MIN_TEST_SCORE,
                )

                # Skip if no test set
                if not test_watchlist:
                    skipped += 1
                    continue

                # ===== Generate recommendations from TRAIN ONLY =====
                user_train = User(user_id, watched_list=train_watchlist)
                if not user_train.get_watchList():
                    skipped += 1
                    del user_train
                    continue

                user_train.findCentersOfClusters(index)
                recs_train = user_train.get_nearest_anime_from_clusters(
                    index, top_k=TOP_K_RECOMMENDATIONS
                )
                rec_ids_train = [int(anime["id"]) for anime in recs_train]
                del user_train

                # ===== Generate recommendations from FULL watchlist =====
                user_full = User(user_id, watched_list=full_watchlist)
                if not user_full.get_watchList():
                    skipped += 1
                    del user_full
                    continue

                user_full.findCentersOfClusters(index)
                recs_full = user_full.get_nearest_anime_from_clusters(
                    index, top_k=TOP_K_RECOMMENDATIONS
                )
                rec_ids_full = [int(anime["id"]) for anime in recs_full]
                del user_full

                # Save to file (NO METRICS)
                result = {
                    "user_id": str(user_id),
                    "train_watchlist": [
                        [int(item[0]), float(item[1])] for item in train_watchlist
                    ],
                    "test_watchlist": [
                        [int(item[0]), float(item[1])] for item in test_watchlist
                    ],
                    "recommendations_from_train": rec_ids_train,
                    "recommendations_from_full": rec_ids_full,
                }
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()  # Force write to disk
                successful += 1

                # Garbage collection
                if (i + 1) % GC_FREQUENCY == 0:
                    gc.collect()

            except Exception as e:
                errors += 1
                log_error(user_id, e)
                continue

    # Summary
    print("\n[5/5] Generation Complete!")
    print(f"       ✓ Successful: {successful}")
    print(f"       ⊘ Skipped: {skipped}")
    print(f"       ✗ Errors: {errors}")
    print(f"\n       💾 File saved to: {RECOMMENDATIONS_FILE}")
    print(f"       📝 Check with: wc -l {RECOMMENDATIONS_FILE}")
    print("=" * 70)


# ============================================================================
# PHASE 2: EVALUATE FROM FILE (CALCULATE ALL METRICS)
# ============================================================================
def calculate_recall_at_k(recommendations, ground_truth, k):
    """Calculate Recall@k"""
    if not ground_truth or not recommendations:
        return 0.0
    top_k_recs = set(recommendations[:k])
    relevant_items = set(ground_truth)
    hits = len(top_k_recs.intersection(relevant_items))
    return hits / len(relevant_items)


def calculate_hit_at_k(recommendations, ground_truth, k):
    """Calculate Hit@k"""
    if not ground_truth or not recommendations:
        return 0.0
    top_k_recs = set(recommendations[:k])
    relevant_items = set(ground_truth)
    return 1.0 if len(top_k_recs.intersection(relevant_items)) > 0 else 0.0


def calculate_catalog_coverage(recommendations, full_catalog):
    """Percentage of catalog items recommended at least once"""
    recommended = set()
    for rec_list in recommendations.values():
        recommended.update(rec_list)
    return len(recommended.intersection(full_catalog)) / len(full_catalog)


def calculate_shannon_entropy(recommendations):
    """Shannon entropy of recommendation distribution"""
    all_recs = []
    for rec_list in recommendations.values():
        all_recs.extend(rec_list)

    if not all_recs:
        return 0.0

    counts = Counter(all_recs)
    total = len(all_recs)
    probs = np.array([count / total for count in counts.values()])
    return -np.sum(probs * np.log2(probs))


def calculate_novelty(recommendations):
    """Average novelty across users"""
    item_counts = Counter()
    for rec_list in recommendations.values():
        item_counts.update(rec_list)

    if not item_counts:
        return 0.0

    max_count = max(item_counts.values())
    popularity_scores = {item: count / max_count for item, count in item_counts.items()}

    user_novelties = []
    for rec_list in recommendations.values():
        if not rec_list:
            continue
        novelty = np.mean([1 - popularity_scores.get(item, 0) for item in rec_list])
        user_novelties.append(novelty)

    return np.mean(user_novelties) if user_novelties else 0.0


def evaluate_from_file():
    """
    Load the JSONL file and calculate ALL metrics.

    Accuracy metrics (Recall@k, Hit@k):
    - Use recommendations_from_train vs test_watchlist

    Beyond-accuracy metrics (Coverage, Entropy, Novelty):
    - Use recommendations_from_full
    """

    print("\n" + "=" * 70)
    print("EVALUATING METRICS FROM FILE")
    print("=" * 70)
    print(f"Input: {RECOMMENDATIONS_FILE}")

    if not os.path.exists(RECOMMENDATIONS_FILE):
        print(f"\n✗ ERROR: File not found: {RECOMMENDATIONS_FILE}")
        print("Run: python script.py generate")
        return

    print(f"\n[1/4] Loading data from file...")

    # Storage for metrics calculation
    recall_scores = []
    hit_scores = []
    recommendations_from_train = {}  # For accuracy metrics
    recommendations_from_full = {}  # For beyond-accuracy metrics

    user_count = 0

    with open(RECOMMENDATIONS_FILE, "r") as f:
        for line in tqdm(f, desc="Loading"):
            try:
                data = json.loads(line)
                user_id = data["user_id"]
                user_count += 1

                # Extract data
                test_anime_ids = [item[0] for item in data["test_watchlist"]]
                recs_train = data["recommendations_from_train"]
                recs_full = data["recommendations_from_full"]

                # Store for beyond-accuracy metrics
                recommendations_from_train[user_id] = recs_train
                recommendations_from_full[user_id] = recs_full

                # Calculate accuracy metrics (using train recommendations vs test set)
                recall = calculate_recall_at_k(recs_train, test_anime_ids, k=EVAL_K)
                hit = calculate_hit_at_k(recs_train, test_anime_ids, k=EVAL_K)

                recall_scores.append(recall)
                hit_scores.append(hit)

            except Exception as e:
                print(f"Error loading line: {e}")
                continue

    print(f"       Loaded {user_count} users")

    # Calculate accuracy metrics
    print(f"\n[2/4] Calculating accuracy metrics...")
    print(f"       Using: recommendations_from_train vs test_watchlist")
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_hit = np.mean(hit_scores) if hit_scores else 0.0

    # Load catalog for beyond-accuracy metrics
    print(f"\n[3/4] Calculating beyond-accuracy metrics...")
    print(f"       Using: recommendations_from_full")

    try:
        df_catalog = pd.read_csv("./Dataset/AnimeList.csv")
        full_catalog = set(df_catalog["id"].unique())

        catalog_coverage = calculate_catalog_coverage(
            recommendations_from_full, full_catalog
        )
        shannon_entropy = calculate_shannon_entropy(recommendations_from_full)
        novelty = calculate_novelty(recommendations_from_full)

    except Exception as e:
        print(f"       Warning: Could not calculate beyond-accuracy metrics: {e}")
        catalog_coverage = shannon_entropy = novelty = 0.0
        full_catalog = set()

    # Display results
    print(f"\n[4/4] Results:")
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n🎯 ACCURACY METRICS")
    print(f"   Source: recommendations_from_train vs test_watchlist")
    print(f"   Users:           {len(recall_scores)}")
    print(f"   Recall@{EVAL_K}:       {avg_recall:.4f} ({avg_recall * 100:.2f}%)")
    print(f"   Hit@{EVAL_K}:          {avg_hit:.4f} ({avg_hit * 100:.2f}%)")

    print(f"\n🌟 BEYOND-ACCURACY METRICS")
    print(f"   Source: recommendations_from_full")
    print(f"   Users:           {len(recommendations_from_full)}")
    print(f"   Catalog size:    {len(full_catalog)}")
    print(
        f"   Catalog Coverage: {catalog_coverage:.4f} ({catalog_coverage * 100:.2f}%)"
    )
    print(f"   Shannon Entropy:  {shannon_entropy:.4f} bits")
    print(f"   Novelty Score:    {novelty:.4f}")

    print("\n" + "=" * 70)

    return {
        "accuracy": {
            "recall": avg_recall,
            "hit": avg_hit,
            "n_users": len(recall_scores),
        },
        "beyond_accuracy": {
            "catalog_coverage": catalog_coverage,
            "shannon_entropy": shannon_entropy,
            "novelty": novelty,
            "n_users": len(recommendations_from_full),
            "catalog_size": len(full_catalog),
        },
    }


def show_file_info():
    """Show information about the saved file"""

    print("\n" + "=" * 70)
    print("FILE INFORMATION")
    print("=" * 70)

    if not os.path.exists(RECOMMENDATIONS_FILE):
        print(f"\n✗ File not found: {RECOMMENDATIONS_FILE}")
        print("Run: python script.py generate")
        return

    # Count lines
    with open(RECOMMENDATIONS_FILE, "r") as f:
        line_count = sum(1 for _ in f)

    # File size
    file_size = os.path.getsize(RECOMMENDATIONS_FILE)
    file_size_mb = file_size / (1024 * 1024)

    print(f"\n📁 File: {RECOMMENDATIONS_FILE}")
    print(f"   Users saved:  {line_count}")
    print(f"   File size:    {file_size_mb:.2f} MB")

    # Show first user as example
    print(f"\n📝 Sample (first user):")
    with open(RECOMMENDATIONS_FILE, "r") as f:
        first_line = f.readline()
        data = json.loads(first_line)
        print(f"   User ID: {data['user_id']}")
        print(f"   Train watchlist: {len(data['train_watchlist'])} items")
        print(f"   Test watchlist:  {len(data['test_watchlist'])} items")
        print(f"   Recs from train: {len(data['recommendations_from_train'])} items")
        print(f"   Recs from full:  {len(data['recommendations_from_full'])} items")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run the evaluation pipeline based on command"""

    import sys

    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("RECOMMENDATION EVALUATION TOOL")
        print("=" * 70)
        print("\nUsage:")
        print("  python script.py generate   - Generate recommendations (resumable)")
        print("  python script.py evaluate   - Evaluate from saved file")
        print("  python script.py info       - Show file information")
        print("\nFile location: " + RECOMMENDATIONS_FILE)
        print("=" * 70)
        return

    command = sys.argv[1].lower()

    if command == "generate":
        generate_recommendations()
    elif command == "evaluate":
        evaluate_from_file()
    elif command == "info":
        show_file_info()
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: generate, evaluate, info")


if __name__ == "__main__":
    main()
