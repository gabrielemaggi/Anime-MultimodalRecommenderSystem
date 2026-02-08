import gc
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm

from Libs.indexing_db import Indexing
from Libs.User import User

# ============================================================================
# CONFIGURATION
# ============================================================================
RECOMMENDATIONS_FILE = "./Embeddings/recommendations_output.jsonl"
ERROR_LOG = "./Embeddings/evaluation_errors.log"

TRAIN_SPLIT = 0.8  # 80% train, 20% test
TOP_K_RECOMMENDATIONS = 50  # Generate top-50 recommendations
EVAL_K_VALUES = [20, 25, 30, 40, 50]  # Test multiple k values for Recall@k and Hit@k
MIN_ITEMS_FOR_SPLIT = 20  # Minimum items needed to create train/test split
MIN_TEST_SCORE = 6  # Only include test items with score >= 6
FIXED_TEST_SIZE = 20  # Fixed number of test items per user
NEAR = True  # True for calculating metrics using cosine similarity
SIMILARITY_THRESHOLD = 0.7  # Threshold for similarity between items

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


def split_watchlist(watchlist, fixed_test_size=5, min_items=20, min_test_score=6):
    """
    Split watchlist into train and test sets with FIXED test size.

    :param watchlist: List of [anime_id, score] pairs
    :param fixed_test_size: Fixed number of items in test set
    :param min_items: Minimum items required to split
    :param min_test_score: Minimum score for test items
    :return: (train_list, test_list) or (watchlist, []) if too few items
    """
    if len(watchlist) < min_items:
        return watchlist, []

    # Filter items with score >= min_test_score (eligible for test set)
    eligible_test = [item for item in watchlist if item[1] >= min_test_score]

    # If not enough eligible items, return all as training
    if len(eligible_test) < fixed_test_size:
        return watchlist, []

    # Shuffle for randomness
    np.random.seed(42)
    shuffled_eligible = eligible_test.copy()
    np.random.shuffle(shuffled_eligible)

    # Take exactly fixed_test_size items for test
    test_list = shuffled_eligible[:fixed_test_size]

    # Everything else goes to training (including low-scored items)
    test_ids = set(item[0] for item in test_list)
    train_list = [item for item in watchlist if item[0] not in test_ids]

    print(len(train_list))

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
    print(f"Fixed Test Size: {FIXED_TEST_SIZE} items per user")
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

                # Split into train and test with FIXED test size
                train_watchlist, test_watchlist = split_watchlist(
                    full_watchlist,
                    fixed_test_size=FIXED_TEST_SIZE,
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


def calculate_ndcg_at_k(recommendations, test_watchlist, k):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain)

    Measures ranking quality by considering:
    1. Whether relevant items are in top-k
    2. WHERE they are ranked (higher = better)
    3. HOW relevant they are (based on user scores)

    :param recommendations: List of recommended anime IDs (ordered by rank)
    :param test_watchlist: List of [anime_id, score] pairs from test set
    :param k: Cutoff position
    :return: NDCG@k score (0-1, higher is better)
    """
    if not test_watchlist or not recommendations:
        return 0.0

    # Create relevance dictionary: {anime_id: user_score}
    relevance = {int(item[0]): float(item[1]) for item in test_watchlist}

    # Calculate DCG@k (Discounted Cumulative Gain)
    dcg = 0.0
    for i, rec_id in enumerate(recommendations[:k]):
        if rec_id in relevance:
            # Gain at position i (0-indexed)
            # Discount by position: log2(i+2) because positions start at 1
            gain = relevance[rec_id]
            discount = np.log2(
                i + 2
            )  # i+2 because: position 0 -> log2(2), position 1 -> log2(3), etc.
            dcg += gain / discount

    # Calculate IDCG@k (Ideal DCG - best possible ranking)
    # Sort test items by score (descending) and calculate ideal DCG
    sorted_scores = sorted(relevance.values(), reverse=True)
    idcg = 0.0
    for i, score in enumerate(sorted_scores[:k]):
        idcg += score / np.log2(i + 2)

    # Normalize: NDCG = DCG / IDCG
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_near_hit_at_k(
    recommendations, test_items, k, indexing, similarity_threshold=0.5
):
    """
    Near-Hit@k: Did we recommend at least one item similar to test items?

    This is a BINARY metric (like Hit@k) but with similarity tolerance.

    :param recommendations: List of recommended anime IDs (ordered)
    :param test_items: List of test anime IDs
    :param k: Cutoff position
    :param embedding_dict: Dict mapping anime_id -> embedding vector
    :param similarity_threshold: Minimum cosine similarity to count as "similar"
    :return: 1.0 if found similar item, 0.0 otherwise
    """
    top_k_recs = recommendations[:k]

    for rec_id in top_k_recs:
        # Skip if recommendation has no embedding

        rec_embedding = indexing.get_db_embedding_by_id(int(rec_id))

        for test_id in test_items:
            test_embedding = indexing.get_db_embedding_by_id(int(test_id))

            # Calculate cosine similarity
            similarity = 1 - cosine(rec_embedding, test_embedding)

            # If similar enough, count as near-hit
            if similarity >= similarity_threshold:
                return 1.0

    return 0.0


def calculate_near_recall_at_k(
    recommendations, test_items, k, indexing, similarity_threshold=0.5
):
    """
    Near-Recall@k: What % of test items have a similar item in top-k?

    This extends Recall@k with similarity tolerance.

    :param recommendations: List of recommended anime IDs (ordered)
    :param test_items: List of test anime IDs
    :param k: Cutoff position
    :param embedding_dict: Dict mapping anime_id -> embedding vector
    :param similarity_threshold: Minimum cosine similarity to count as "similar"
    :return: Proportion of test items with similar recommendation (0-1)
    """
    if not test_items:
        return 0.0

    top_k_recs = recommendations[:k]
    matched_test_items = 0

    for test_id in test_items:
        # Skip if test item has no embedding

        test_embedding = indexing.get_db_embedding_by_id(int(test_id))
        found_similar = False

        for rec_id in top_k_recs:
            # Skip if recommendation has no embedding

            rec_embedding = indexing.get_db_embedding_by_id(int(rec_id))

            # Calculate cosine similarity
            similarity = 1 - cosine(rec_embedding, test_embedding)

            # If similar enough, this test item is "matched"
            if similarity >= similarity_threshold:
                found_similar = True
                break

        if found_similar:
            matched_test_items += 1

    return matched_test_items / len(test_items)


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


def calculate_novelty(recommendations, training_data):
    """
    Average novelty across users.

    Novelty = average of (1 - popularity_score(i)) for recommended items
    where popularity_score(i) = (# users who interacted with anime i) / (max popularity in catalog)

    :param recommendations: Dict of {user_id: [recommended_anime_ids]}
    :param training_data: DataFrame with columns ['user_id', 'anime_id'] - SUBSET of training data
    :return: Average novelty score
    """
    try:
        # Count how many USERS interacted with each anime in the training subset
        item_user_counts = (
            training_data.groupby("anime_id")["user_id"].nunique().to_dict()
        )

        if not item_user_counts:
            return 0.0

        # Get max popularity (most popular item)
        max_popularity = max(item_user_counts.values())

        # Calculate normalized popularity score for each item
        # popularity_score(i) = (# users who interacted with i) / max_popularity
        popularity_scores = {
            item: count / max_popularity for item, count in item_user_counts.items()
        }

        # Calculate novelty for each user
        user_novelties = []
        for rec_list in recommendations.values():
            if not rec_list:
                continue

            # For each recommended item, calculate (1 - popularity_score)
            # Items not in training data have popularity_score = 0 (max novelty)
            novelty_values = [1 - popularity_scores.get(item, 0) for item in rec_list]
            user_novelty = np.mean(novelty_values)
            user_novelties.append(user_novelty)

        return np.mean(user_novelties) if user_novelties else 0.0

    except Exception as e:
        print(f"       Warning: Could not calculate novelty: {e}")
        return 0.0


def evaluate_from_file():
    """
    Load the JSONL file and calculate ALL metrics at MULTIPLE K VALUES.

    Accuracy metrics (Recall@k, Hit@k, NDCG@k):
    - Use recommendations_from_train vs test_watchlist
    - Evaluate at multiple k values

    Beyond-accuracy metrics (Coverage, Entropy, Novelty):
    - Use recommendations_from_full
    - Only use training data from the SAME users in the recommendations file
    """
    index = Indexing()
    index.load_vector_database()

    print("\n" + "=" * 70)
    print("EVALUATING METRICS FROM FILE")
    print("=" * 70)
    print(f"Input: {RECOMMENDATIONS_FILE}")
    print(f"Testing k values: {EVAL_K_VALUES}")

    if not os.path.exists(RECOMMENDATIONS_FILE):
        print(f"\n✗ ERROR: File not found: {RECOMMENDATIONS_FILE}")
        print("Run: python script.py generate")
        return

    print(f"\n[1/5] Loading data from file...")

    # Storage for metrics calculation at different k values
    metrics_by_k = {
        k: {"recall": [], "hit": [], "ndcg": [], "near_recall": [], "near_hit": []}
        for k in EVAL_K_VALUES
    }

    recommendations_from_train = {}  # For accuracy metrics
    recommendations_from_full = {}  # For beyond-accuracy metrics
    user_ids_in_file = []  # Track which users are in the file

    user_count = 0

    with open(RECOMMENDATIONS_FILE, "r") as f:
        for line in tqdm(f, desc="Loading"):
            try:
                data = json.loads(line)
                user_id = data["user_id"]
                user_count += 1
                user_ids_in_file.append(user_id)

                # Extract data
                test_watchlist = data[
                    "test_watchlist"
                ]  # Keep full [id, score] pairs for NDCG
                test_anime_ids = [item[0] for item in test_watchlist]
                recs_train = data["recommendations_from_train"]
                recs_full = data["recommendations_from_full"]

                # Store for beyond-accuracy metrics
                recommendations_from_train[user_id] = recs_train
                recommendations_from_full[user_id] = recs_full

                # Calculate accuracy metrics for each k value
                for k in EVAL_K_VALUES:
                    recall = calculate_recall_at_k(recs_train, test_anime_ids, k=k)
                    hit = calculate_hit_at_k(recs_train, test_anime_ids, k=k)
                    ndcg = calculate_ndcg_at_k(recs_train, test_watchlist, k=k)

                    if NEAR:
                        near_recall = calculate_near_recall_at_k(
                            recs_train,
                            test_anime_ids,
                            k=k,
                            indexing=index,
                            similarity_threshold=SIMILARITY_THRESHOLD,
                        )
                        near_hit = calculate_near_hit_at_k(
                            recs_train,
                            test_anime_ids,
                            k=k,
                            indexing=index,
                            similarity_threshold=SIMILARITY_THRESHOLD,
                        )
                    else:
                        near_hit = 0.0
                        near_recall = 0.0

                    metrics_by_k[k]["recall"].append(recall)
                    metrics_by_k[k]["hit"].append(hit)
                    metrics_by_k[k]["ndcg"].append(ndcg)
                    metrics_by_k[k]["near_recall"].append(near_recall)
                    metrics_by_k[k]["near_hit"].append(near_hit)

            except Exception as e:
                print(f"Error loading line: {e}")
                continue

    print(f"       Loaded {user_count} users")

    # Calculate average metrics for each k
    print(f"\n[2/5] Calculating accuracy metrics at multiple k values...")
    print(f"       Using: recommendations_from_train vs test_watchlist")

    avg_metrics_by_k = {}
    for k in EVAL_K_VALUES:
        avg_metrics_by_k[k] = {
            "recall": np.mean(metrics_by_k[k]["recall"])
            if metrics_by_k[k]["recall"]
            else 0.0,
            "hit": np.mean(metrics_by_k[k]["hit"]) if metrics_by_k[k]["hit"] else 0.0,
            "ndcg": np.mean(metrics_by_k[k]["ndcg"])
            if metrics_by_k[k]["ndcg"]
            else 0.0,
            "near_recall": np.mean(metrics_by_k[k]["near_recall"])
            if metrics_by_k[k]["near_recall"]
            else 0.0,
            "near_hit": np.mean(metrics_by_k[k]["near_hit"])
            if metrics_by_k[k]["near_hit"]
            else 0.0,
        }

    # Load ONLY the training data for users in the recommendations file
    print(f"\n[3/5] Loading training data for {len(user_ids_in_file)} users...")
    print(f"       (for novelty calculation)")

    try:
        # Load only data for users present in recommendations file
        df_train_subset = pd.read_parquet(
            "./Dataset/UserAnimeList.parquet",
            filters=[("user_id", "in", user_ids_in_file)],
        )
        print(f"       Loaded {len(df_train_subset)} interactions")
    except Exception as e:
        print(f"       Warning: Could not load training data subset: {e}")
        df_train_subset = None

    # Load catalog and calculate beyond-accuracy metrics
    print(f"\n[4/5] Calculating beyond-accuracy metrics...")
    print(f"       Using: recommendations_from_full")

    try:
        df_catalog = pd.read_csv("./Dataset/AnimeList.csv")
        full_catalog = set(df_catalog["id"].unique())

        catalog_coverage = calculate_catalog_coverage(
            recommendations_from_full, full_catalog
        )
        shannon_entropy = calculate_shannon_entropy(recommendations_from_full)

        # Calculate novelty using ONLY the subset of training data
        if df_train_subset is not None:
            novelty = calculate_novelty(
                recommendations_from_full, training_data=df_train_subset
            )
        else:
            novelty = 0.0

    except Exception as e:
        print(f"       Warning: Could not calculate beyond-accuracy metrics: {e}")
        catalog_coverage = shannon_entropy = novelty = 0.0
        full_catalog = set()

    # Display results
    print(f"\n[5/5] Results:")
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n🎯 ACCURACY METRICS (at different k values)")
    print(f"   Source: recommendations_from_train vs test_watchlist")
    print(f"   Users:  {user_count}")
    print(f"   Fixed test size: {FIXED_TEST_SIZE} items per user")
    print("\n   " + "-" * 60)
    print(
        f"   {'k':<6} {'Recall@k':<18} {'Hit@k':<18} {'NDCG@k':<18} {'Near Recall@k':<18} {'Near Hit@k':<18}"
    )
    print(f"   " + "-" * 100)

    for k in EVAL_K_VALUES:
        metrics = avg_metrics_by_k[k]

        # Prepare strings for each column to make the final f-string readable
        rec_str = f"{metrics['recall']:.4f} ({metrics['recall'] * 100:5.2f}%)"
        hit_str = f"{metrics['hit']:.4f} ({metrics['hit'] * 100:5.2f}%)"
        ndcg_str = f"{metrics['ndcg']:.4f} ({metrics['ndcg'] * 100:5.2f}%)"
        n_rec_str = (
            f"{metrics['near_recall']:.4f} ({metrics['near_recall'] * 100:5.2f}%)"
        )
        n_hit_str = f"{metrics['near_hit']:.4f} ({metrics['near_hit'] * 100:5.2f}%)"

        print(
            f"   {k:<6} "
            f"{rec_str:<18} "
            f"{hit_str:<18} "
            f"{ndcg_str:<18} "
            f"{n_rec_str:<18} "
            f"{n_hit_str:<18}"
        )
    print(f"   " + "-" * 100)

    print(f"\n🌟 BEYOND-ACCURACY METRICS")
    print(f"   Source: recommendations_from_full")
    print(f"   Users:           {len(recommendations_from_full)}")
    print(f"   Catalog size:    {len(full_catalog)}")
    print(f"   Training data:   {len(user_ids_in_file)} users only")
    print(
        f"   Catalog Coverage: {catalog_coverage:.4f} ({catalog_coverage * 100:.2f}%)"
    )
    print(f"   Shannon Entropy:  {shannon_entropy:.4f} bits")
    print(f"   Novelty Score:    {novelty:.4f}")

    print("\n" + "=" * 70)
    print("\n💡 METRIC INTERPRETATIONS:")
    print("   Recall@k:  % of relevant items found in top-k")
    print("   Hit@k:     % of users with ≥1 relevant item in top-k")
    print("   NDCG@k:    Ranking quality (0-1, considers position + scores)")
    print("=" * 70)

    return {
        "accuracy_by_k": avg_metrics_by_k,
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
        print(
            "  python Evaluation.py generate   - Generate recommendations (resumable)"
        )
        print("  python Evaluation.py evaluate   - Evaluate from saved file")
        print("  python Evaluation.py info       - Show file information")
        print("\nConfiguration:")
        print(f"  TOP_K_RECOMMENDATIONS: {TOP_K_RECOMMENDATIONS}")
        print(f"  EVAL_K_VALUES: {EVAL_K_VALUES}")
        print(f"  FIXED_TEST_SIZE: {FIXED_TEST_SIZE}")
        print(f"  MIN_TEST_SCORE: {MIN_TEST_SCORE}")
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
