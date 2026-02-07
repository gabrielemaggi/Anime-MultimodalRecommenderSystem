import gc
import json
import os
import traceback
from collections import Counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm import tqdm

from Libs.User import *


class RecommenderEvaluator:
    """
    A specialized evaluation class for Recommendation Systems.
    Focuses on beyond-accuracy metrics: Catalog Coverage, Distributional Coverage (Entropy)
    and Novelty, plus accuracy metrics: Precision@k and Recall@k.
    """

    def __init__(self, train_df, catalog_items):
        """
        Initialize the evaluator with historical data and the full product list.
        :param train_df: DataFrame containing historical interactions ['user_id', 'anime_id']
        :param catalog_items: List or array of all unique Item IDs available in the system
        """

        self.catalog = set(catalog_items)
        self.n_catalog = len(self.catalog)

        # pre-calculate item popularity for Novelty metric (Self-Information)
        # p(i) = frequency of item i in the training set
        total_interactions = len(train_df)
        item_counts = train_df["anime_id"].value_counts()

        # store as dictionary
        self.item_popularity = (item_counts / total_interactions).to_dict()

        # define the probability for items not seen in training
        self.min_prob = 1 / (total_interactions + 1)

    def evaluate(self, rec_dict, ground_truth_dict=None, k=10):
        """
        Run the evaluation on a set of generated recommendations.
        :param rec_dict: Dictionary {user_id: [list_of_recommended_items]}
        :param ground_truth_dict: Dictionary {user_id: [list_of_relevant_items]} for Precision/Recall
        :param k: Number of top recommendations to consider for Precision@k and Recall@k
        :return: Dictionary containing all computed metrics
        """

        # flatten all recommendation lists into a single stream of items
        all_recs_flattened = [item for sublist in rec_dict.values() for item in sublist]
        unique_recs = set(all_recs_flattened)

        if not all_recs_flattened:
            return {"error": "No recommendations provided for evaluation."}

        metrics = {
            "catalog_coverage": self._calculate_catalog_coverage(unique_recs),
            "shannon_entropy": self._calculate_shannon_entropy(all_recs_flattened),
            "novelty_score": self._calculate_novelty(rec_dict),
        }

        # Add Precision@k and Recall@k if ground truth is provided
        if ground_truth_dict is not None:
            metrics[f"precision@{k}"] = self._calculate_precision_at_k(
                rec_dict, ground_truth_dict, k
            )
            metrics[f"recall@{k}"] = self._calculate_recall_at_k(
                rec_dict, ground_truth_dict, k
            )

        return metrics

    def _calculate_catalog_coverage(self, unique_recs):
        """
        Measures the percentage of items in the catalog recommended at least once.
        Reflects the system's ability to exploit the full inventory.
        """
        recommended_in_catalog = unique_recs.intersection(self.catalog)
        return len(recommended_in_catalog) / self.n_catalog

    def _calculate_shannon_entropy(self, all_recs):
        """
        Measures the uncertainty/diversity of the recommendation distribution.
        Higher entropy indicates a more diverse and balanced recommendation system.
        """
        counts = Counter(all_recs)
        total_recs = len(all_recs)

        # Evaluate probabilities for each recommended item
        probs = np.array([count / total_recs for count in counts.values()])
        return -np.sum(probs * np.log2(probs))

    def _calculate_novelty(self, rec_dict):
        """
        Measures how unexpected the recommended items are to users, focusing on less-known items.
        Novelty = Average across all users of: (1 / |R_u|) * Σ (1 - popularity_score(i)) for i in R_u

        Higher novelty indicates the system is recommending less popular, more obscure items.
        """
        if not rec_dict:
            return 0.0

        # First, calculate the popularity_score correctly
        # popularity_score(i) = (number of users who interacted with i) / (max popularity in catalog)

        # Count interactions per item
        item_interaction_count = {}
        for user_id, rec_list in rec_dict.items():
            for item in rec_list:
                item_interaction_count[item] = item_interaction_count.get(item, 0) + 1

        # Find maximum popularity in the catalog
        max_popularity = (
            max(item_interaction_count.values()) if item_interaction_count else 1.0
        )

        # Calculate normalized popularity_score for each item
        popularity_scores = {}
        for item, count in item_interaction_count.items():
            popularity_scores[item] = count / max_popularity

        user_novelty_scores = []

        # Calculate novelty for each user
        for user_id, rec_list in rec_dict.items():
            if not rec_list:
                continue

            novelty_sum = 0.0

            for item in rec_list:
                # Get the normalized popularity score for this item
                popularity = popularity_scores.get(item, 0.0)

                # Add (1 - popularity_score)
                novelty_sum += 1 - popularity

            # Average for this user: (1 / |R_u|) * Σ (1 - popularity_score(i))
            user_novelty = novelty_sum / len(rec_list)
            user_novelty_scores.append(user_novelty)

        # Return the average novelty across all users
        return np.mean(user_novelty_scores) if user_novelty_scores else 0.0

    def _calculate_precision_at_k(self, rec_dict, ground_truth_dict, k):
        """
        Calculates Precision@k: the proportion of recommended items in the top-k set
        that are relevant to the user.

        Precision@k = (Number of relevant items in top-k) / k

        :param rec_dict: Dictionary {user_id: [list_of_recommended_items]}
        :param ground_truth_dict: Dictionary {user_id: [list_of_relevant_items]}
        :param k: Number of top recommendations to consider
        :return: Average Precision@k across all users
        """
        precision_scores = []

        for user_id, rec_list in rec_dict.items():
            # Get ground truth (relevant items) for this user
            relevant_items = set(ground_truth_dict.get(user_id, []))

            if not relevant_items:
                continue

            # Consider only top-k recommendations
            top_k_recs = set(rec_list[:k])

            # Count how many recommended items are relevant
            relevant_in_top_k = len(top_k_recs.intersection(relevant_items))

            # Precision@k = relevant_in_top_k / k
            precision = relevant_in_top_k / k if k > 0 else 0.0
            precision_scores.append(precision)

        # Return average precision across all users
        return np.mean(precision_scores) if precision_scores else 0.0

    def _calculate_recall_at_k(self, rec_dict, ground_truth_dict, k):
        """
        Calculates Recall@k: the proportion of relevant items that are found in the
        top-k recommendations.

        Recall@k = |{Relevant items} ∩ {Top-k recommended items}| / |{Relevant items}|

        :param rec_dict: Dictionary {user_id: [list_of_recommended_items]}
        :param ground_truth_dict: Dictionary {user_id: [list_of_relevant_items]}
        :param k: Number of top recommendations to consider
        :return: Average Recall@k across all users
        """
        recall_scores = []

        for user_id, rec_list in rec_dict.items():
            # Get ground truth (relevant items) for this user
            relevant_items = set(ground_truth_dict.get(user_id, []))

            if not relevant_items:
                continue

            # Consider only top-k recommendations
            top_k_recs = set(rec_list[:k])

            # Count how many relevant items are in top-k
            relevant_in_top_k = len(top_k_recs.intersection(relevant_items))

            # Recall@k = relevant_in_top_k / total_relevant_items
            recall = relevant_in_top_k / len(relevant_items)
            recall_scores.append(recall)

        # Return average recall across all users
        return np.mean(recall_scores) if recall_scores else 0.0


import gc
import json
import os
import traceback
from contextlib import contextmanager

import numpy as np
import pandas as pd
from recommender_evaluator import RecommenderEvaluator
from tqdm import tqdm

# Assumendo che questi siano già importati nel tuo progetto
# from your_module import User, Indexing


# Constants
OUTPUT_FILE = "./Embeddings/attention_recs_output_with_metrics.jsonl"
ERROR_LOG = "./Embeddings/attention_processing_errors.log"
CHUNK_SIZE = 1000  # Smaller chunks for safety
GC_FREQUENCY = 100  # Garbage collect every N users
TOP_K = 10  # Number of recommendations to generate
EVAL_K = 10  # K for Precision@k and Recall@k evaluation
TRAIN_SPLIT = 0.8  # 50% for training, 50% for testing


@contextmanager
def memory_cleanup():
    """Context manager for automatic memory cleanup"""
    try:
        yield
    finally:
        gc.collect()


def log_error(user_id, error):
    """Log errors to file for debugging"""
    with open(ERROR_LOG, "a") as f:
        f.write(f"User {user_id}: {str(error)}\n")


def get_processed_users():
    """Load set of already processed users"""
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed.add(str(data["user_id"]))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading output file: {e}")
    return processed


def get_unique_users_list(parquet_path):
    """Extract unique user list without loading full dataset"""
    try:
        # using pyarrow directly for minimal memory usage
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        user_ids = []

        # read in batches
        for batch in parquet_file.iter_batches(batch_size=50000, columns=["user_id"]):
            user_ids.extend(batch.column("user_id").to_pylist())

        unique_users = list(set(user_ids))
        del user_ids
        gc.collect()

        return unique_users

    except ImportError:
        # use pandas with column selection if pyarrow not available
        print("PyArrow not available, using pandas (slower)...")
        df = pd.read_parquet(parquet_path, columns=["user_id"])
        unique_users = df["user_id"].unique().tolist()
        del df
        gc.collect()
        return unique_users


def split_watchlist(watchlist, train_ratio=0.5, min_items=5):
    """
    Split watchlist into train and test sets.

    :param watchlist: List of [anime_id, score] pairs
    :param train_ratio: Ratio of items to use for training (default 0.5)
    :param min_items: Minimum number of items required to perform split
    :return: (train_list, test_list) or (watchlist, []) if too few items
    """
    if len(watchlist) < min_items:
        # Se ci sono troppo pochi item, usa tutto per il training
        return watchlist, []

    # Shuffle per randomizzare
    np.random.seed(42)  # Per riproducibilità
    shuffled = watchlist.copy()
    np.random.shuffle(shuffled)

    # Split
    split_point = int(len(shuffled) * train_ratio)
    train_list = shuffled[:split_point]
    test_list = shuffled[split_point:]

    for anime in test_list:
        if anime[1] < 6:
            test_list.remove(anime)

    # opzionale riordinare test
    test_list.sort(key=lambda x: x[1], reverse=True)

    if len(test_list) < EVAL_K:
        return train_list, []

    return train_list, test_list[:EVAL_K]


def calculate_user_metrics(recommendations, ground_truth, k=10):
    """
    Calculate Precision@k and Recall@k for a single user.

    :param recommendations: List of recommended anime IDs
    :param ground_truth: List of relevant anime IDs (test set)
    :param k: Number of top recommendations to consider
    :return: Dictionary with precision and recall
    """
    if not ground_truth or not recommendations:
        return {"precision": 0.0, "recall": 0.0}

    # Consider only top-k recommendations
    top_k_recs = set(recommendations[:k])
    relevant_items = set(ground_truth)

    # Calculate metrics
    relevant_in_top_k = len(top_k_recs.intersection(relevant_items))

    precision = relevant_in_top_k / k if k > 0 else 0.0
    recall = relevant_in_top_k / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return {"precision": precision, "recall": recall}


def process_single_user(
    user_id, user_data, index, train_split=0.5, top_k=30, eval_k=10
):
    """
    Process a single user with train/test split and metric calculation.

    :return: Dictionary with all user data and metrics, or None if processing failed
    """
    try:
        # Prepare full watchlist
        if "my_score" in user_data.columns:
            full_watchlist = user_data[["anime_id", "my_score"]].values.tolist()
        else:
            full_watchlist = [
                [int(anime_id), 0] for anime_id in user_data["anime_id"].values
            ]

        if not full_watchlist:
            return None

        # Split watchlist into train and test
        train_watchlist, test_watchlist = split_watchlist(
            full_watchlist, train_ratio=train_split, min_items=5
        )

        # Skip users without test set (too few items)
        if not test_watchlist:
            return None

        # Create user instance with ONLY training data
        u = User(user_id, watched_list=train_watchlist)

        # If user doesn't have valid watch list
        if not u.get_watchList():
            del u
            return None

        # Generate recommendations based on training data
        u.findCentersOfClusters(index)
        recs_dicts = u.get_nearest_anime_from_clusters(index, top_k=top_k)

        # Extract anime IDs from recommendations
        rec_ids = [int(anime["id"]) for anime in recs_dicts]

        # Extract test set anime IDs (ground truth)
        test_anime_ids = [int(item[0]) for item in test_watchlist]

        # Calculate Precision@k and Recall@k
        metrics = calculate_user_metrics(rec_ids, test_anime_ids, k=eval_k)

        # Prepare result
        result = {
            "user_id": str(user_id),
            "watchlist_train": [
                [int(item[0]), float(item[1])] for item in train_watchlist
            ],
            "watchlist_test": [
                [int(item[0]), float(item[1])] for item in test_watchlist
            ],
            "recommendations": rec_ids,
            f"precision@{eval_k}": metrics["precision"],
            f"recall@{eval_k}": metrics["recall"],
        }

        # Cleanup
        del u
        del recs_dicts
        del train_watchlist
        del test_watchlist

        return result

    except Exception as e:
        raise Exception(f"User processing failed: {str(e)}")


def generate_recommendations_safe():
    """Main generation function with comprehensive error handling and metrics"""

    print("=" * 60)
    print("GENERATING RECOMMENDATIONS WITH TRAIN/TEST SPLIT")
    print("=" * 60)
    print(
        f"Train Split: {TRAIN_SPLIT * 100:.0f}% | Test Split: {(1 - TRAIN_SPLIT) * 100:.0f}%"
    )
    print(f"Recommendations: Top-{TOP_K} | Evaluation: @{EVAL_K}")

    # Check resume point
    print("\n[1/5] Checking resume point...")
    processed_users = get_processed_users()
    print(f"       Users already processed: {len(processed_users)}")

    # Load vector database
    print("\n[2/5] Loading Vector Database...")
    try:
        index = Indexing()
        index.load_vector_database()
        print("       Vector DB loaded successfully")
    except Exception as e:
        print(f"       ERROR: {e}")
        return

    # Get user list
    print("\n[3/5] User list extraction...")
    try:
        unique_users = get_unique_users_list("./Dataset/UserAnimeList.parquet")
        print(f"       Found {len(unique_users)} total users")

        # Filter already processed
        users_to_process = [u for u in unique_users if str(u) not in processed_users]
        print(f"       To process: {len(users_to_process)} users")

        if len(users_to_process) == 0:
            print("\n✓ All users already processed!")
            return

    except Exception as e:
        print(f"       ERROR: {e}")
        return

    # Process in chunks
    print("\n[4/5] Starting batch user processing...")
    print(f"       Batch size: {CHUNK_SIZE} users")
    print(f"       Garbage collection every: {GC_FREQUENCY} users")

    total_chunks = (len(users_to_process) + CHUNK_SIZE - 1) // CHUNK_SIZE
    successful_count = 0
    error_count = 0
    skipped_count = 0  # Users with too few items for split

    # Metrics aggregation
    total_precision = 0.0
    total_recall = 0.0

    with open(OUTPUT_FILE, "a", buffering=1) as f_out:  # Line buffering
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, len(users_to_process))
            user_chunk = users_to_process[chunk_start:chunk_end]

            print(
                f"\n>>> Batch {chunk_idx + 1}/{total_chunks} (users {chunk_start + 1}-{chunk_end})"
            )

            # Load data for this batch only
            try:
                with memory_cleanup():
                    df_chunk = pd.read_parquet(
                        "./Dataset/UserAnimeList.parquet",
                        filters=[("user_id", "in", user_chunk)],
                    )

                    # Fix column names (just in case)
                    if (
                        "anime_id" not in df_chunk.columns
                        and "anime_id" in df_chunk.columns
                    ):
                        df_chunk = df_chunk.rename(columns={"anime_id": "anime_id"})
                    if (
                        "my_score" not in df_chunk.columns
                        and "score" in df_chunk.columns
                    ):
                        df_chunk = df_chunk.rename(columns={"score": "my_score"})

                    # Check if anime_id exists
                    if "anime_id" not in df_chunk.columns:
                        print(
                            f"       WARNING: 'anime_id' column missing, skipping batch"
                        )
                        continue

                    user_groups = df_chunk.groupby("user_id")

                    # Process each user
                    pbar = tqdm(
                        user_chunk, desc=f"    Batch {chunk_idx + 1}", leave=False
                    )

                    for i, user_id in enumerate(pbar):
                        try:
                            # If user has no data, skip
                            if user_id not in user_groups.groups:
                                skipped_count += 1
                                continue

                            user_data = user_groups.get_group(user_id)

                            # Process user with train/test split
                            result = process_single_user(
                                user_id,
                                user_data,
                                index,
                                train_split=TRAIN_SPLIT,
                                top_k=TOP_K,
                                eval_k=EVAL_K,
                            )

                            if result:
                                # Save to file
                                f_out.write(json.dumps(result) + "\n")
                                successful_count += 1

                                # Accumulate metrics
                                total_precision += result[f"precision@{EVAL_K}"]
                                total_recall += result[f"recall@{EVAL_K}"]
                            else:
                                skipped_count += 1

                            # Clean up user data from memory
                            del user_data

                            # Garbage collection every N users
                            if (i + 1) % GC_FREQUENCY == 0:
                                gc.collect()

                        except Exception as e:
                            error_count += 1
                            log_error(user_id, e)
                            pbar.set_postfix({"errors": error_count}, refresh=False)
                            continue

                    pbar.close()

                    # Update progress
                    avg_precision = (
                        total_precision / successful_count
                        if successful_count > 0
                        else 0
                    )
                    avg_recall = (
                        total_recall / successful_count if successful_count > 0 else 0
                    )
                    print(
                        f"     Batch completed - Successes: {successful_count}, "
                        f"Errors: {error_count}, Skipped: {skipped_count}"
                    )
                    print(
                        f"     Running Avg - Precision@{EVAL_K}: {avg_precision:.4f}, "
                        f"Recall@{EVAL_K}: {avg_recall:.4f}"
                    )

            except Exception as e:
                print(f"     ERROR loading batch: {e}")
                print(f"     Process will restart from next batch...")
                traceback.print_exc()
                continue

    # Step 5: Results and review
    print("\n" + "=" * 60)
    print("[5/5] PROCESS COMPLETED")
    print("=" * 60)
    print(f"✓ Generated recommendations: {successful_count}")
    print(f"✗ Total errors: {error_count}")
    print(f"⊘ Skipped (too few items): {skipped_count}")

    if successful_count > 0:
        avg_precision = total_precision / successful_count
        avg_recall = total_recall / successful_count
        print(f"\n📊 AVERAGE METRICS:")
        print(f"   Precision@{EVAL_K}: {avg_precision:.4f}")
        print(f"   Recall@{EVAL_K}: {avg_recall:.4f}")

    print(f"\n💾 Data saved in: {OUTPUT_FILE}")
    if error_count > 0:
        print(f"⚠  Error log: {ERROR_LOG}")
    print("=" * 60)


def evaluate_from_file(
    recs_file="./Embeddings/attention_recs_output_with_metrics.jsonl",
):
    """Evaluate and aggregate metrics from generated recommendations file"""

    print("\n" + "=" * 60)
    print("RECOMMENDATION EVALUATION FROM FILE")
    print("=" * 60)

    print("\n[1/2] Reading recommendations and metrics...")
    all_recommendations = {}
    all_ground_truths = {}
    precision_scores = []
    recall_scores = []

    try:
        with open(recs_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    user_id = data["user_id"]

                    all_recommendations[user_id] = data["recommendations"]
                    all_ground_truths[user_id] = [
                        item[0] for item in data["watchlist_test"]
                    ]

                    # Extract individual metrics
                    precision_key = [
                        k for k in data.keys() if k.startswith("precision@")
                    ]
                    recall_key = [k for k in data.keys() if k.startswith("recall@")]

                    if precision_key:
                        precision_scores.append(data[precision_key[0]])
                    if recall_key:
                        recall_scores.append(data[recall_key[0]])

                except Exception as e:
                    continue

        print(f"       Loaded data for {len(all_recommendations)} users")

    except FileNotFoundError:
        print(f"       ERROR: Recommendation file '{recs_file}' not found")
        return

    if not all_recommendations:
        print("       WARNING: No recommendations found")
        return

    print("\n[2/2] Calculating aggregated metrics...")

    # Calculate average accuracy metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0

    # Calculate beyond-accuracy metrics using the evaluator
    print("       Loading dataset for beyond-accuracy metrics...")
    try:
        df_interactions = pd.read_parquet("./Dataset/UserAnimeList.parquet")
        if "anime_id" in df_interactions.columns:
            df_interactions = df_interactions.rename(columns={"anime_id": "anime_id"})

        df_catalog = pd.read_csv("./Dataset/AnimeList.csv")
        full_catalog_ids = df_catalog["id"].unique()

        evaluator = RecommenderEvaluator(df_interactions, full_catalog_ids)
        beyond_metrics = evaluator.evaluate(all_recommendations)

    except Exception as e:
        print(f"       Warning: Could not calculate beyond-accuracy metrics: {e}")
        beyond_metrics = {}

    # Display results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    print("\n🎯 ACCURACY METRICS (per-user average):")
    print(
        f"   Precision@{EVAL_K}:        {avg_precision:.4f} ({avg_precision * 100:.2f}%)"
    )
    print(f"   Recall@{EVAL_K}:           {avg_recall:.4f} ({avg_recall * 100:.2f}%)")

    if beyond_metrics:
        print("\n🌟 BEYOND-ACCURACY METRICS:")
        print(
            f"   Catalog Coverage:      {beyond_metrics.get('catalog_coverage', 0):.2%}"
        )
        print(
            f"   Shannon Entropy:       {beyond_metrics.get('shannon_entropy', 0):.4f} bits"
        )
        print(f"   Novelty Score:         {beyond_metrics.get('novelty_score', 0):.4f}")

    print("\n" + "=" * 60)


def analyze_sample_users(
    recs_file="./Embeddings/attention_recs_output_with_metrics.jsonl", n_samples=5
):
    """Display detailed analysis for sample users"""

    print("\n" + "=" * 60)
    print(f"SAMPLE USER ANALYSIS (showing {n_samples} users)")
    print("=" * 60)

    try:
        with open(recs_file, "r") as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break

                data = json.loads(line)
                user_id = data["user_id"]

                precision_key = [k for k in data.keys() if k.startswith("precision@")][
                    0
                ]
                recall_key = [k for k in data.keys() if k.startswith("recall@")][0]

                print(f"\n👤 User ID: {user_id}")
                print(f"   Train Set Size: {len(data['watchlist_train'])} anime")
                print(f"   Test Set Size: {len(data['watchlist_test'])} anime")
                print(f"   Recommendations: {len(data['recommendations'])} anime")
                print(f"   {precision_key}: {data[precision_key]:.4f}")
                print(f"   {recall_key}: {data[recall_key]:.4f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        # Generate recommendations with train/test split and metrics
        generate_recommendations_safe()

        # Evaluate aggregated results
        print("\n")
        evaluate_from_file()

        # Show sample user details
        print("\n")
        analyze_sample_users(n_samples=5)

    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt - Progress saved")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
