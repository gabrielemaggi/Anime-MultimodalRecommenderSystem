import gc
import json
import os
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm import tqdm

# Force CPU-only operation - disable CUDA before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide all CUDA devices
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Try to set device to CPU for common ML libraries
try:
    import torch

    torch.set_default_device("cpu")
except ImportError:
    pass

try:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
except ImportError:
    pass

from Libs.User import *


class RecommenderEvaluator:
    """
    A specialized evaluation class for Recommendation Systems.
    Focuses on beyond-accuracy metrics: Catalog Coverage, Distributional Coverage (Gini/Entropy),
    Novelty (Inverse Popularity), and Serendipity.
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

    def evaluate(self, rec_dict):
        """
        Run the evaluation on a set of generated recommendations.
        :param rec_dict: Dictionary {user_id: [list_of_recommended_items]}
        :return: Dictionary containing all computed metrics
        """
        # flatten all recommendation lists into a single stream of items
        all_recs_flattened = [item for sublist in rec_dict.values() for item in sublist]
        unique_recs = set(all_recs_flattened)

        if not all_recs_flattened:
            return {"error": "No recommendations provided for evaluation."}

        return {
            "catalog_coverage": self._calculate_catalog_coverage(unique_recs),
            "shannon_entropy": self._calculate_shannon_entropy(all_recs_flattened),
            "novelty_score": self._calculate_novelty(rec_dict),
        }

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

        user_novelty_scores = []

        # Calculate novelty for each user
        for user_id, rec_list in rec_dict.items():
            if not rec_list:
                continue

            novelty_sum = 0.0
            all_popularities = []

            for item in rec_list:
                # Get the popularity score for this item
                if item in self.item_popularity:
                    popularity = self.item_popularity[item]
                else:
                    popularity = self.min_prob

                # Add (1 - popularity_score)
                novelty_sum += 1 - popularity
                all_popularities.append(popularity)

            # Average for this user: (1 / |R_u|) * Σ (1 - popularity_score(i))
            user_novelty = novelty_sum / len(rec_list)
            user_novelty_scores.append(user_novelty)

        # Return the average novelty across all users
        return np.mean(user_novelty_scores) if user_novelty_scores else 0.0


# Constants
OUTPUT_FILE = "./Embeddings/recs_output.jsonl"
ERROR_LOG = "./Embeddings/processing_errors.log"
CHUNK_SIZE = 5000
NUM_THREADS = 8  # Number of concurrent threads (adjust based on CPU)


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


def process_single_user(user_id, user_data_dict, index):
    """
    Process a single user and return recommendations
    """
    try:
        # Reconstruct user data from dictionary
        watched_list = user_data_dict.get("watched_list", [])

        if not watched_list:
            return None

        # create user instance
        u = User(user_id, watched_list=watched_list)

        # if user have valid watch list
        if not u.get_watchList():
            del u
            return None

        # evaluate clusters and recommendations
        u.findCentersOfClusters()
        recs_dicts = u.get_nearest_anime_from_clusters(index, top_k=30)

        # extract anime id
        rec_ids = [int(anime["id"]) for anime in recs_dicts]

        # cleanup for memory management
        del u
        del recs_dicts

        return rec_ids if rec_ids else None
    except Exception as e:
        raise Exception(f"User processing failed: {str(e)}")


def process_user_threaded(user_batch_data, index):
    """
    Thread worker function for parallel processing
    :param user_batch_data: tuple of (user_id, user_data_dict)
    :param index: shared index object
    :return: tuple of (user_id, recommendations or None, error_msg or None)
    """
    user_id, user_data_dict = user_batch_data
    try:
        rec_ids = process_single_user(user_id, user_data_dict, index)
        return (user_id, rec_ids, None)
    except Exception as e:
        return (user_id, None, str(e))


def prepare_user_data_batch(df_chunk):
    """
    Prepare user data for parallel processing
    Returns list of (user_id, user_data_dict) tuples
    """
    user_groups = df_chunk.groupby("user_id")
    batch_data = []

    for user_id in df_chunk["user_id"].unique():
        try:
            user_data = user_groups.get_group(user_id)

            # Prepare watched list
            if "my_score" in user_data.columns:
                watched_list = user_data[["anime_id", "my_score"]].values.tolist()
            else:
                watched_list = [
                    [int(anime_id), 0] for anime_id in user_data["anime_id"].values
                ]

            user_data_dict = {"watched_list": watched_list}

            batch_data.append((user_id, user_data_dict))
        except Exception as e:
            continue

    return batch_data


def generate_recommendations_threaded():
    """Main generation function using threading (no multiprocessing issues)"""
    print("=" * 60)
    print("GENERATING RECOMMENDATIONS (THREADED - CPU ONLY)")
    print("=" * 60)
    print(f"Using {NUM_THREADS} threads")
    print("⚠️  CUDA disabled - running on CPU only")
    print("ℹ️  Using threading instead of multiprocessing (CUDA-safe)")

    # check resume point
    print("\n[1/5] Checking resume point...")
    processed_users = get_processed_users()
    print(f"       Users already processed: {len(processed_users)}")

    # load vector database
    print("\n[2/5] Loading Vector Database...")
    try:
        index = Indexing()
        index.load_vector_database()
        print("       Vector DB loaded")
    except Exception as e:
        print(f"       Error: {e}")
        return

    # get user list
    print("\n[3/5] User list extraction...")
    try:
        unique_users = get_unique_users_list("./Dataset/UserAnimeList.parquet")
        print(f"       Found {len(unique_users)} users")

        # filter already processed
        users_to_process = [u for u in unique_users if str(u) not in processed_users]
        print(f"       To process: {len(users_to_process)} users")

        if len(users_to_process) == 0:
            print("\n✓ All users already processed")
            return
    except Exception as e:
        print(f"       Error: {e}")
        return

    # process in chunks with threaded workers
    print("\n[4/5] Starting threaded batch processing...")
    print(f"      Chunk size: {CHUNK_SIZE} users")
    print(f"      Threads: {NUM_THREADS}")

    total_chunks = (len(users_to_process) + CHUNK_SIZE - 1) // CHUNK_SIZE
    successful_count = 0
    error_count = 0

    with open(OUTPUT_FILE, "a", buffering=1) as f_out:
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, len(users_to_process))
            user_chunk = users_to_process[chunk_start:chunk_end]

            print(
                f"\n>>> Chunk {chunk_idx + 1}/{total_chunks} (users {chunk_start + 1}-{chunk_end})"
            )

            # load data for this batch only
            try:
                with memory_cleanup():
                    df_chunk = pd.read_parquet(
                        "./Dataset/UserAnimeList.parquet",
                        filters=[("user_id", "in", user_chunk)],
                    )

                    # fixing columns name (just in case)
                    if (
                        "anime_id" in df_chunk.columns
                        and "anime_id" not in df_chunk.columns
                    ):
                        df_chunk = df_chunk.rename(columns={"anime_id": "anime_id"})
                    if (
                        "my_score" not in df_chunk.columns
                        and "score" in df_chunk.columns
                    ):
                        df_chunk = df_chunk.rename(columns={"score": "my_score"})

                    # if anime not exists
                    if "anime_id" not in df_chunk.columns:
                        print(f"      Warning: 'anime_id' column missing, skipping")
                        continue

                    # Prepare batch data for parallel processing
                    print("      Preparing batch data...")
                    batch_data = prepare_user_data_batch(df_chunk)

                    if not batch_data:
                        print("      No valid users in this chunk")
                        continue

                    # Process with threads (shares memory, no CUDA fork issues)
                    print(f"      Processing {len(batch_data)} users with threads...")

                    chunk_success = 0
                    chunk_errors = 0

                    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                        # Submit all tasks
                        future_to_user = {
                            executor.submit(
                                process_user_threaded, user_data, index
                            ): user_data[0]
                            for user_data in batch_data
                        }

                        # Process completed tasks with progress bar
                        with tqdm(
                            total=len(batch_data),
                            desc=f"    Chunk {chunk_idx + 1}",
                            leave=False,
                        ) as pbar:
                            for future in as_completed(future_to_user):
                                try:
                                    user_id, rec_ids, error_msg = future.result()

                                    if error_msg:
                                        chunk_errors += 1
                                        log_error(user_id, error_msg)
                                    elif rec_ids:
                                        record = {
                                            "user_id": str(user_id),
                                            "recommendations": rec_ids,
                                        }
                                        f_out.write(json.dumps(record) + "\n")
                                        chunk_success += 1

                                    pbar.update(1)
                                    pbar.set_postfix(
                                        {
                                            "success": chunk_success,
                                            "errors": chunk_errors,
                                        }
                                    )

                                except Exception as e:
                                    chunk_errors += 1
                                    user_id = future_to_user[future]
                                    log_error(user_id, str(e))
                                    pbar.update(1)

                    successful_count += chunk_success
                    error_count += chunk_errors

                    print(
                        f"     Chunk completed - Successes: {chunk_success}, Errors: {chunk_errors}"
                    )
                    print(
                        f"     Total so far - Successes: {successful_count}, Errors: {error_count}"
                    )

            except Exception as e:
                print(f"     Error loading chunk: {e}")
                print(f"     Process will restart from next chunk...")
                traceback.print_exc()
                continue

    # step 5 results and review
    print("\n" + "=" * 60)
    print("[5/5] PROCESS COMPLETED")
    print("=" * 60)
    print(f"✓ Generated recommendations: {successful_count}")
    print(f"✗ Total errors: {error_count}")
    print(f"📁 Data saved in: {OUTPUT_FILE}")
    if error_count > 0:
        print(f"📋 Error log: {ERROR_LOG}")

    # Calculate success rate
    total_attempted = successful_count + error_count
    if total_attempted > 0:
        success_rate = (successful_count / total_attempted) * 100
        print(f"📊 Success rate: {success_rate:.2f}%")

    print("=" * 60)


def evaluate_from_file(recs_file="recs_output.jsonl"):
    """Evaluate generated recommendations"""
    print("\n" + "=" * 60)
    print("RECOMMENDATION EVALUATION")
    print("=" * 60)

    print("\n[1/3] Loading the dataset...")
    try:
        df_interactions = pd.read_parquet("./Dataset/UserAnimeList.parquet")
        if "anime_id" in df_interactions.columns:
            df_interactions = df_interactions.rename(columns={"anime_id": "anime_id"})

        df_catalog = pd.read_csv("./AnimeList.csv")
        full_catalog_ids = df_catalog["id"].unique()
        print("       Dataset loaded")
    except Exception as e:
        print(f"       Error: {e}")
        return

    print("\n[2/3] Reading recommendations...")
    all_recommendations = {}
    try:
        with open(recs_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_recommendations[data["user_id"]] = data["recommendations"]
                except:
                    continue
        print(f"       Loaded recommendations for {len(all_recommendations)} users")
    except FileNotFoundError:
        print(f"       Recommendation file {recs_file} not found")
        return

    if not all_recommendations:
        print("       No recommendations found")
        return

    print("\n[3/3] Evaluating metrics...")
    evaluator = RecommenderEvaluator(df_interactions, full_catalog_ids)
    metrics = evaluator.evaluate(all_recommendations)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Catalog Coverage:          {metrics['catalog_coverage']:.2%}")
    print(f"Distributional Coverage:   {metrics['shannon_entropy']:.4f} bits")
    print(f"Novelty Score:             {metrics['novelty_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        generate_recommendations_threaded()
        # Uncomment to run evaluation after generation
        print("\n")
        evaluate_from_file()
    except KeyboardInterrupt:
        print("\n\n⚠ Keyboard interrupt - progress saved")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
