import numpy as np
import pandas as pd
from collections import Counter
from indexing_db import *
from User import *

class RecommenderEvaluator:
    """
    A specialized evaluation class for Recommendation Systems.
    Focuses on beyond-accuracy metrics: Catalog Coverage, Distributional Coverage (Gini/Entropy),
    and Novelty.
    """

    def __init__(self, train_df, catalog_items):
        """
        Initialize the evaluator with historical data and the full product list.

        :param train_df: DataFrame containing historical interactions ['user_id', 'anime_id']
        :param catalog_items: List or array of all unique Item IDs available in the system
        """
        self.catalog = set(catalog_items)
        self.n_catalog = len(self.catalog)

        # Pre-calculate item popularity for Novelty metric (Self-Information)
        # p(i) = frequency of item i in the training set
        total_interactions = len(train_df)
        item_counts = train_df['anime_id'].value_counts()

        # Store as dictionary for O(1) lookup during evaluation
        self.item_popularity = (item_counts / total_interactions).to_dict()

        # Define a fallback probability for items not seen in training (Laplace smoothing)
        self.min_prob = 1 / (total_interactions + 1)

    def evaluate(self, rec_dict):
        """
        Run the evaluation suite on a set of generated recommendations.

        :param rec_dict: Dictionary {user_id: [list_of_recommended_items]}
        :return: Dictionary containing all computed metrics
        """
        # Flatten all recommendation lists into a single stream of items
        all_recs_flattened = [item for sublist in rec_dict.values() for item in sublist]
        unique_recs = set(all_recs_flattened)

        if not all_recs_flattened:
            return {"error": "No recommendations provided for evaluation."}

        return {
            "catalog_coverage": self._calculate_catalog_coverage(unique_recs),
            "gini_index": self._calculate_gini(all_recs_flattened),
            "shannon_entropy": self._calculate_shannon_entropy(all_recs_flattened),
            "novelty_score": self._calculate_novelty(all_recs_flattened)
        }

    def _calculate_catalog_coverage(self, unique_recs):
        """
        Measures the percentage of items in the catalog recommended at least once.
        Reflects the system's ability to exploit the full inventory.
        """
        recommended_in_catalog = unique_recs.intersection(self.catalog)
        return len(recommended_in_catalog) / self.n_catalog

    def _calculate_gini(self, all_recs):
        """
        Calculates the Gini Index for recommendation distribution.
        Values near 0: Uniform distribution (fair system).
        Values near 1: Highly skewed distribution (popularity bias).
        """
        n = self.n_catalog
        counts = Counter(all_recs)

        # Map frequencies to all items in the catalog (unrecommended items get 0)
        frequencies = np.array([counts.get(item, 0) for item in self.catalog])
        frequencies = np.sort(frequencies)

        # Mathematical Gini formula implementation
        index = np.arange(1, n + 1)
        sum_freq = np.sum(frequencies)

        if sum_freq == 0: return 1.0  # Maximum inequality if nothing is recommended

        gini = (np.sum((2 * index - n - 1) * frequencies)) / (n * sum_freq)
        return gini

    def _calculate_shannon_entropy(self, all_recs):
        """
        Measures the uncertainty/diversity of the recommendation distribution.
        Higher entropy indicates a more diverse and balanced recommendation system.
        """
        counts = Counter(all_recs)
        total_recs = len(all_recs)
        # Calculate probabilities for each recommended item
        probs = [count / total_recs for count in counts.values()]
        return -np.sum(probs * np.log2(probs))

    def _calculate_novelty(self, all_recs):
        """
        Calculates the average Self-Information of recommended items.
        A high score means the system suggests 'long-tail' (rare) items
        rather than just popular ones.
        """
        # Calculate -log2(p(i)) for every recommended item
        self_info = [
            -np.log2(self.item_popularity.get(item, self.min_prob))
            for item in all_recs
        ]
        return np.mean(self_info)


import pandas as pd
import json
import os
import gc
import traceback
from tqdm import tqdm
from contextlib import contextmanager

# Constants
OUTPUT_FILE = "recs_output.jsonl"
ERROR_LOG = "processing_errors.log"
CHUNK_SIZE = 5000  # Smaller chunks for safety
GC_FREQUENCY = 20  # Garbage collect every N users


@contextmanager
def memory_cleanup():
    """Context manager for automatic memory cleanup"""
    try:
        yield
    finally:
        gc.collect()


def log_error(user_id, error):
    """Log errors to file for debugging"""
    with open(ERROR_LOG, 'a') as f:
        f.write(f"User {user_id}: {str(error)}\n")


def get_processed_users():
    """Load set of already processed users"""
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed.add(str(data['user_id']))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading output file: {e}")
    return processed


def get_unique_users_list(parquet_path):
    """Extract unique user list without loading full dataset"""
    try:
        # Method 1: Try using pyarrow directly for minimal memory
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        user_ids = []

        # Read in batches
        for batch in parquet_file.iter_batches(batch_size=50000, columns=['user_id']):
            user_ids.extend(batch.column('user_id').to_pylist())

        unique_users = list(set(user_ids))
        del user_ids
        gc.collect()

        return unique_users

    except ImportError:
        # Fallback: Use pandas with column selection
        print("PyArrow not available, using pandas (slower)...")
        df = pd.read_parquet(parquet_path, columns=['user_id'])
        unique_users = df['user_id'].unique().tolist()
        del df
        gc.collect()
        return unique_users


def process_single_user(user_id, user_data, index):
    """Process a single user and return recommendations"""
    try:
        # Prepare watched list
        if 'my_score' in user_data.columns:
            watched_list = user_data[['anime_id', 'my_score']].values.tolist()
        else:
            watched_list = [[int(anime_id), 0] for anime_id in user_data['anime_id'].values]

        if not watched_list:
            return None

        # Create user instance
        u = User(user_id, watched_list=watched_list)

        # Check valid watch list
        if not u.get_watchList():
            del u
            return None

        # Calculate clusters and recommendations
        u.findCentersOfClusters()
        recs_dicts = u.get_nearest_anime_from_clusters(index, top_k=10)

        # Extract IDs
        rec_ids = [int(anime['id']) for anime in recs_dicts]

        # Cleanup
        del u
        del recs_dicts

        return rec_ids if rec_ids else None

    except Exception as e:
        raise Exception(f"User processing failed: {str(e)}")


def generate_recommendations_safe():
    """Main generation function with comprehensive error handling"""

    print("=" * 60)
    print("GENERAZIONE RACCOMANDAZIONI - VERSIONE ULTRA-SAFE")
    print("=" * 60)

    # Step 1: Check resume point
    print("\n[1/5] Controllo stato precedente...")
    processed_users = get_processed_users()
    print(f"      ✓ Utenti già processati: {len(processed_users)}")

    # Step 2: Load vector database
    print("\n[2/5] Caricamento Vector Database...")
    try:
        index = Indexing()
        index.load_vector_database()
        print("      ✓ Vector DB caricato con successo")
    except Exception as e:
        print(f"      ✗ ERRORE CRITICO: {e}")
        return

    # Step 3: Get user list
    print("\n[3/5] Estrazione lista utenti...")
    try:
        unique_users = get_unique_users_list("./Resources/UserAnimeList.parquet")
        print(f"      ✓ Trovati {len(unique_users)} utenti totali")

        # Filter already processed
        users_to_process = [u for u in unique_users if str(u) not in processed_users]
        print(f"      ✓ Da processare: {len(users_to_process)} utenti")

        if len(users_to_process) == 0:
            print("\n✓ Tutti gli utenti già processati!")
            return

    except Exception as e:
        print(f"      ✗ ERRORE: {e}")
        return

    # Step 4: Process in chunks
    print("\n[4/5] Inizio elaborazione a chunk...")
    print(f"      Dimensione chunk: {CHUNK_SIZE} utenti")
    print(f"      Garbage collection ogni {GC_FREQUENCY} utenti")

    total_chunks = (len(users_to_process) + CHUNK_SIZE - 1) // CHUNK_SIZE
    successful_count = 0
    error_count = 0

    with open(OUTPUT_FILE, 'a', buffering=1) as f_out:  # Line buffering

        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, len(users_to_process))
            user_chunk = users_to_process[chunk_start:chunk_end]

            print(f"\n>>> Chunk {chunk_idx + 1}/{total_chunks} (utenti {chunk_start + 1}-{chunk_end})")

            # Load data for this chunk only
            try:
                with memory_cleanup():
                    df_chunk = pd.read_parquet(
                        "./Resources/UserAnimeList.parquet",
                        filters=[('user_id', 'in', user_chunk)]
                    )

                    # Standardize columns
                    if 'anime_id' in df_chunk.columns and 'anime_id' not in df_chunk.columns:
                        df_chunk = df_chunk.rename(columns={'anime_id': 'anime_id'})
                    if 'my_score' not in df_chunk.columns and 'score' in df_chunk.columns:
                        df_chunk = df_chunk.rename(columns={'score': 'my_score'})

                    # Ensure anime_id exists
                    if 'anime_id' not in df_chunk.columns:
                        print(f"      ⚠ Warning: 'anime_id' column missing, skipping chunk")
                        continue

                    user_groups = df_chunk.groupby('user_id')

                    # Process each user
                    pbar = tqdm(user_chunk, desc=f"    Chunk {chunk_idx + 1}", leave=False)

                    for i, user_id in enumerate(pbar):
                        try:
                            # Check if user has data
                            if user_id not in user_groups.groups:
                                continue

                            user_data = user_groups.get_group(user_id)

                            # Process user
                            rec_ids = process_single_user(user_id, user_data, index)

                            if rec_ids:
                                # Save immediately
                                record = {"user_id": str(user_id), "recommendations": rec_ids}
                                f_out.write(json.dumps(record) + "\n")
                                successful_count += 1

                            # Clean up user data
                            del user_data

                            # Periodic garbage collection
                            if (i + 1) % GC_FREQUENCY == 0:
                                gc.collect()

                        except Exception as e:
                            error_count += 1
                            log_error(user_id, e)
                            pbar.set_postfix({"errors": error_count}, refresh=False)
                            continue

                    pbar.close()

                    # Update progress
                    print(f"    ✓ Chunk completato - Successi: {successful_count}, Errori: {error_count}")

            except Exception as e:
                print(f"    ✗ Errore caricamento chunk: {e}")
                print(f"    Continuando con chunk successivo...")
                traceback.print_exc()
                continue

    # Step 5: Final report
    print("\n" + "=" * 60)
    print("[5/5] COMPLETAMENTO")
    print("=" * 60)
    print(f"✓ Raccomandazioni generate: {successful_count}")
    print(f"✗ Errori riscontrati: {error_count}")
    print(f"📁 Output salvato in: {OUTPUT_FILE}")
    if error_count > 0:
        print(f"📋 Log errori in: {ERROR_LOG}")
    print("=" * 60)


def evaluate_from_file(recs_file="recs_output.jsonl"):
    """Evaluate generated recommendations"""

    print("\n" + "=" * 60)
    print("VALUTAZIONE RACCOMANDAZIONI")
    print("=" * 60)

    print("\n[1/3] Caricamento dataset...")
    try:
        df_interactions = pd.read_parquet("./Resources/UserAnimeList.parquet")
        if 'anime_id' in df_interactions.columns:
            df_interactions = df_interactions.rename(columns={'anime_id': 'anime_id'})

        df_catalog = pd.read_csv("./AnimeList.csv")
        full_catalog_ids = df_catalog['id'].unique()
        print("      ✓ Dataset caricati")

    except Exception as e:
        print(f"      ✗ Errore: {e}")
        return

    print("\n[2/3] Lettura raccomandazioni...")
    all_recommendations = {}
    try:
        with open(recs_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_recommendations[data['user_id']] = data['recommendations']
                except:
                    continue
        print(f"      ✓ Caricate raccomandazioni per {len(all_recommendations)} utenti")

    except FileNotFoundError:
        print(f"      ✗ File {recs_file} non trovato")
        return

    if not all_recommendations:
        print("      ✗ Nessuna raccomandazione valida")
        return

    print("\n[3/3] Calcolo metriche...")
    evaluator = RecommenderEvaluator(df_interactions, full_catalog_ids)
    metrics = evaluator.evaluate(all_recommendations)

    print("\n" + "=" * 60)
    print("RISULTATI VALUTAZIONE")
    print("=" * 60)
    print(f"Catalog Coverage:      {metrics['catalog_coverage']:.2%}  (target: >30%)")
    print(f"Gini Index:            {metrics['gini_index']:.4f}  (target: <0.95)")
    print(f"Shannon Entropy:       {metrics['shannon_entropy']:.4f}  (maggiore è meglio)")
    print(f"Novelty Score:         {metrics['novelty_score']:.4f} bits")
    print("=" * 60)


if __name__ == "__main__":
    try:
        generate_recommendations_safe()

        # Uncomment to run evaluation after generation
        # print("\n")
        # evaluate_from_file()

    except KeyboardInterrupt:
        print("\n\n⚠ Interruzione manuale - progresso salvato")
    except Exception as e:
        print(f"\n\n✗ ERRORE FATALE: {e}")
        traceback.print_exc()