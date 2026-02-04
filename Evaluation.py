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

        :param train_df: DataFrame containing historical interactions ['user_id', 'item_id']
        :param catalog_items: List or array of all unique Item IDs available in the system
        """
        self.catalog = set(catalog_items)
        self.n_catalog = len(self.catalog)

        # Pre-calculate item popularity for Novelty metric (Self-Information)
        # p(i) = frequency of item i in the training set
        total_interactions = len(train_df)
        item_counts = train_df['item_id'].value_counts()

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


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. SETUP SISTEMA
    # ---------------------------------------------------------
    print("Caricamento Index e User...")
    index = Indexing()
    index.load_vector_database()

    # Creiamo l'utente (o ne carichiamo uno esistente)
    u = User("MrPeanut02")

    # Generiamo i cluster e otteniamo le raccomandazioni
    print(f"Generazione raccomandazioni per {u.id}...")
    u.findCentersOfClusters()

    # Otteniamo i dizionari completi degli anime raccomandati (es. top 10)
    recommendations_dicts = u.get_nearest_anime_from_clusters(index, top_k=10)

    # ---------------------------------------------------------
    # 2. ADATTAMENTO DATI (Il "Ponte" tra User e Evaluator)
    # ---------------------------------------------------------

    # Estraiamo solo gli ID dalle raccomandazioni per l'evaluator
    # La tua funzione ritorna dicts, quindi facciamo una list comprehension
    rec_ids = [int(anime['id']) for anime in recommendations_dicts]

    # Creiamo il dizionario richiesto dall'evaluator: {user_id: [lista_item_ids]}
    rec_dict_for_eval = {u.id: rec_ids}

    # ---------------------------------------------------------
    # 3. PREPARAZIONE DATI STORICI E CATALOGO
    # ---------------------------------------------------------
    # Per calcolare Coverage e Novelty serve il catalogo completo e lo storico.
    # Assumiamo tu abbia 'AnimeList.csv' nella cartella come nel codice User.
    try:
        df_full_catalog = pd.read_csv("./AnimeList.csv")

        # CATALOGO: Lista di tutti gli ID disponibili
        full_catalog_ids = df_full_catalog['id'].unique()

        # TRAINING DATA:
        # L'evaluator si aspetta un DataFrame ['user_id', 'item_id'] per calcolare la popolarità.
        # SE NON HAI LO STORICO DI TUTTI GLI UTENTI:
        # Usiamo il catalogo stesso come "train" fittizio per evitare errori,
        # ma nota che la metrica "Novelty" sarà piatta (tutti popolarità 1) se non hai dati reali di utilizzo.
        # Se invece il CSV ha una colonna "members" o "popularity", potremmo simulare un dataset,
        # ma per ora passiamo un dataframe minimale per far funzionare il codice.
        df_train_simulated = df_full_catalog[['id']].rename(columns={'id': 'item_id'})

        # Se hai un vero log delle interazioni (es. UserDBConnector), caricalo qui al posto di df_train_simulated
        # df_train = pd.read_csv("interactions.csv")

    except FileNotFoundError:
        print("Errore: AnimeList.csv non trovato. Impossibile calcolare le metriche.")
        full_catalog_ids = []
        df_train_simulated = pd.DataFrame()

    # ---------------------------------------------------------
    # 4. ESECUZIONE VALUTAZIONE
    # ---------------------------------------------------------
    if len(full_catalog_ids) > 0:
        print("\nAvvio valutazione...")
        evaluator = RecommenderEvaluator(df_train_simulated, full_catalog_ids)
        metrics = evaluator.evaluate(rec_dict_for_eval)

        print(f"\n{'=' * 40}")
        print(f"RISULTATI VALUTAZIONE PER USER: {u.id}")
        print(f"{'=' * 40}")
        print(f"Items Raccomandati (ID): {rec_ids}")
        print(f"-" * 40)
        # Catalog Coverage: % di anime del DB coperti (basso su 1 solo utente è normale)
        print(f"Catalog Coverage:      {metrics['catalog_coverage']:.6%}")
        # Gini/Entropy: Diversità della distribuzione
        print(f"Gini Index (0=Equo):   {metrics['gini_index']:.4f}")
        print(f"Shannon Entropy:       {metrics['shannon_entropy']:.4f}")
        # Novelty: Sorpresa (richiede storico reale per essere significativo)
        print(f"Novelty Score:         {metrics['novelty_score']:.4f}")
        print(f"{'=' * 40}\n")