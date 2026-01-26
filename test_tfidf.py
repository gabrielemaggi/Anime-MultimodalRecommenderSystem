import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class AnimeRetriever:
    def __init__(self, n_neighbors=5, max_features=None, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features, ngram_range=ngram_range
        )
        self.model_nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
        self.df = None
        self.tfidf_matrix = None

    def fit(self, df):
        self.df = df.reset_index(drop=True)
        # Uniamo i campi per creare il profilo testuale di ogni anime
        soup = (
            self.df["title"].fillna("")
            + " "
            + self.df["genre"].fillna("")
            + " "
            + self.df["sypnopsis"].fillna("")
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(soup)
        print(f"Dimensione vocabolario: {retriever.tfidf_matrix.shape}")
        self.model_nn.fit(self.tfidf_matrix)
        print(f"Sistema pronto! Caricati {len(self.df)} anime.")

    def _print_formatted_results(self, results, header):
        """Metodo di supporto per stampare i risultati in modo pulito"""
        print(f"\n{'=' * 40}")
        print(f" {header}")
        print(f"{'=' * 40}")

        for i, (idx, row) in enumerate(results.iterrows(), 1):
            print(f"{i}. ID:    {row['id']}")
            print(f"   TITLE: {row['title']}")
            print(f"   GENRE: {row['genre']}")
            print(f"   SIMILARITÀ:      {(1 - row['distance']) * 100:.2f}%")

    def find_by_id(self, anime_id):
        """Ricerca partendo dall'ID di un anime presente nel dataset"""
        try:
            # Trova l'indice della riga corrispondente all'ID
            idx = self.df[self.df["id"] == anime_id].index[0]
            distances, indices = self.model_nn.kneighbors(self.tfidf_matrix[idx])

            # Escludiamo il primo risultato (se stesso)
            results = self.df.iloc[indices.flatten()[1:]].copy()
            results["distance"] = distances.flatten()[1:]
            self._print_formatted_results(results, f"Simili all'ID: {anime_id}")
        except IndexError:
            print(f"Errore: L'ID {anime_id} non esiste nel dataset.")

    def find_by_query(self, query):
        """Ricerca tramite testo libero"""
        vec = self.vectorizer.transform([query])
        # Qui usiamo n_neighbors=5 perché non dobbiamo escludere l'input stesso
        distances, indices = self.model_nn.kneighbors(vec, n_neighbors=5)

        # FIX: Aggiunto .copy() per evitare il SettingWithCopyWarning
        results = self.df.iloc[indices.flatten()].copy()
        results["distance"] = distances.flatten()
        self._print_formatted_results(results, f"Risultati per: '{query}'")


# --- TEST ---

# Assicurati che il file AnimeList.csv sia nella stessa cartella
try:
    df = pd.read_csv("AnimeList.csv")
    retriever = AnimeRetriever()
    retriever.fit(df)

    # Esempio di ricerca per ID
    retriever.find_by_id(20)

    # Esempio di ricerca per query arbitraria
    retriever.find_by_query("space pirate adventure")

except FileNotFoundError:
    print("File AnimeList.csv non trovato. Carica il dataset per testare.")
