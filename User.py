import pandas as pd
from clusterFinder import clusterFinder
from indexing_db import *
from mal import client
from sympy.codegen.ast import Raise
from UserDBConnector import UserDBConnector
import textwrap
import matplotlib.pyplot as plt

class User:
    # get a user from the db
    def __init__(self, id):
        self.userDBConnector = UserDBConnector()
        self.embeddings = None

        if self.userDBConnector.check_if_user_exists(id):
            self.id = id
            self.watched = self.userDBConnector.get_anime_watched_by_user(id)

        else:
            API_ID = "d79a8a3b8f42750e317b0b7abc47adf2"
            api = client.Client(API_ID)

            if isinstance(id, int):
                raise ValueError("User ID must be a String")

            watch_list = api.get_anime_list(username=id, limit=100, include_nsfw=True)
            self.id = id

            df_anime = pd.read_csv("./dataset/AnimeList.csv")

            watched = []
            for anime in watch_list:
                # TODO recupera anche: sinossi, immagine, genere e studio
                # creo dizionario e poi chiamo vector_db add vector, da creare anche i metadata.
                # solo dei dati non nel dizionario
                title_mal = anime.entry.title
                score = anime.list_status.score

                match = df_anime[df_anime["title"].str.lower() == title_mal.lower()]
                do_not_match = df_anime[
                    df_anime["title"].str.lower() != title_mal.lower()
                ]

                if not match.empty:
                    anime_id = match.iloc[0]["id"]
                    watched.append([anime_id, score])
                else:
                    print(f"Warning: {title_mal} do not find in the anime database.")

            self.watched = watched
            self.watch_anime_info = watched

            # TODO add user to DB

    # create a new User in the system
    def createNew(self, username, watched):
        self.userDBConnector = UserDBConnector()
        self.id = self.userDBConnector.get_unused_user_id()
        self.watched = watched
        self.userDBConnector.add_User(
            username, watched
        )  # maybe to delete, not sure if neccessary

    def get_watchList(self):
        return self.watched

    def findCentersOfClusters(self):
        """
        set numbers of clusters and return the centers
        :return:
        """
        kmean = clusterFinder(self.watched)
        self.K = kmean.getK()
        self.embeddings = kmean.get_centers()
        return self.embeddings

    def get_nearest_anime_from_clusters(self, vector_db, top_k: int = 10):
        """
        Find nearest anime to each cluster center and return their IDs
        """
        if not hasattr(self, "embeddings"):
            self.findCentersOfClusters()

        all_anime_entries = []

        # 1. Gather all recommendations
        for center_embedding in self.embeddings:
            nearest_entries = vector_db.search(
                query_embedding=center_embedding, top_k=top_k
            )
            all_anime_entries.extend(nearest_entries)

        # 2. PRE-PROCESS WATCHED IDS
        # Convert self.watched [[id, score], ...] into a set of IDs for fast O(1) lookup.
        # We cast to str() to ensure '123' (string) matches 123 (int) to avoid type mismatches.
        watched_ids = {str(item[0]) for item in self.watched}

        # 3. Filter duplicates and watched items
        seen_ids = set()
        unique_anime_entries = []
        for anime_data in all_anime_entries:
            # Extract the ID from the dictionary (vector DB result)
            # Ensure we use the correct key, e.g., 'id' or 'anime_id'
            current_id = str(anime_data.get("id"))

            # Check if we have seen this ID in this loop OR if the user watched it
            if current_id not in seen_ids and current_id not in watched_ids:
                seen_ids.add(current_id)
                unique_anime_entries.append(anime_data)

        # Sort by similarity and slice to respect the actual top_k requested
        unique_anime_entries.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return unique_anime_entries

    def debug_plot_watchlist(self):
        """
        Debug method: Load AnimeList.csv and plot titles and synopsis of watched anime
        self.watched format: [[anime_id, score], [anime_id, score], ...]
        """


        try:

            # Load the anime database
            anime_df = pd.read_csv("./dataset/AnimeList.csv")

            # Extract just the anime IDs from self.watched
            watched_ids = [anime[0] for anime in self.watched]

            # Create a dictionary of user scores for quick lookup
            user_scores = {anime[0]: anime[1] for anime in self.watched}

            # Filter to only watched anime
            watched_anime = anime_df[anime_df["id"].isin(watched_ids)].copy()

            # Add user's personal scores
            watched_anime["user_score"] = watched_anime["id"].map(user_scores)

            print(f"\n{'=' * 80}")
            print(f"WATCHLIST FOR USER {self.id}")
            print(f"Total watched anime: {len(watched_anime)}")
            print(f"{'=' * 80}\n")

            # Display each anime
            for idx, row in watched_anime.iterrows():
                print(f"\n{'-' * 80}")
                print(f"ANIME ID: {row['id']}")
                print(f"TITLE: {row.get('title', 'N/A')}")
                print(f"YOUR SCORE: {row['user_score']}/10")
                print(f"{'-' * 80}")

                # Handle synopsis
                synopsis = row.get("sypnopsis", "No synopsis available")
                if pd.isna(synopsis):
                    synopsis = "No synopsis available"

                # Wrap text for better readability
                wrapped_synopsis = textwrap.fill(str(synopsis), width=80)
                print(f"\nSYNOPSIS:\n{wrapped_synopsis}\n")

                # Additional info if available
                if "genre" in row and not pd.isna(row["genre"]):
                    print(f"GENRES: {row['genre']}")

            print(f"\n{'=' * 80}\n")

        except FileNotFoundError:
            print("Error: AnimeList.csv not found!")
        except Exception as e:
            print(f"Error loading watchlist: {e}")
            import traceback

            traceback.print_exc()

    def add_anime(self, anime_id, rating):
        print(self.watched)
        self.watched.append([int(anime_id), int(rating)])

    def add_filtering(self, query_vector, mode='append'):
        """
        Add filtering to user's cluster centroids.
        Parameters:
        -----------
        query_vector : numpy array or list
            The query vector to use for filtering
        mode : str, default='append'
            - 'append': Add the query vector as a new centroid
            - 'move': Move existing centroids toward the query vector
        Returns:
        --------
        numpy array : Updated embeddings
        """
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run findCentersOfClusters() first.")
        query_vector = np.array(query_vector)
        if mode == 'append':
            # Original behavior: append the query as a new centroid
            self.embeddings = np.append(self.embeddings, [query_vector], axis=0)
        elif mode == 'move':
            # New behavior: move centroids toward the query vector
            # Calculate distances from each centroid to the query
            distances = np.linalg.norm(self.embeddings - query_vector, axis=1)
            # Find the maximum distance (most distant centroid)
            max_distance = np.max(distances)
            # Heuristic: move each centroid toward query by a fraction of max_distance
            # This ensures we don't move too much relative to the spread of centroids
            # The farther a centroid is from the query, the less we move it (proportionally)
            alpha = 0.2  # Tunable parameter: how much to move (0 = no move, 1 = full move to query)
            moved_embeddings = []
            for i, centroid in enumerate(self.embeddings):
                # Calculate the direction vector from centroid to query
                direction = query_vector - centroid
                # Scale movement inversely with distance to preserve cluster structure
                # Closer centroids move more, distant ones move less
                movement_scale = alpha * (1 - distances[i] / max_distance) if max_distance > 0 else 0
                # Move the centroid
                new_centroid = centroid + movement_scale * direction
                moved_embeddings.append(new_centroid)
            self.embeddings = np.array(moved_embeddings)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'append' or 'move'.")
        return self.embeddings


if __name__ == "__main__":
    index = Indexing()
    index.load_vector_database()

    u = User("MrPeanut02")

    u.debug_plot_watchlist()

    u.findCentersOfClusters()
    print(u.get_nearest_anime_from_clusters(index, 100))
