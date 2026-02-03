import pandas as pd
from UserDBConnector import UserDBConnector
from clusterFinder import clusterFinder
from indexing_db import *

class User:

    # API_ID = d79a8a3b8f42750e317b0b7abc47adf2

    # get a user from the db
    def __init__(self, id):
        self.userDBConnector = UserDBConnector()
        self.embeddings = None

        if(self.userDBConnector.check_if_user_exists(id)):
            self.id = id;
            self.watched = self.userDBConnector.get_anime_watched_by_user(id)

        else:

            if isinstance(id, int):
                # call api solo con id
                pass
            else:
                # call api con user
                pass


    # create a new User in the system
    def createNew(self, username, watched):
        self.userDBConnector = UserDBConnector()
        self.id = self.userDBConnector.get_unused_user_id()
        self.watched = watched
        self.userDBConnector.add_User(username, watched) #maybe to delete, not sure if neccessary

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
        if not hasattr(self, 'embeddings'):
            self.findCentersOfClusters()
        all_anime_entries = []
        # 1. Gather all recommendations
        for center_embedding in self.embeddings:
            nearest_entries = vector_db.search(
                query_embedding=center_embedding,
                top_k=top_k
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
            current_id = str(anime_data.get('id'))
            # Check if we have seen this ID in this loop OR if the user watched it
            if current_id not in seen_ids and current_id not in watched_ids:
                seen_ids.add(current_id)
                unique_anime_entries.append(anime_data)
        return unique_anime_entries

    def debug_plot_watchlist(self):
        """
        Debug method: Load AnimeList.csv and plot titles and synopsis of watched anime
        self.watched format: [[anime_id, score], [anime_id, score], ...]
        """
        import matplotlib.pyplot as plt
        import textwrap

        try:
            # Load the anime database
            anime_df = pd.read_csv('AnimeList.csv')

            # Extract just the anime IDs from self.watched
            watched_ids = [anime[0] for anime in self.watched]

            # Create a dictionary of user scores for quick lookup
            user_scores = {anime[0]: anime[1] for anime in self.watched}

            # Filter to only watched anime
            watched_anime = anime_df[anime_df['id'].isin(watched_ids)].copy()

            # Add user's personal scores
            watched_anime['user_score'] = watched_anime['id'].map(user_scores)

            print(f"\n{'='*80}")
            print(f"WATCHLIST FOR USER {self.id}")
            print(f"Total watched anime: {len(watched_anime)}")
            print(f"{'='*80}\n")

            # Display each anime
            for idx, row in watched_anime.iterrows():
                print(f"\n{'-'*80}")
                print(f"ANIME ID: {row['id']}")
                print(f"TITLE: {row.get('title', 'N/A')}")
                print(f"YOUR SCORE: {row['user_score']}/10")
                print(f"{'-'*80}")

                # Handle synopsis
                synopsis = row.get('sypnopsis', 'No synopsis available')
                if pd.isna(synopsis):
                    synopsis = 'No synopsis available'

                # Wrap text for better readability
                wrapped_synopsis = textwrap.fill(str(synopsis), width=80)
                print(f"\nSYNOPSIS:\n{wrapped_synopsis}\n")

                # Additional info if available
                if 'genre' in row and not pd.isna(row['genre']):
                    print(f"GENRES: {row['genre']}")

            print(f"\n{'='*80}\n")

        except FileNotFoundError:
            print("Error: AnimeList.csv not found!")
        except Exception as e:
            print(f"Error loading watchlist: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    index = Indexing()
    index.load_vector_database()

    u = User(1289601)

    u.debug_plot_watchlist()

    u.findCentersOfClusters()
    print(u.get_nearest_anime_from_clusters(index, 5))
