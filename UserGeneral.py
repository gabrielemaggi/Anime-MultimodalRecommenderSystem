from clusterFinder import clusterFinder
from indexing_db import *
class GeneralUser:

    def __init__(self ,watched =None):
        self.embeddings = None
        self.watched = watched


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

    def get_nearest_anime_from_clusters(self, vector_db , top_k: int = 10):
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


    def add_anime(self, anime_id, rating):
        print(self.watched)
        self.watched.append([int(anime_id), int(rating)])