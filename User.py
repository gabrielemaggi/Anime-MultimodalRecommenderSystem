import pandas as pd
from UserDBConnector import UserDBConnector
from clusterFinder import clusterFinder
from indexing_db import *

class User:

    # get a user from the db
    def __init__(self, id ):
        self.userDBConnector = UserDBConnector()
        self.embeddings = None
        if(self.userDBConnector.check_if_user_exists(id)):
            self.id = id;
            self.watched = self.userDBConnector.get_anime_watched_by_user(id)
        else:
            print("no id found")


    # create a new User in the system
    def createNew(self, username, watched):
        self.userDBConnector = UserDBConnector()
        self.id = self.userDBConnector.get_unused_user_id()
        self.watched =watched
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

        :param vector_db: Your vector database instance
        :param top_k: Number of nearest anime to retrieve per cluster
        :return: List of anime IDs
        """
        if not hasattr(self, 'embeddings'):
            self.findCentersOfClusters()

        all_anime_ids = []

        # For each cluster center
        for center_embedding in self.embeddings:
            # Query the vector database for nearest neighbors
            # The exact method depends on your vector DB implementation
            nearest_ids = vector_db.search(
                query_embedding=center_embedding,
                top_k=top_k
            )
            all_anime_ids.extend(nearest_ids)

        # Remove duplicates while preserving order
        """
        seen = set()
        unique_anime_ids = []
        for anime_id in all_anime_ids:
            if anime_id not in seen and anime_id not in self.watched:
                seen.add(anime_id)
                unique_anime_ids.append(anime_id)

        """
        return all_anime_ids

if __name__ == "__main__":
    index = Indexing()
    index.load_vector_database()

    u = User(-1)
    u.findCentersOfClusters()
    print(u.get_nearest_anime_from_clusters(index, 1))
