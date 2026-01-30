from math import sqrt
import numpy as np
from sklearn.cluster import KMeans
class clusterFinder:

    def __init__(self , vecArray):
        """
        :param vecArray: an array of vector and scores e.g : [ [ v1, 10] , [v2 , score2 ] ....]
        """
        self.num_of_anime = len(vecArray)
        vectors_temp = np.vstack(vecArray[:, 0])
        self.vectors = [get_Anime_Embding_by_id(v) for  v in vectors_temp]
        self.scores = vecArray[:, 1].astype(int)
        self.K = int(sqrt(self.num_of_anime / 2 ))

    def get_centers(self):
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=10)

        # run with weights
        kmeans.fit(self.vectors, sample_weight=self.scores)

        # the ceneters are the user profile
        user_profiles = kmeans.cluster_centers_
        return user_profiles

    def getK(self):
        return self.K


