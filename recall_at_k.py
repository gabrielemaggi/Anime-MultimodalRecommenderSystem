from User_last_watch_df import User_test_df
from indexing_db import *
from TemporalUser import TemporalUser
class recall_at_k:



    def __init__(self, k, number_of_last_watched):
        self.k = k
        self.df_last_watched = self.get_last_watched_df(number_of_last_watched)

        #load the vector dataBase
        index = Indexing()
        index.load_vector_database()
        self.index = index

    def get_last_watched_df(self, n ):
        U_df =User_test_df(n)
        return U_df.getDataFrame()


    def predict_next_anime_df(self):
        """
        return a data frame [username , watched , lastwatch, last_watches_vectors , prediction]
        uses the self.K

        """
        df_predictions = self.df_last_watched
        df_predictions['last_watches_vectors']

    def _get_last_watches_vectors(self,last_watched_animes ):
        return [self.index.encode_by_id(anime)]

