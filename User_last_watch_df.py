import os.path
import pandas as pd
from UserDBConnector import UserDBConnector
class User_test_df:
    """
        a class to get a data frame of the user watched
        """
    def __init__(self, last_k_animes_to_predict =  1):
        self.userAnime_last_watched = self._create_user_last_watched_df(last_k_animes_to_predict)

    filterd_df_last_watched_path = "Resources/user_watches_filtered.parquet"
    userAnime_df_path = "Resources/UserAnimeList.parquet"

    def _create_user_filtered_df(self):
        if(not os.path.exists(User_test_df.userAnime_df_path)):
            UserDBConnector()
        userAnime_df = pd.read_parquet(User_test_df.userAnime_df_path)
        cols = ['anime_id', 'my_score']
        userAnime_df['id_score_pair'] = userAnime_df[cols].values.tolist()
        grouped = userAnime_df.groupby('username')['id_score_pair'].agg(list).reset_index()
        filtered_users = grouped[grouped['id_score_pair'].apply(len) > 1].copy()
        filtered_users.to_parquet(User_test_df.filterd_df_last_watched_path)
        self.filterd_df = filtered_users

    def _create_user_last_watched_df(self, k):
        if(not os.path.exists(User_test_df.filterd_df_last_watched_path)):
            self._create_user_filtered_df()
        else:
            self.filterd_df = pd.read_parquet(User_test_df.filterd_df_last_watched_path)
            self.filterd_df['id_score_pair'] = self.filterd_df['id_score_pair'].apply(
                lambda row: [item.tolist() for item in row]
            )
        self.filterd_df = self.filterd_df[self.filterd_df['id_score_pair'].apply(len) > 3*k]
        self.filterd_df['watched_anime'] = self.filterd_df['id_score_pair'].apply(lambda x: x[:-k])
        self.filterd_df['lastwatched'] = self.filterd_df['id_score_pair'].apply(lambda x: x[-k:])
        self.filterd_df['lastwatched'] = self.filterd_df['lastwatched'].apply(lambda x : [pair for pair in x if pair[1] > 6])
        self.filterd_df = self.filterd_df[self.filterd_df['lastwatched'].apply(len) > 0]
        self.filterd_df = self.filterd_df[['username', 'watched_anime', 'lastwatched']]
    def getDataFrame(self):
        return self.filterd_df

#test for user_last_watched_df
if __name__ =='__main__':
    from recall_at_k import *
    r = precision_recall_at_k(10 ,5)
    r.predict_next_anime_test()
    r.calculate_precision_recall_at_k_df()