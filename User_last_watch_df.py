import os.path
import pandas as pd
from UserDBConnector import UserDBConnector
class User_test_df:

    def __init__(self):
        self.userAnime_last_watched = self.get_user_last_watched_df()

    df_user_next_parquet_path = "Resources/user_watches_next.parquet"
    userAnime_df_path = "Resources/UserAnimeList.parquet"
    """
    a class to get a data frame of the user watched 
    """
    def _create_user_next_df(self):
        if(not os.path.exists(User_test_df.userAnime_df_path)):
            UserDBConnector()
        userAnime_df = pd.read_parquet(User_test_df.userAnime_df_path)
        cols = ['anime_id', 'my_score']
        userAnime_df['id_score_pair'] = userAnime_df[cols].values.tolist()
        grouped = userAnime_df.groupby('username')['id_score_pair'].agg(list).reset_index()
        filtered_users = grouped[grouped['id_score_pair'].apply(len) > 1].copy()
        filtered_users['watched_anime'] = filtered_users['id_score_pair'].apply(lambda x: x[:-1])
        filtered_users['lastwatched'] = filtered_users['id_score_pair'].apply(lambda x: x[-1])
        user_next_df = filtered_users[['username', 'watched_anime', 'lastwatched']]
        user_next_df =user_next_df[user_next_df['lastwatched'].apply(lambda x: x[1] > 6)]
        user_next_df.to_parquet(User_test_df.df_user_next_parquet_path)


    def get_user_last_watched_df(self):
        if(not os.path.exists(User_test_df.df_user_next_parquet_path)):
            self._create_user_next_df()
        return pd.read_parquet(User_test_df.df_user_next_parquet_path)

    def getDataFrame(self):
        return self.userAnime_last_watched

