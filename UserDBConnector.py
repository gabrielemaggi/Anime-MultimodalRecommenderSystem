
import pandas as pd
from UserDBCleaner import UserDBCleaner
class UserDBConnector:

    def __init__(self):
        self.dbCleaner = UserDBCleaner()
        self.userMapDf = self.dbCleaner.get_user_df()
        self.userAnimeDF = self.dbCleaner.get_userAnime_df()

    def _load_initial_data(self):
        try:
            # Loading Parquet is significantly faster than CSV
            self.anime_list_df = pd.read_parquet("Resources/UserAnimeList.parquet")
            print("Loaded via Parquet (Fast).")
        except FileNotFoundError:
            # Fallback to CSV if parquet doesn't exist yet
            print("error with reading the parquet, returning None. ")
            return


    def get_unused_user_id(self):
        """
        Returns the highest user_id found in the mapping database and add one.
        """
        # Ensure the mapping dataframe exists and is not empty
        if self.userMapDf is None or self.userMapDf.empty:
            print("Warning: User mapping database is empty or not loaded.")
            return 0

        # Use the pandas max() method on the 'user_id' column
        # This is highly optimized for performance
        max_id = self.userMapDf['user_id'].max()

        return max_id + 1


    def add_User(self, user): # maybe delete.
        pass

    def get_anime_watched_by_user(self, name):
        """
        Returns a list of pairs [anime_id, my_score] watched by a specific username.
        Each element in the returned list is a list of size 2.
        """
        # Ensure the anime list dataframe is already loaded into memory
        if self.userAnimeDF is None:
            print("Error: Anime list not loaded.")
            return []

        #return the list of animes and thier score as [[anime_id1 , score] , [anime_id2, score ...]
        return self.userAnimeDF.loc[self.userAnimeDF['username'] == name, ['anime_id', 'my_score']].values.tolist()


    def check_if_user_exists(self, identifier):
        if isinstance(identifier, int):
            # Ensure the mapping dataframe is loaded
            if self.userMapDf is None:
                print("Error: User mapping database is not loaded.")
                return False

            # Check if the user_id exists in the 'user_id' column
            # We use .values to check for existence efficiently
            exists = identifier in self.userMapDf['user_id'].values
            return exists
        if isinstance(identifier, str):
            # Ensure the mapping dataframe is loaded
            if self.userMapDf is None:
                print("Error: User mapping database is not loaded.")
                return False

            # Check if the user_id exists in the 'user_id' column
            # We use .values to check for existence efficiently
            exists = identifier in self.userMapDf['username'].values
            return exists

    def getUserId_by_name(self ,name):
        df = self.userMapDf
        if df is None:
            print("Error: User mapping database is not loaded.")
            return
        return df.loc[df['username'] == name,'user_id'].iloc[0]

    def get_username_byID(self ,id):
        df = self.userMapDf
        if df is None:
            print("Error: User mapping database is not loaded.")
            return
        return  df.loc[df['user_id'] == id, 'username'].iloc[0]
