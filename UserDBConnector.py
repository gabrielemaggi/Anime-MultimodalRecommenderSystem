
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

    def check_if_user_exists(self, user_id):
        """
        Checks if a given user_id exists in the user mapping database.
        Returns True if found, False otherwise.
        """
        # Ensure the mapping dataframe is loaded
        if self.userMapDf is None:
            print("Error: User mapping database is not loaded.")
            return False

        # Check if the user_id exists in the 'user_id' column
        # We use .values to check for existence efficiently
        exists = user_id in self.userMapDf['user_id'].values
        return exists

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

    def get_anime_watched_by_user(self, user_id):
        """
        Returns a list of pairs [anime_id, my_score] watched by a specific user_id.
        Each element in the returned list is a list of size 2.
        """
        # Ensure the anime list dataframe is already loaded into memory
        if self.userAnimeDF is None:
            print("Error: Anime list not loaded.")
            return []

        # Filter the dataframe for the given user_id
        user_data = self.userAnimeDF[self.userAnimeDF['user_id'] == user_id]

        # Select both 'anime_id' and 'my_score' columns and convert to a list of lists
        # .values.tolist() is highly efficient for creating nested lists in Python
        watched_data = user_data[['anime_id', 'my_score']].values.tolist()

        return watched_data
