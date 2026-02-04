
import pandas as pd
import os
class UserDBCleaner:
    _initialized = False

    def __init__(self):
        # Prevent re-initialization if UserDBCleaner() is called again
        if self._initialized:
            print("Already init")
            return

        self.user_list_path = "Resources/UserList.csv"
        self.user_AnimeList_path = "Resources/UserAnimeList.csv"
        self.user_scores_path = "Resources/UserScores"

        self.user_list_parquet = "Resources/UserList.parquet"
        self.user_anime_parquet = "Resources/UserAnimeList.parquet"

        print("Initializing Singleton: Loading databases into memory...")
        if(not self._load_initial_data()):
            self.process_and_finalize_user_data()

        self._initialized = True


    def _load_initial_data(self):
        """Loads dataframes from Parquet if available, otherwise falls back to CSV."""
        # Load User Anime List
        status = True
        if os.path.exists(self.user_anime_parquet):
            self.anime_list_df = pd.read_parquet(self.user_anime_parquet)
            print("Loaded Anime List from Parquet (Fast).")
        else:
            try:
                self.anime_list_df = pd.read_csv(self.user_AnimeList_path)
                print("Loaded Anime List from CSV.")
            except FileNotFoundError:
                self.anime_list_df = None
                status = False
            status = False

        # Load User Mapping
        if os.path.exists(self.user_list_parquet):
            self.user_mapping_df = pd.read_parquet(self.user_list_parquet)
            print("Loaded User Mapping from Parquet (Fast).")
        else:
            try:
                self.user_mapping_df = pd.read_csv(self.user_list_path)
                print("Loaded User Mapping from CSV.")
            except FileNotFoundError:
                self.user_mapping_df = None
                status = False
            status = False
        return  status
    def create_username_scores_csv(self, ratings_path, users_path, output_path):
        df_ratings = pd.read_csv(ratings_path)
        df_users = pd.read_csv(users_path)

        merged_df = pd.merge(df_ratings, df_users, on='user_id', how='left')

        final_df = merged_df[['username', 'anime_id', 'my_score']]

        if output_path:
            final_df.to_csv(output_path, index=False)

        return final_df
    def process_and_finalize_user_data(self):
        """Processes the data and saves it back to both CSV and Parquet formats."""
        if self.anime_list_df is None:
            return

        modified = False
        df = self.anime_list_df

        # --- PART A: CLEAN COLUMNS ---
        columns_to_remove = ['my_watched_episodes', 'my_start_date', 'my_finish_date',
                             'my_status', 'my_rewatching', 'my_last_updated', 'my_tags','my_rewatching_ep']
        existing_cols_to_remove = list(set(df.columns).intersection(columns_to_remove))

        if existing_cols_to_remove:
            df = df.drop(columns=existing_cols_to_remove)
            modified = True

            self.anime_list_df = self.create_username_scores_csv(self.user_AnimeList_path,self.user_list_path,self.user_scores_path)

        if not os.path.exists(self.user_list_parquet):
            self.user_mapping_df.to_parquet(self.user_list_parquet, index = False)

        # --- PART B: SYNC TO DISK ---
        # We always save to Parquet on the first run to ensure future fast loads
        if modified or not os.path.exists(self.user_anime_parquet):
            try:
                self.user_mapping_df.to_parquet(self.user_list_parquet, index=False)
                self.anime_list_df.to_parquet(self.user_anime_parquet, index=False)
                print("Database synchronized (CSV and Parquet).")
            except PermissionError:
                print("Permission denied. Ensure files are closed in other programs.")
        self._initialized = True


    def get_userAnime_df(self):
        return self.anime_list_df


    def get_user_df(self):
        return self.user_mapping_df
