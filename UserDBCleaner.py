
import pandas as pd
import os
class UserDBCleaner:
    _initialized = False

    def __init__(self):
        # Prevent re-initialization if UserDBCleaner() is called again
        if self._initialized:
            print("Already init")
            return

        self.user_list_path = "Var/UserList.csv"
        self.user_AnimeList_path = "Var/UserAnimeList.csv"

        self.user_list_parquet = "Resources/UserList.parquet"
        self.user_anime_parquet = "Resources/UserAnimeList.parquet"

        print("Initializing Singleton: Loading databases into memory...")
        self._load_initial_data()
        self.process_and_finalize_user_data()

        self._initialized = True


    def _load_initial_data(self):
        """Loads dataframes from Parquet if available, otherwise falls back to CSV."""
        # Load User Anime List
        if os.path.exists(self.user_anime_parquet):
            self.anime_list_df = pd.read_parquet(self.user_anime_parquet)
            print("Loaded Anime List from Parquet (Fast).")
        else:
            try:
                self.anime_list_df = pd.read_csv(self.user_AnimeList_path)
                print("Loaded Anime List from CSV.")
            except FileNotFoundError:
                self.anime_list_df = None

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

    def process_and_finalize_user_data(self):
        """Processes the data and saves it back to both CSV and Parquet formats."""
        if self.anime_list_df is None:
            return

        modified = False
        df = self.anime_list_df

        # --- PART A: CLEAN COLUMNS ---
        columns_to_remove = ['my_watched_episodes', 'my_start_date', 'my_finish_date',
                             'my_status', 'my_rewatching', 'my_last_updated', 'my_tags']
        existing_cols_to_remove = list(set(df.columns).intersection(columns_to_remove))

        if existing_cols_to_remove:
            df = df.drop(columns=existing_cols_to_remove)
            modified = True

        # --- PART B: REPLACE USERNAME WITH USER_ID ---
        if 'user_id' not in df.columns and self.user_mapping_df is not None:
            # Cleanup mapping table if needed
            if len(self.user_mapping_df.columns) > 2:
                self.user_mapping_df = self.user_mapping_df[['username', 'user_id']]
                self.user_mapping_df.to_csv(self.user_list_path, index=False)
                self.user_mapping_df.to_parquet(self.user_list_parquet, index=False)

            df = pd.merge(df, self.user_mapping_df, on='username', how='left')
            if 'user_id' in df.columns:
                cols = ['user_id'] + [c for c in df.columns if c not in ['username', 'user_id']]
                df = df[cols]
                modified = True

        self.anime_list_df = df

        # --- PART C: SYNC TO DISK ---
        # We always save to Parquet on the first run to ensure future fast loads
        if modified or not os.path.exists(self.user_anime_parquet):
            try:
                # Save to CSV for readability
                self.anime_list_df.to_csv(self.user_AnimeList_path, index=False)
                # Save to Parquet for performance
                self.anime_list_df.to_parquet(self.user_anime_parquet, index=False)
                print("Database synchronized (CSV and Parquet).")
            except PermissionError:
                print("Permission denied. Ensure files are closed in other programs.")

    def get_userAnime_df(self):
        return self.anime_list_df


    def get_user_df(self):
        return self.user_mapping_df
