import pandas as pd

from User_last_watch_df import User_test_df
from indexing_db import *
from TemporalUser import TemporalUser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class precision_recall_at_k:



    def __init__(self, k, number_of_last_watched):
        self.k = k
        self.df_last_watched = self.get_last_watched_df(number_of_last_watched)
        self.df_predictions = None

        #load the vector dataBase
        index = Indexing()
        index.load_vector_database()
        self.index = index

    def get_last_watched_df(self, n ):
        U_df =User_test_df(n)
        return U_df.getDataFrame()


    def predict_next_anime_df(self):
        """
        return a data frame [username , watched_anime , lastwatched, last_watches_vectors , predictions]
        """
        df_predictions = self.df_last_watched
        # adding a last wathed vectors column
        df_predictions['last_watches_vectors'] = df_predictions['lastwatched'].apply(
            lambda x: self._get_last_watches_vectors(x))
        # create a column of list of pairs, each pair is [anime_id, anime vector]
        df_predictions['predictions'] = df_predictions.apply(
            lambda row: self._get_predictions(row['watched_anime'], row['lastwatched']), axis=1)
        df_predictions = df_predictions[df_predictions['predictions'].apply(len) > 0]
        self.df_predictions = df_predictions
        return df_predictions

    def _get_last_watches_vectors(self,last_watched_animes ):
        return [[anime_id_score_pair[0] ,self.index.get_db_embedding_by_id(anime_id_score_pair[0]) ]for anime_id_score_pair in last_watched_animes]

    def _get_predictions(self, watched_anime ,last_watched  ):
        tu  = TemporalUser(watched_anime, last_watched) # create a temporal user and find its K centers
        self.predictions = tu.recomand(self.index ,self.k) #find recomandation for the temporal user and return them
        return  self.predictions

    def _get_animeID_vector_pairs(self, predictions):
        return [[int(anime['id']) , self.index.get_db_embedding_by_id(int(anime['id']))] for anime in predictions]


    def predict_next_anime_test(self ):
        """
        like predict_next_anime but on a small set
        """
        df_predictions = self.df_last_watched.head(20)
        # adding a last wathed vectors column
        df_predictions['last_watches_vectors'] = df_predictions['lastwatched'].apply(
            lambda x: self._get_last_watches_vectors(x))
        # create a column of list of pairs, each pair is [anime_id, anime vector]
        df_predictions['predictions'] = df_predictions.apply(
            lambda row: self._get_predictions(row['watched_anime'], row['lastwatched']), axis=1)
        df_predictions = df_predictions[df_predictions['predictions'].apply(len) > 0]
        self.df_predictions = df_predictions
        return df_predictions

    def calculate_precision_recall_at_k_df(self , similarity_param = 0.7):
        #now we have the DF with the predictions
        df = self.df_predictions
        results = [
            self._calculate_precision_recall_at_k(truth, preds, similarity_param)
            for truth, preds in zip(df['last_watches_vectors'], df['predictions'])
        ]
        metrics_df = pd.DataFrame(results)
        df['recall_at_k'] = metrics_df['recall_at_k'].values
        df['precision_at_k'] = metrics_df['precision_at_k'].values
        self.precision_recall_df = df
        return self.precision_recall_df

    def get_precision_recall_df(self):
        return self.precision_recall_df



        #calculate recall at k for a row
    def _calculate_precision_recall_at_k(self, last_watched, predictions , similarity_param  ):
        # if(last_watched ==None or predictions == None):
        #     raise ValueError("last_watched or predictions cannot be None")
        if(similarity_param > 1 or similarity_param< -1 ):
            raise ValueError(" similarity_param must be between -1 and 1")
        hits = 0 #store the successes of the recomendations
        last_watched_id = [anime[0] for anime in last_watched]
        predictions_id = [anime[0]for anime in predictions]

        exact_match = set() # animes id with exact match between the prediction and the last_watched
        for id in predictions_id:
            if(int(id) in last_watched_id):
                exact_match.add(id)
                hits += 1
        predictions_filtered_vectors = [pair[1] for pair in predictions if pair[0] not in exact_match]
        last_watched_vectors = [pair[1] for pair in last_watched]
        #calcute similarity between vectors in the predictions to vectors in the last match, if they are similar enough , increment hits
        for vector1 in predictions_filtered_vectors:
            for vector2 in last_watched_vectors:
                similarity_temp = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
                similiarity = similarity_temp[0][0]
                if(similiarity > similarity_param ):
                    hits+= 1
                    break
        recall_val = hits / len(last_watched) if len(last_watched) > 0 else 0
        precision_val = hits / len(predictions) if len(predictions) > 0 else 0
        metrics = {
            "recall_at_k": recall_val,
            "precision_at_k": precision_val
        }
        print(metrics)
        return metrics






