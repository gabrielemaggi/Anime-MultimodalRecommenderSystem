
from UserGeneral import GeneralUser

class TemporalUser(GeneralUser):
    """
    User with field of watched animes and field of last watched.
    """
    def __init__(self, watched, last_wat):
        super().__init__(watched)

        if(self.watched != None ):
            if len(self.watched)>0:
                self.findCentersOfClusters() #set the self.embeddings field and self.K field
        self.predictions = None

    def recomand(self, vec_db , num_of_recomanditions: int = 10):
        if(vec_db == None):
            raise ValueError("You must provide vec_db index")
        predictions_dicts = self.get_nearest_anime_from_clusters( vec_db, num_of_recomanditions) # create a list of
        predictions_id_vector_pairs = [[p['id'] ,vec_db.get_db_embedding_by_id(int(p['id'])) ]for p in predictions_dicts  ] # create a list of id and its anime embadding
        self.predictions = predictions_id_vector_pairs
        return self.predictions




