
from UserGeneral import GeneralUser

class TemporalUser(GeneralUser):
    """
    User with field of watched animes and field of last watched.
    """
    def __init__(self, watched, last_watched):
        super().__init__(watched)
        self.last_watched = last_watched

        if(self.watched != None ):
            if len(self.watched)>0:
                self.findCentersOfClusters() #set the self.embeddings field and self.K field
        self.predictions = None

    def recomand(self, num_of_recomanditions: int = 10):
        self.predictions = self.get_nearest_anime_from_clusters( 10)




