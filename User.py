import pandas as pd
from UserDBConnector import UserDBConnector
class User:

    # get a user from the db
    def __init__(self, id ):
        self.userDBConnector = UserDBConnector()
        if(self.userDBConnector.check_if_user_exists(id)):
            self.id = id;
            self.watched = self.userDBConnector.get_anime_watched_by_user(id)


    # create a new User in the system
    def createNew(self, username, watched):
        self.userDBConnector = UserDBConnector()
        self.id = self.userDBConnector.get_unused_user_id()
        self.watched =watched
        self.userDBConnector.add_User(username, watched) #maybe to delete, not sure if neccessary

    def get_watchList(self):
        return self.watched



