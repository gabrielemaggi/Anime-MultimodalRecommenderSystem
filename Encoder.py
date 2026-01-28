import abc

class Encoder(abc.ABC):

    @abc.abstractmethod
    def encode(self, paths):
        pass

    @abc.abstractmethod
    def run_model(self, sentences_or_images):
        pass

    @abc.abstractmethod
    def __load(filepath):
        pass
