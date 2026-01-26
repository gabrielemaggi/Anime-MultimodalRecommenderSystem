import subprocess
import sys

def install_and_import():
    package = 'sentence_transformers'
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not find, installing ")
        # running the pip commend
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"installation of {package} is completed")

install_and_import()

import pandas as pd

data = pd.read_csv("AnimeList.csv", sep= ',')
data_sample = data.head(5)

sypnopsis = data['sypnopsis'].to_list() # get a list of all the synopsis of the anime data base
sypnopsis_sample = data_sample['sypnopsis'].to_list() #get a list of 5 first synopsis of the anime data base


#example from the document of sentenceBert
from sentence_transformers import SentenceTransformer


# Load a pretrained Sentence Transformer model
def getModel():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# Calculate embeddings by calling model.encode()
#return a list of all the embedding of all the sentence
def get_embeddings(model , sentences):
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    # [3, 384]
    return embeddings

#  Calculate the embedding similarities
def calc_emb_similarity(model,  embeddings):
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # tensor([[1.0000, 0.6660, 0.1046],
    #         [0.6660, 1.0000, 0.1411],
    #         [0.1046, 0.1411, 1.0000]])


# The sentences to encode
sentences = sypnopsis_sample


def run_model(sentences):
    model = getModel()
    embeddings = get_embeddings(model, sentences)
    calc_emb_similarity(model, embeddings)