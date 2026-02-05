from Libs.indexing_db import *
from Libs.Fusion import *
import numpy as np

def log_embeddings_info(embeddings, sample_key=None, top_n=5):
    """
    Logs metadata and a snippet of the embeddings dictionary.
    """
    print("\n--- [Embedding Inspection Log] ---")

    if not isinstance(embeddings, dict):
        print(f"Error: Expected dict, got {type(embeddings)}")
        return

    total_keys = len(embeddings)
    print(f"Total Entries: {total_keys}")

    if total_keys == 0:
        print("Status: Dictionary is empty.")
        return

    # Grab a sample key if none provided
    if sample_key is None or sample_key not in embeddings:
        sample_key = list(embeddings.keys())[0]

    sample_val = embeddings[sample_key]

    # Handle both numpy arrays and lists
    shape = getattr(sample_val, 'shape', len(sample_val))
    dtype = getattr(sample_val, 'dtype', type(sample_val[0]))

    print(f"Sample ID:    {sample_key}")
    print(f"Vector Shape: {shape}")
    print(f"Data Type:    {dtype}")

    # Show a snippet of the actual values
    snippet = sample_val[:top_n]
    print(f"Vector Preview (first {top_n}): {snippet}")
    print("----------------------------------\n")

if __name__ == "__main__":

    indexer = Indexing()
    indexer.load_vector_database()


    image_path = "./dataset/images/26055.jpg"
    syn = "Joutarou Kuujou and his allies have finally made it to Egypt, where the immortal Dio awaits. Upon their arrival, the group gains a new comrade: Iggy, a mutt who wields the Stand \"The Fool.\" It's not all good news however, as standing in their path is a new group of Stand users who serve Dio, each with a Stand representative of an ancient Egyptian god. As their final battle approaches, it is a race against time to break Joutarou's mother free from her curse and end Dio's reign of terror over the Joestar family once and for all."
    id = 11123
    title = "JoJo no Kimyou na Bouken: Stardust Crusaders 2nd Season"

    data = {
       'title': title,
       'sypnopsis': syn
    }
    nana = 877
    jojo = 666
    blackclover = 34572

    query = indexer.encode_by_id(nana)
    # v = indexer.encode_image(image)
    # s = indexer.encode_sypnopsis(data)
    # t = indexer.encode_tabular("Kishibe Rohan wa Ugokanai")

    # query = fuser.


    results = indexer.search(query)

    info = indexer.get_database_info()
    print("-"*50)
    print("results: ", results)
    print("-"*50)
    print("info: ", info)
