import pandas as pd
from Libs.indexing_db import Indexing
from Libs.User import *


def test_user_sync(username):
    print(f"--- Starting Sync Test for User: {username} ---")

    # 1. Initialize the Indexing (required for certain methods, though not for the init)
    # index = Indexing()
    # index.load_vector_database()

    # 2. Initialize User
    # The warnings are printed during the __init__ phase when it loops through the watch_list
    print("Fetching data from MAL and comparing with local database...")
    user = User(username)

    # 3. Summary of results
    total_local = len(user.watched)

    print("\n" + "=" * 50)
    print("SYNC SUMMARY")
    print(f"User: {user.id}")
    print(f"Successfully matched and loaded: {total_local} anime")
    print("=" * 50)

    if total_local > 0:
        print("\nFirst 5 items in matched watchlist (ID, Score):")
        print(user.watched[:5])
    else:
        print("\nNo matches found. Check if your CSV titles match MAL titles exactly.")


if __name__ == "__main__":
    # Test with your specific username
    test_user_sync("MrPeanut02")
