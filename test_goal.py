from Libs import *

if __name__ == "__main__":
    # 1. Setup Indexer
    indexer = Indexing()
    indexer.build_vector_database()
    indexer.load_vector_database()

    # 2. Setup User
    u = User("MrPeanut02")
    u.debug_plot_watchlist()
    u.findCentersOfClusters(indexer)

    # 3. Setup Goal Parsing
    # Ensure 'animelist.csv' is in the same folder
    goal_parser = GoalParsing()

    # 4. Simulate a text request
    user_input_text = "I want a Romance anime, in the style of mappa studio"

    # 5. Apply the logic
    goal_parser.process_request(user_input_text, u, indexer)

    # 6. Print final results
    print("---" * 40)
    print("Final Recommendations:")
    print(u.get_nearest_anime_from_clusters(indexer, 10))
