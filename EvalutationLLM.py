import json

import ollama

from Libs import *
from Libs.indexing_db import *

# Configuration
MODEL_NAME = "qwen3:30b-a3b"  # Make sure you have run 'ollama pull llama3'
OLLAMA_FORMAT = "json"  # Forces the model to output valid JSON


# 1. Data Structure (Mock Data)
# In a real scenario, this comes from your Database or API
def get_user_history(user, index):
    result = []
    for id, score in user.get_watchList():
        data = index.get_anime_info_by_id(id)
        result.append(
            {
                "title": data["title"],
                "genre": data["genre"],
                "studio": data["studio"],
                "sypnopsis": data["sypnopsis"],
            }
        )
    return result


# 2. Your Recommender System
# Replace this function with your actual recommendation logic
def my_recommender_system(user, index, k=10):
    user.findCentersOfClusters(index)
    return user.get_nearest_anime_from_clusters(index, k)


# 3. Evaluation Function using Ollama
def evaluate_recommendation_with_ollama(history, recommendation):
    """
    Constructs the prompt and asks Ollama for a consistency score.
    """

    # Serialize data to JSON strings for the prompt
    history_str = json.dumps(history, indent=2)
    rec_str = json.dumps(recommendation, indent=2)

    # Prompt Engineering
    prompt = f"""
    You are an expert anime critic and a Recommender System Quality Evaluator.

    Your task is to evaluate if the Recommended Anime ("RECOMMENDATION") is consistent with the User's taste based on their history ("USER_HISTORY").

    EVALUATION CRITERIA:
    1. Analyze GENRES: Is there an overlap or affinity?
    2. Analyze STUDIO: Does the user prefer a specific visual style?
    3. Analyze SYNOPSIS (Plot/Themes): Are the themes (e.g., psychological, dark, slice of life) compatible?

    INPUT DATA:
    ---
    USER_HISTORY:
    {history_str}
    ---
    RECOMMENDATION:
    {rec_str}
    ---

    REQUIRED OUTPUT:
    Respond ONLY with a JSON object containing these two fields:
    - "score": an integer from 1 to 5 (1 = terrible recommendation, 5 = perfect fit).
    - "reasoning": a concise explanation of why you gave that score, citing genre, studio, or thematic similarities.
    - "explaination": explain why these recommendations were done based on the user's history.
    """

    print(prompt)

    print(f"🔄 Requesting evaluation from {MODEL_NAME}...")

    client = ollama.Client(host="http://192.168.99.95:11434")

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt},
            ],
            format=OLLAMA_FORMAT,
        )

        # Parse the response
        content = response["message"]["content"]
        result = json.loads(content)
        return result

    except json.JSONDecodeError:
        return {"score": 0, "reasoning": "Error: Model did not return valid JSON."}
    except Exception as e:
        return {"score": 0, "reasoning": f"Generic Error: {str(e)}"}


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    index = Indexing()
    index.load_vector_database()

    user = User("MrPeanut02")

    ###
    # 1. Load User Data
    history = get_user_history(user, index)
    # 2. Get Recommendation
    prediction = my_recommender_system(user, index, k=5)

    evaluation = evaluate_recommendation_with_ollama(history, prediction)

    print(evaluation)

    # 5. Print Evaluation Results
    print("\n" + "=" * 40)
    print(f" 🤖  LLM EVALUATION ({MODEL_NAME})")
    print("=" * 40)
    print(f"Score:     {evaluation.get('score')}/5")
    print(f"Reasoning: {evaluation.get('reasoning')}")
    print(f"Explaination: {evaluation.get('explaination')}")
    print("=" * 40 + "\n")
