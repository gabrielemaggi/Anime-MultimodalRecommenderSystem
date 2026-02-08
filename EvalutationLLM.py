import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama

from Libs import *
from Libs.indexing_db import *

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_TO_TEST = [
    "qwen2.5:14b",
    "gemma3:latest",
    "qwen3:30b-a3b",
    # Add more models as needed
    # "mistral:latest",
    # "qwen3:30b-a3b",
]

USERS_TO_TEST = [
    "symonx99",
    "MrPeanut02",
    "Zlaftor",
    # Add more usernames here
]

GENRE_GOALS_TO_TEST = [
    ["Cars"],
    ["Sci-Fi"],
    ["Action", "Adventure"],
    ["Action", "Romance"],
    ["Shoujo", "Romance", "Military"],
    None,  # Test without genre goal
]

OLLAMA_HOST = "http://192.168.99.95:11434"
OLLAMA_FORMAT = "json"
OUTPUT_DIR = Path("./evaluation_results")
K_RECOMMENDATIONS = 10

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging():
    """Configure logging with both file and console handlers"""
    # Create logs directory
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("AnimeEvaluation")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


# Initialize logger
logger = setup_logging()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_user_history(user, index):
    """Get user's watch history"""
    logger.debug(f"Fetching watch history for user: {user.id}")
    result = []

    try:
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
        logger.debug(f"Retrieved {len(result)} items from watch history")
        return result
    except Exception as e:
        logger.error(f"Error fetching user history: {e}", exc_info=True)
        return []


def my_recommender_system(user, index, k=10, genre_goal=None, studio_goal=None):
    """Modified recommender that can optionally filter by genre goal"""
    logger.debug(
        f"Running recommender for user: {user.id}, k={k}, genre_goal={genre_goal}"
    )

    try:
        user.findCentersOfClusters(index)

        if genre_goal is not None:
            logger.debug(f"Applying genre filtering: {genre_goal}")
            results = index.encode_tabular_genre_studio(
                genres=genre_goal, studios=studio_goal
            )
            for gen in genre_goal:
                embedding = results.get("genres").get(gen)
                query = index.align_embedding(embedding, modality="tab")
                user.add_filtering(query, "move")

        if studio_goal is not None:
            logger.debug(f"Applying studio filtering: {studio_goal}")
            results = index.encode_tabular_genre_studio(
                genres=genre_goal, studios=studio_goal
            )
            for stu in studio_goal:
                embedding = results.get("studios").get(stu)
                query = index.align_embedding(embedding, modality="tab")
                user.add_filtering(query, "move")

        recommendations = user.get_nearest_anime_from_clusters(index, k)
        logger.debug(f"Generated {len(recommendations)} recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error in recommender system: {e}", exc_info=True)
        return []


def evaluate_recommendation_with_ollama(
    model_name, history, recommendation, genre_goal=None
):
    """Constructs the prompt and asks Ollama for a consistency score."""
    logger.debug(f"Requesting evaluation from model: {model_name}")

    history_str = json.dumps(history, indent=2)
    rec_str = json.dumps(recommendation, indent=2)

    # Add genre goal context if provided
    genre_goal_context = ""
    if genre_goal:
        genre_goal_context = f"""
    GENRE GOAL:
    The user has explicitly requested recommendations focusing on these genres: {", ".join(genre_goal)}
    You should evaluate if the recommendations align with this specific genre goal.
    """

    prompt = f"""
    You are an expert anime critic and a Recommender System Quality Evaluator.
    Your task is to evaluate if the Recommended Anime ("RECOMMENDATION") is consistent with the User's taste based on their history ("USER_HISTORY").

    {genre_goal_context}

    EVALUATION CRITERIA:
    1. Analyze GENRES: Is there an overlap or affinity?{" Does it match the GENRE GOAL?" if genre_goal else ""}
    2. Analyze STUDIO: Does the user prefer a specific visual style?
    3. Analyze SYNOPSIS (Plot/Themes): Are the themes (e.g., psychological, dark, slice of life) compatible?
    {"4. GENRE GOAL ALIGNMENT: How well do the recommendations match the specified genre goal?" if genre_goal else ""}

    INPUT DATA:
    ---
    USER_HISTORY:
    {history_str}
    ---
    RECOMMENDATION:
    {rec_str}
    ---

    REQUIRED OUTPUT:
    Respond ONLY with a JSON object containing these fields:
    - "score": an integer from 1 to 5 (1 = terrible recommendation, 5 = perfect fit).
    - "reasoning": a concise explanation of why you gave that score, citing genre, studio, or thematic similarities.
    - "explanation": explain why these recommendations were done based on the user's history.
    {'- "genre_goal_score": an integer from 1 to 5 rating how well the recommendations match the genre goal.' if genre_goal else ""}
    {'- "genre_goal_reasoning": explain how well the recommendations align with the requested genres.' if genre_goal else ""}
    """

    client = ollama.Client(host=OLLAMA_HOST)

    try:
        logger.debug(f"Sending evaluation request to {model_name}")
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            format=OLLAMA_FORMAT,
        )
        content = response["message"]["content"]
        result = json.loads(content)

        logger.info(
            f"✓ Evaluation complete - Score: {result.get('score', 0)}/5 (Model: {model_name})"
        )
        logger.debug(f"Evaluation reasoning: {result.get('reasoning', 'N/A')}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error from {model_name}: {e}", exc_info=True)
        return {
            "score": 0,
            "reasoning": "Error: Model did not return valid JSON.",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error communicating with {model_name}: {e}", exc_info=True)
        return {"score": 0, "reasoning": f"Generic Error: {str(e)}", "error": str(e)}


# ============================================================================
# BATCH EVALUATION FUNCTIONS
# ============================================================================


def evaluate_single_user_model(model_name, username, index, genre_goal=None):
    """Evaluate a single user with a single model"""
    genre_desc = f"{genre_goal}" if genre_goal else "None"
    logger.info(f"Evaluating: User={username}, Model={model_name}, Genre={genre_desc}")

    try:
        # Load user
        user = User(username)
        history = get_user_history(user, index)

        if len(history) == 0:
            logger.warning(f"User {username} has no watch history - skipping")
            return None

        logger.debug(f"User {username} has {len(history)} items in history")

        # Get recommendations
        recommendations = my_recommender_system(
            user, index, k=K_RECOMMENDATIONS, genre_goal=genre_goal
        )

        if len(recommendations) == 0:
            logger.warning(f"No recommendations generated for {username} - skipping")
            return None

        logger.debug(f"Generated {len(recommendations)} recommendations for {username}")

        # Evaluate
        evaluation = evaluate_recommendation_with_ollama(
            model_name, history, recommendations, genre_goal=genre_goal
        )

        result = {
            "username": username,
            "model": model_name,
            "genre_goal": genre_goal,
            "history_count": len(history),
            "recommendations_count": len(recommendations),
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat(),
        }

        score = evaluation.get("score", 0)
        logger.info(f"✓ Completed: {username} | {model_name} | Score: {score}/5")

        # Log detailed results
        logger.debug(f"Full evaluation result: {json.dumps(result, indent=2)}")

        return result

    except Exception as e:
        logger.error(
            f"Error evaluating {username} with {model_name}: {e}", exc_info=True
        )
        return {
            "username": username,
            "model": model_name,
            "genre_goal": genre_goal,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def run_batch_evaluation():
    """Run evaluation across all users and models"""
    logger.info("=" * 80)
    logger.info("STARTING BATCH EVALUATION: MULTIPLE USERS × MULTIPLE MODELS")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Models: {MODELS_TO_TEST}")
    logger.info(f"  Users: {USERS_TO_TEST}")
    logger.info(f"  Genre Goals: {GENRE_GOALS_TO_TEST}")
    logger.info(f"  K Recommendations: {K_RECOMMENDATIONS}")
    logger.info(f"  Ollama Host: {OLLAMA_HOST}")

    # Initialize index once
    logger.info("Loading vector database...")
    try:
        index = Indexing()
        index.load_vector_database()
        logger.info("✓ Vector database loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load vector database: {e}", exc_info=True)
        raise

    # Store all results
    all_results = []
    results_by_model = defaultdict(list)

    # Iterate over all combinations
    total_combinations = (
        len(MODELS_TO_TEST) * len(USERS_TO_TEST) * len(GENRE_GOALS_TO_TEST)
    )
    current = 0

    logger.info(f"Total evaluations to run: {total_combinations}")
    logger.info("=" * 80)

    for model_name in MODELS_TO_TEST:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"MODEL: {model_name}")
        logger.info("=" * 80)

        model_start_time = time.time()

        for username in USERS_TO_TEST:
            for genre_goal in GENRE_GOALS_TO_TEST:
                current += 1
                logger.info(f"[{current}/{total_combinations}] Starting evaluation...")

                result = evaluate_single_user_model(
                    model_name, username, index, genre_goal
                )

                if result:
                    all_results.append(result)
                    results_by_model[model_name].append(result)
                    logger.debug(
                        f"Result added to collection. Total results: {len(all_results)}"
                    )
                else:
                    logger.warning(f"No result returned for this evaluation")

                # Small delay to avoid overwhelming the server
                time.sleep(1)

        model_elapsed = time.time() - model_start_time
        logger.info(f"Model {model_name} completed in {model_elapsed:.2f} seconds")

    # Save detailed results
    logger.info("=" * 80)
    logger.info("Saving results...")

    detailed_output = OUTPUT_DIR / f"detailed_results_{timestamp}.json"
    try:
        with open(detailed_output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"✓ Detailed results saved: {detailed_output}")
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}", exc_info=True)

    # Compute and save statistics
    logger.info("Computing statistics...")
    try:
        statistics = compute_statistics(results_by_model)
        stats_output = OUTPUT_DIR / f"statistics_{timestamp}.json"
        with open(stats_output, "w") as f:
            json.dump(statistics, f, indent=2)
        logger.info(f"✓ Statistics saved: {stats_output}")
        logger.debug(f"Statistics: {json.dumps(statistics, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to compute/save statistics: {e}", exc_info=True)
        statistics = {}

    # Generate and save summary
    logger.info("Generating summary...")
    try:
        summary = generate_summary(statistics)
        summary_output = OUTPUT_DIR / f"summary_{timestamp}.txt"
        with open(summary_output, "w") as f:
            f.write(summary)
        logger.info(f"✓ Summary saved: {summary_output}")

        # Print summary to console
        print("\n" + summary)

    except Exception as e:
        logger.error(f"Failed to generate/save summary: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info(f"BATCH EVALUATION COMPLETED")
    logger.info(f"Total evaluations: {len(all_results)}")
    logger.info(f"Results directory: {OUTPUT_DIR}")
    logger.info("=" * 80)

    return all_results, statistics


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================


def compute_statistics(results_by_model):
    """Compute mean and std for each model"""
    logger.info("Computing statistics for all models...")
    statistics = {}

    for model_name, results in results_by_model.items():
        logger.debug(f"Computing statistics for {model_name}")

        # Extract scores
        scores = []
        genre_goal_scores = []
        errors = 0

        for result in results:
            if "error" in result:
                errors += 1
                continue

            eval_data = result.get("evaluation", {})
            score = eval_data.get("score", 0)

            if score > 0:  # Valid score
                scores.append(score)

                # Genre goal score if available
                genre_score = eval_data.get("genre_goal_score")
                if genre_score is not None:
                    genre_goal_scores.append(genre_score)

        # Compute statistics
        stats = {
            "total_evaluations": len(results),
            "successful_evaluations": len(scores),
            "errors": errors,
            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_std": float(np.std(scores)) if scores else 0.0,
            "score_min": float(np.min(scores)) if scores else 0.0,
            "score_max": float(np.max(scores)) if scores else 0.0,
            "score_median": float(np.median(scores)) if scores else 0.0,
            "genre_goal_score_mean": (
                float(np.mean(genre_goal_scores)) if genre_goal_scores else None
            ),
            "genre_goal_score_std": (
                float(np.std(genre_goal_scores)) if genre_goal_scores else None
            ),
            "all_scores": scores,
            "all_genre_goal_scores": genre_goal_scores,
        }

        statistics[model_name] = stats

        logger.info(
            f"  {model_name}: Mean={stats['score_mean']:.2f}, "
            f"Std={stats['score_std']:.2f}, "
            f"Success={stats['successful_evaluations']}/{stats['total_evaluations']}"
        )

    return statistics


def generate_summary(statistics):
    """Generate a text summary of the evaluation"""
    logger.debug("Generating summary report...")

    summary = []
    summary.append("=" * 80)
    summary.append("📊 BATCH EVALUATION SUMMARY")
    summary.append("=" * 80)
    summary.append(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Models Tested: {len(statistics)}")
    summary.append(f"Users: {', '.join(USERS_TO_TEST)}")
    summary.append(f"Genre Goals: {GENRE_GOALS_TO_TEST}")

    summary.append("\n" + "=" * 80)
    summary.append("MODEL PERFORMANCE COMPARISON")
    summary.append("=" * 80)

    # Sort models by mean score
    sorted_models = sorted(
        statistics.items(), key=lambda x: x[1]["score_mean"], reverse=True
    )

    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        summary.append(f"\n{'─' * 80}")
        summary.append(f"Rank #{rank}: {model_name}")
        summary.append(f"{'─' * 80}")
        summary.append(f"  Total Evaluations:      {stats['total_evaluations']}")
        summary.append(f"  Successful:             {stats['successful_evaluations']}")
        summary.append(f"  Errors:                 {stats['errors']}")
        summary.append(f"\n  📈 Overall Scores:")
        summary.append(f"     Mean:                {stats['score_mean']:.2f} / 5.00")
        summary.append(f"     Std Dev:             {stats['score_std']:.2f}")
        summary.append(f"     Min:                 {stats['score_min']:.2f}")
        summary.append(f"     Max:                 {stats['score_max']:.2f}")
        summary.append(f"     Median:              {stats['score_median']:.2f}")

        if stats["genre_goal_score_mean"] is not None:
            summary.append(f"\n  🎯 Genre Goal Alignment:")
            summary.append(
                f"     Mean:                {stats['genre_goal_score_mean']:.2f} / 5.00"
            )
            summary.append(
                f"     Std Dev:             {stats['genre_goal_score_std']:.2f}"
            )

    # Model Ranking
    summary.append("\n" + "=" * 80)
    summary.append("🏆 MODEL RANKING (by Mean Score)")
    summary.append("=" * 80)

    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        medal = (
            "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        )
        summary.append(
            f"{medal} {rank}. {model_name:25s} - {stats['score_mean']:.2f} ± {stats['score_std']:.2f}"
        )

    # Critical Analysis
    summary.append("\n" + "=" * 80)
    summary.append("💡 CRITICAL ANALYSIS")
    summary.append("=" * 80)

    if sorted_models:
        best_model = sorted_models[0]
        worst_model = sorted_models[-1]

        summary.append(f"\n✅ Best Performing Model: {best_model[0]}")
        summary.append(f"   Mean Score: {best_model[1]['score_mean']:.2f}")
        summary.append(f"   Consistency (Std): {best_model[1]['score_std']:.2f}")

        summary.append(f"\n❌ Worst Performing Model: {worst_model[0]}")
        summary.append(f"   Mean Score: {worst_model[1]['score_mean']:.2f}")
        summary.append(f"   Consistency (Std): {worst_model[1]['score_std']:.2f}")

        # Consistency analysis
        most_consistent = min(sorted_models, key=lambda x: x[1]["score_std"])
        summary.append(f"\n🎯 Most Consistent Model: {most_consistent[0]}")
        summary.append(f"   Std Dev: {most_consistent[1]['score_std']:.2f}")

    # Score distribution
    summary.append("\n" + "=" * 80)
    summary.append("📊 SCORE DISTRIBUTION")
    summary.append("=" * 80)

    for model_name, stats in sorted_models:
        summary.append(f"\n{model_name}:")
        if stats["all_scores"]:
            score_counts = {i: 0 for i in range(1, 6)}
            for score in stats["all_scores"]:
                score_counts[int(score)] += 1

            for score in range(5, 0, -1):
                count = score_counts[score]
                bar = "█" * count
                summary.append(f"  {score}/5: {bar} ({count})")

    summary.append("\n" + "=" * 80)

    logger.debug("Summary generation complete")
    return "\n".join(summary)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("ANIME RECOMMENDATION BATCH EVALUATION")
    print("🎯" * 40)

    print(f"\nConfiguration:")
    print(f"  Models: {len(MODELS_TO_TEST)}")
    print(f"  Users: {len(USERS_TO_TEST)}")
    print(f"  Genre Goals: {len(GENRE_GOALS_TO_TEST)}")
    print(
        f"  Total Evaluations: {len(MODELS_TO_TEST) * len(USERS_TO_TEST) * len(GENRE_GOALS_TO_TEST)}"
    )

    input("\nPress ENTER to start the batch evaluation...")

    logger.info("=" * 80)
    logger.info("BATCH EVALUATION STARTED")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        all_results, statistics = run_batch_evaluation()
        elapsed_time = time.time() - start_time

        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        logger.info(
            f"Average time per evaluation: {elapsed_time / len(all_results):.2f} seconds"
        )
        logger.info("✓ Batch evaluation completed successfully!")
        logger.info(f"Results directory: {OUTPUT_DIR}")

        print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")
        print(f"✅ Batch evaluation completed!")
        print(f"📁 Results saved in: {OUTPUT_DIR}")

    except Exception as e:
        logger.critical(f"Batch evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Evaluation failed! Check logs for details.")
        raise
