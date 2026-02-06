"""
Test script for visualizing user clusters and recommendations using t-SNE
IMPROVED VERSION: Uses cosine similarity and better visualization parameters
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Import your classes
from Libs.indexing_db import Indexing
from Libs.User import User


class UserClusterVisualizer:
    """
    Visualize user's anime clusters and recommendations in 2D using t-SNE
    """

    def __init__(self, username, index):
        """
        Initialize visualizer with a user and vector database

        Parameters:
        -----------
        username : str
            MyAnimeList username
        index : Indexing
            Loaded vector database
        """
        self.username = username
        self.index = index
        self.user = User(username)

        # Find clusters
        print(f"Finding clusters for user: {username}")
        self.user.findCentersOfClusters(self.index)

        # Get recommendations
        print(f"Getting recommendations...")
        self.recommendations = self.user.get_nearest_anime_from_clusters(
            self.index, top_k=15
        )

        # Get watched anime embeddings
        print(f"Loading watched anime embeddings...")
        self.watched_embeddings = self._get_watched_embeddings()

    def _get_watched_embeddings(self):
        """Get embeddings for all watched anime"""
        embeddings = []
        for anime_id, score in self.user.watched:
            try:
                # Get embedding from vector database
                embedding = self.index.get_db_embedding_by_id(int(anime_id))
                if embedding is not None:
                    embeddings.append(
                        {"id": anime_id, "score": score, "embedding": embedding}
                    )
            except Exception as e:
                print(f"Warning: Could not get embedding for anime {anime_id}: {e}")

        return embeddings

    def _get_recommendation_embeddings(self):
        """Get embeddings for recommended anime"""
        embeddings = []
        for rec in self.recommendations:
            try:
                anime_id = rec.get("id")
                embedding = self.index.get_db_embedding_by_id(int(anime_id))
                if embedding is not None:
                    embeddings.append(
                        {
                            "id": anime_id,
                            "title": rec.get("title", "Unknown"),
                            "similarity": rec.get("similarity", 0),
                            "embedding": embedding,
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not get embedding for recommendation: {e}")

        return embeddings

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _closest_cluster_cosine(self, embedding):
        """Find the closest cluster centroid using cosine similarity"""
        similarities = [
            self._cosine_similarity(embedding, centroid)
            for centroid in self.user.embeddings
        ]
        return np.argmax(similarities)  # Highest similarity = closest

    def plot_clusters_and_recommendations(
        self, perplexity=30, random_state=42, show_genres=None
    ):
        """
        Create a 2D t-SNE visualization of clusters, watched anime, and recommendations

        Parameters:
        -----------
        perplexity : int, default=30
            t-SNE perplexity parameter
        random_state : int, default=42
            Random seed for reproducibility
        show_genres : list of str, optional
            List of genre names to show as background bubbles (e.g., ["Action", "Romance"])
        """
        # Prepare data for t-SNE
        all_embeddings = []
        labels = []
        colors = []
        sizes = []
        alphas = []

        # 0. Add genre embeddings (if requested) - these go first as background
        genre_info = []
        if show_genres:
            print(
                f"\nEncoding {len(show_genres)} genres for background visualization..."
            )
            for genre_name in show_genres:
                try:
                    # Encode the genre
                    results = self.index.encode_tabular_genre_studio(
                        genres=[genre_name], studios=[]
                    )
                    genre_embedding = results.get("genres", {}).get(genre_name)

                    if genre_embedding is not None:
                        # Align embedding to the same space as anime embeddings
                        aligned_embedding = self.index.align_embedding(
                            genre_embedding, modality="tab"
                        )
                        all_embeddings.append(aligned_embedding)
                        labels.append(f"Genre: {genre_name}")
                        genre_info.append(
                            {"name": genre_name, "index": len(all_embeddings) - 1}
                        )
                        colors.append("gray")
                        sizes.append(500)  # Large size for visibility
                        alphas.append(0.2)  # Very transparent
                        print(f"  ✓ Added genre: {genre_name}")
                    else:
                        print(f"  ✗ Could not encode genre: {genre_name}")
                except Exception as e:
                    print(f"  ✗ Error encoding genre {genre_name}: {e}")

        n_genres = len(genre_info)

        # 1. Add cluster centroids
        for i, centroid in enumerate(self.user.embeddings):
            all_embeddings.append(centroid)
            labels.append(f"Cluster {i + 1}")
            colors.append("red")
            sizes.append(300)
            alphas.append(1.0)

        # 2. Add watched anime
        watched_scores = {}
        for item in self.watched_embeddings:
            all_embeddings.append(item["embedding"])
            labels.append(f"Watched (Score: {item['score']})")
            watched_scores[len(all_embeddings) - 1] = item["score"]
            colors.append("blue")
            # Size based on score
            sizes.append(50 + item["score"] * 10)
            alphas.append(0.6)

        # 3. Add recommendations
        rec_embeddings = self._get_recommendation_embeddings()
        for item in rec_embeddings:
            all_embeddings.append(item["embedding"])
            labels.append(f"{item['title']} (sim: {item['similarity']:.3f})")
            colors.append("green")
            # Size based on similarity
            sizes.append(100 + item["similarity"] * 100)
            alphas.append(0.8)

        # Convert to numpy array
        all_embeddings = np.array(all_embeddings)

        print(f"\nRunning t-SNE on {len(all_embeddings)} points...")
        if show_genres:
            print(f"  - {n_genres} genre embeddings (background)")
        print(f"  - {len(self.user.embeddings)} cluster centroids")
        print(f"  - {len(self.watched_embeddings)} watched anime")
        print(f"  - {len(rec_embeddings)} recommendations")

        # IMPROVED: Use cosine similarity as the distance metric
        # First, normalize all embeddings
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        all_embeddings_normalized = all_embeddings / (norms + 1e-8)

        # Run t-SNE with better parameters
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(all_embeddings) - 1),
            random_state=random_state,
            max_iter=1000,  # Increased iterations (default is 1000)
            learning_rate=200.0,  # Better learning rate
            metric="cosine",  # Use cosine distance - critical for semantic embeddings!
            init="random",
            verbose=1,
        )
        embeddings_2d = tsne.fit_transform(all_embeddings_normalized)

        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 12))

        # Separate indices for each category
        genre_idx = range(0, n_genres) if show_genres else []
        n_clusters = len(self.user.embeddings)
        n_watched = len(self.watched_embeddings)
        n_recommendations = len(rec_embeddings)

        cluster_idx = range(n_genres, n_genres + n_clusters)
        watched_idx = range(n_genres + n_clusters, n_genres + n_clusters + n_watched)
        rec_idx = range(n_genres + n_clusters + n_watched, len(all_embeddings))

        # Plot genre bubbles as background (if requested)
        if show_genres and genre_info:
            # Define distinct colors for genres
            genre_colors = [
                "purple",
                "orange",
                "pink",
                "cyan",
                "yellow",
                "brown",
                "lime",
                "magenta",
            ]

            for i, genre in enumerate(genre_info):
                idx = genre["index"]
                color = genre_colors[i % len(genre_colors)]

                # Draw large semi-transparent circle
                circle = Circle(
                    (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    radius=8,  # Large radius for background effect
                    fill=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.15,
                    linewidth=2,
                    zorder=0,  # Behind everything
                )
                ax.add_patch(circle)

                # Add genre label
                ax.text(
                    embeddings_2d[idx, 0],
                    embeddings_2d[idx, 1],
                    genre["name"],
                    fontsize=14,
                    fontweight="bold",
                    color=color,
                    alpha=0.6,
                    ha="center",
                    va="center",
                    zorder=1,
                )

        # Plot watched anime (with score-based coloring)
        watched_points = embeddings_2d[watched_idx]
        watched_scores_list = [watched_scores[i] for i in watched_idx]
        scatter_watched = ax.scatter(
            watched_points[:, 0],
            watched_points[:, 1],
            c=watched_scores_list,
            cmap="Blues",
            s=[sizes[i] for i in watched_idx],
            alpha=0.6,
            edgecolors="darkblue",
            linewidth=1,
            label="Watched Anime",
            vmin=0,
            vmax=10,
            zorder=3,
        )

        # Plot recommendations
        rec_points = embeddings_2d[rec_idx]
        ax.scatter(
            rec_points[:, 0],
            rec_points[:, 1],
            c="green",
            s=[sizes[i] for i in rec_idx],
            alpha=0.7,
            edgecolors="darkgreen",
            linewidth=1.5,
            marker="^",
            label="Recommendations",
            zorder=4,
        )

        # Plot cluster centroids with star markers
        cluster_points = embeddings_2d[cluster_idx]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c="red",
            s=400,
            alpha=1.0,
            edgecolors="darkred",
            linewidth=2,
            marker="*",
            label="Cluster Centroids",
            zorder=5,
        )

        # IMPROVED: Calculate adaptive radius for circles based on actual distances
        # in the t-SNE space
        for i, cluster_idx_val in enumerate(cluster_idx):
            # Find points assigned to this cluster
            cluster_points_indices = [
                j
                for j in watched_idx
                if self._closest_cluster_cosine(all_embeddings[j]) == i
            ]

            if cluster_points_indices:
                # Calculate average distance to points in this cluster
                cluster_center = embeddings_2d[cluster_idx_val]
                distances = [
                    np.linalg.norm(embeddings_2d[j] - cluster_center)
                    for j in cluster_points_indices
                ]
                avg_distance = np.mean(distances) if distances else 5
                radius = avg_distance * 1.5  # Scale factor for visibility
            else:
                radius = 5

            circle = Circle(
                (embeddings_2d[cluster_idx_val, 0], embeddings_2d[cluster_idx_val, 1]),
                radius=radius,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.3,
                zorder=2,
            )
            ax.add_patch(circle)

        # Annotate cluster centroids
        for i in cluster_idx:
            ax.annotate(
                f"C{i + 1}",
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=12,
                fontweight="bold",
                color="darkred",
                ha="center",
                va="center",
            )

        # Annotate top watched anime (highest rated)
        # Sort watched anime by score to show the best ones
        watched_with_idx = [
            (
                i + n_clusters,
                self.watched_embeddings[i],
            )  # (index in embeddings_2d, item)
            for i in range(len(self.watched_embeddings))
        ]
        watched_sorted = sorted(
            watched_with_idx, key=lambda x: x[1]["score"], reverse=True
        )

        num_watched_to_show = 5  # Change this number to show more/fewer watched anime
        for rank, (idx, item) in enumerate(watched_sorted[:num_watched_to_show]):
            # Get anime title from the index
            anime_info = self.index.get_anime_by_id(int(item["id"]))
            title = (
                anime_info.get("title", f"Anime {item['id']}")
                if anime_info
                else f"Anime {item['id']}"
            )

            ax.annotate(
                f"{title[:25]} ({item['score']}⭐)",
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                fontsize=8,
                color="darkblue",
                xytext=(-10, -10),  # Offset in opposite direction from recommendations
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3,rad=-0.3", color="blue"
                ),
            )

        # Annotate top recommendations
        num_recs_to_show = 5  # Change this number to show more/fewer recommendations
        for i, idx in enumerate(list(rec_idx)[:num_recs_to_show]):
            rec_info = rec_embeddings[i]
            ax.annotate(
                rec_info["title"][:30],
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                fontsize=8,
                color="darkgreen",
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3,rad=0.3", color="green"
                ),
            )

        # Add colorbar for watched anime scores
        cbar = plt.colorbar(scatter_watched, ax=ax)
        cbar.set_label("User Score (Watched Anime)", fontsize=12)

        # Styling
        ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
        ax.set_title(
            f'User "{self.username}" - Anime Clusters and Recommendations (t-SNE with Cosine Distance)',
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()

        return fig, ax

    def plot_cluster_details(self):
        """
        Create a detailed multi-panel plot showing cluster statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Cluster sizes (how many watched anime are closest to each cluster)
        cluster_assignments = self._assign_watched_to_clusters()
        cluster_sizes = [
            len(cluster_assignments[i]) for i in range(len(self.user.embeddings))
        ]

        axes[0, 0].bar(
            range(1, len(cluster_sizes) + 1),
            cluster_sizes,
            color="skyblue",
            edgecolor="navy",
        )
        axes[0, 0].set_xlabel("Cluster ID", fontsize=12)
        axes[0, 0].set_ylabel("Number of Watched Anime", fontsize=12)
        axes[0, 0].set_title("Cluster Sizes", fontsize=14, fontweight="bold")
        axes[0, 0].grid(axis="y", alpha=0.3)

        # 2. Average scores per cluster
        cluster_avg_scores = []
        for i in range(len(self.user.embeddings)):
            scores = [
                item["score"]
                for item in self.watched_embeddings
                if self._closest_cluster_cosine(item["embedding"]) == i
            ]
            cluster_avg_scores.append(np.mean(scores) if scores else 0)

        axes[0, 1].bar(
            range(1, len(cluster_avg_scores) + 1),
            cluster_avg_scores,
            color="lightcoral",
            edgecolor="darkred",
        )
        axes[0, 1].set_xlabel("Cluster ID", fontsize=12)
        axes[0, 1].set_ylabel("Average User Score", fontsize=12)
        axes[0, 1].set_title(
            "Average Scores by Cluster", fontsize=14, fontweight="bold"
        )
        axes[0, 1].set_ylim(0, 10)
        axes[0, 1].grid(axis="y", alpha=0.3)

        # 3. Score distribution of watched anime
        all_scores = [item["score"] for item in self.watched_embeddings]
        axes[1, 0].hist(
            all_scores,
            bins=10,
            range=(0, 10),
            color="mediumpurple",
            edgecolor="indigo",
            alpha=0.7,
        )
        axes[1, 0].set_xlabel("User Score", fontsize=12)
        axes[1, 0].set_ylabel("Frequency", fontsize=12)
        axes[1, 0].set_title(
            "Distribution of User Scores", fontsize=14, fontweight="bold"
        )
        axes[1, 0].grid(axis="y", alpha=0.3)

        # 4. Top recommendations with similarity scores
        rec_embeddings = self._get_recommendation_embeddings()[:10]
        titles = [
            rec["title"][:25] + "..." if len(rec["title"]) > 25 else rec["title"]
            for rec in rec_embeddings
        ]
        similarities = [rec["similarity"] for rec in rec_embeddings]

        y_pos = np.arange(len(titles))
        axes[1, 1].barh(y_pos, similarities, color="lightgreen", edgecolor="darkgreen")
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(titles, fontsize=9)
        axes[1, 1].set_xlabel("Cosine Similarity Score", fontsize=12)
        axes[1, 1].set_title("Top 10 Recommendations", fontsize=14, fontweight="bold")
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(axis="x", alpha=0.3)

        plt.suptitle(
            f'User "{self.username}" - Detailed Cluster Analysis',
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        return fig, axes

    def plot_similarity_heatmap(self):
        """
        Create a heatmap showing cosine similarities between cluster centroids
        """
        n_clusters = len(self.user.embeddings)

        # Calculate pairwise cosine similarities
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                similarity_matrix[i, j] = self._cosine_similarity(
                    self.user.embeddings[i], self.user.embeddings[j]
                )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0.5,
            vmin=0,
            vmax=1,
            xticklabels=[f"C{i + 1}" for i in range(n_clusters)],
            yticklabels=[f"C{i + 1}" for i in range(n_clusters)],
            ax=ax,
            cbar_kws={"label": "Cosine Similarity"},
        )
        ax.set_title(
            f"Cluster Centroid Similarity Matrix\n(User: {self.username})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        return fig, ax

    def _assign_watched_to_clusters(self):
        """Assign each watched anime to its nearest cluster using cosine similarity"""
        assignments = {i: [] for i in range(len(self.user.embeddings))}

        for item in self.watched_embeddings:
            cluster_id = self._closest_cluster_cosine(item["embedding"])
            assignments[cluster_id].append(item)

        return assignments

    def save_visualizations(self, output_dir="./visualizations", show_genres=None):
        """
        Save all visualizations to files

        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        show_genres : list of str, optional
            List of genres to show as background bubbles
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Main t-SNE plot
        print("Generating t-SNE visualization...")
        fig1, _ = self.plot_clusters_and_recommendations(show_genres=show_genres)
        fig1.savefig(
            f"{output_dir}/user_{self.username}_tsne.png", dpi=300, bbox_inches="tight"
        )
        print(f"Saved: {output_dir}/user_{self.username}_tsne.png")

        # Cluster details
        print("Generating cluster details...")
        fig2, _ = self.plot_cluster_details()
        fig2.savefig(
            f"{output_dir}/user_{self.username}_details.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved: {output_dir}/user_{self.username}_details.png")

        # Similarity heatmap
        print("Generating similarity heatmap...")
        fig3, _ = self.plot_similarity_heatmap()
        fig3.savefig(
            f"{output_dir}/user_{self.username}_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved: {output_dir}/user_{self.username}_heatmap.png")

        plt.close("all")


def test_user_visualization(username="MrPeanut02", show_genres=None):
    """
    Main test function to visualize a user's clusters and recommendations

    Parameters:
    -----------
    username : str
        MyAnimeList username to visualize
    show_genres : list of str, optional
        List of genres to show as background bubbles (e.g., ["Action", "Romance", "Fantasy"])
    """
    print("=" * 80)
    print(f"USER CLUSTER VISUALIZATION TEST (IMPROVED WITH COSINE SIMILARITY)")
    print("=" * 80)

    # Load vector database
    print("\n1. Loading vector database...")
    index = Indexing()
    index.load_vector_database()
    print("✓ Vector database loaded")

    # Create visualizer
    print(f"\n2. Creating visualizer for user: {username}")
    visualizer = UserClusterVisualizer(username, index)
    print(f"✓ Found {len(visualizer.user.embeddings)} clusters")
    print(f"✓ User has watched {len(visualizer.watched_embeddings)} anime")
    print(f"✓ Generated {len(visualizer.recommendations)} recommendations")

    # Generate visualizations
    print("\n3. Generating t-SNE visualization...")
    if show_genres:
        print(f"   Including genre bubbles: {show_genres}")
    fig1, ax1 = visualizer.plot_clusters_and_recommendations(show_genres=show_genres)

    print("\n4. Generating cluster details...")
    fig2, ax2 = visualizer.plot_cluster_details()

    print("\n5. Generating similarity heatmap...")
    fig3, ax3 = visualizer.plot_similarity_heatmap()

    # Save visualizations
    print("\n6. Saving visualizations...")
    visualizer.save_visualizations(show_genres=show_genres)

    # Show plots
    print("\n7. Displaying plots...")
    plt.show()

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return visualizer


if __name__ == "__main__":
    # Test with default user and genre visualization
    genres_to_show = ["Fantasy", "Action", "Romance"]  # Add/remove genres as needed
    visualizer = test_user_visualization("MrPeanut02", show_genres=genres_to_show)

    # You can also test without genres:
    # visualizer = test_user_visualization("MrPeanut02")

    # Or with different genres:
    # visualizer = test_user_visualization("MrPeanut02", show_genres=["Drama", "Comedy", "Sci-Fi"])
