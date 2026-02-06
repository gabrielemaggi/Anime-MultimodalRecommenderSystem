"""
Test script for visualizing user clusters and recommendations using t-SNE
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from sklearn.manifold import TSNE

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

    def plot_clusters_and_recommendations(self, perplexity=30, random_state=42):
        """
        Create a 2D t-SNE visualization of clusters, watched anime, and recommendations

        Parameters:
        -----------
        perplexity : int, default=30
            t-SNE perplexity parameter
        random_state : int, default=42
            Random seed for reproducibility
        """
        # Prepare data for t-SNE
        all_embeddings = []
        labels = []
        colors = []
        sizes = []
        alphas = []

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
        print(f"  - {len(self.user.embeddings)} cluster centroids")
        print(f"  - {len(self.watched_embeddings)} watched anime")
        print(f"  - {len(rec_embeddings)} recommendations")

        # Run t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(all_embeddings) - 1),
            random_state=random_state,
            # n_iter=1000,
        )
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 12))

        # Separate indices for each category
        n_clusters = len(self.user.embeddings)
        n_watched = len(self.watched_embeddings)
        n_recommendations = len(rec_embeddings)

        cluster_idx = range(0, n_clusters)
        watched_idx = range(n_clusters, n_clusters + n_watched)
        rec_idx = range(n_clusters + n_watched, len(all_embeddings))

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

        # Add circles around clusters to show their influence
        for i in cluster_idx:
            circle = Circle(
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                radius=5,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.3,
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

        # Annotate top 5 recommendations
        for i, idx in enumerate(list(rec_idx)[:5]):
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
            f'User "{self.username}" - Anime Clusters and Recommendations (t-SNE Visualization)',
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
                if self._closest_cluster(item["embedding"]) == i
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
        axes[1, 1].set_xlabel("Similarity Score", fontsize=12)
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

    def _assign_watched_to_clusters(self):
        """Assign each watched anime to its nearest cluster"""
        assignments = {i: [] for i in range(len(self.user.embeddings))}

        for item in self.watched_embeddings:
            cluster_id = self._closest_cluster(item["embedding"])
            assignments[cluster_id].append(item)

        return assignments

    def _closest_cluster(self, embedding):
        """Find the closest cluster centroid to an embedding"""
        distances = np.linalg.norm(self.user.embeddings - embedding, axis=1)
        return np.argmin(distances)

    def save_visualizations(self, output_dir="./visualizations"):
        """Save all visualizations to files"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Main t-SNE plot
        fig1, _ = self.plot_clusters_and_recommendations()
        fig1.savefig(
            f"{output_dir}/user_{self.username}_tsne.png", dpi=300, bbox_inches="tight"
        )
        print(f"Saved: {output_dir}/user_{self.username}_tsne.png")

        # Cluster details
        fig2, _ = self.plot_cluster_details()
        fig2.savefig(
            f"{output_dir}/user_{self.username}_details.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved: {output_dir}/user_{self.username}_details.png")

        plt.close("all")


def test_user_visualization(username="MrPeanut02"):
    """
    Main test function to visualize a user's clusters and recommendations

    Parameters:
    -----------
    username : str
        MyAnimeList username to visualize
    """
    print("=" * 80)
    print(f"USER CLUSTER VISUALIZATION TEST")
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
    fig1, ax1 = visualizer.plot_clusters_and_recommendations()

    print("\n4. Generating cluster details...")
    fig2, ax2 = visualizer.plot_cluster_details()

    # Save visualizations
    print("\n5. Saving visualizations...")
    visualizer.save_visualizations()

    # Show plots
    print("\n6. Displaying plots...")
    plt.show()

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return visualizer


if __name__ == "__main__":
    # Test with default user
    visualizer = test_user_visualization("symonx99")

    # You can also test with a different user:
    # visualizer = test_user_visualization("YourUsername")
