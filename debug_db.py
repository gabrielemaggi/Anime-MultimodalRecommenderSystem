import pickle
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class VectorDatabaseDebugger:
    def __init__(self, index_path, metadata_path, dataset_path):
        """
        Initialize debugger

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            dataset_path: Path to original CSV dataset
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dataset_path = dataset_path

        # Load database
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Load original dataset
        self.df = pd.read_csv(dataset_path)

        print(f"✅ Loaded database:")
        print(f"   - Index size: {self.index.ntotal} vectors")
        print(f"   - Metadata entries: {len(self.metadata)}")
        print(f"   - Dimension: {self.index.d}")
        print(f"   - Original dataset: {len(self.df)} rows")

    def extract_all_embeddings(self):
        """Extract all embeddings from FAISS index"""
        n_vectors = self.index.ntotal
        embeddings = np.zeros((n_vectors, self.index.d), dtype="float32")

        for i in range(n_vectors):
            embeddings[i] = self.index.reconstruct(i)

        return embeddings

    def validate_id_mapping(self):
        """Check if database IDs match the original dataset"""
        print("\n" + "=" * 80)
        print("ID MAPPING VALIDATION")
        print("=" * 80)

        db_ids = []
        db_titles = []

        for idx, meta in enumerate(self.metadata):
            db_id = meta.get("id")
            db_title = meta.get("title", "N/A")
            db_ids.append(db_id)
            db_titles.append(db_title)

        db_ids_set = set(str(id_) for id_ in db_ids)
        dataset_ids_set = set(str(id_) for id_ in self.df["id"].values)

        # Find discrepancies
        missing_in_db = dataset_ids_set - db_ids_set
        extra_in_db = db_ids_set - dataset_ids_set

        print(f"\n📊 ID Coverage:")
        print(f"   - Total in dataset: {len(dataset_ids_set)}")
        print(f"   - Total in database: {len(db_ids_set)}")
        print(f"   - Missing from database: {len(missing_in_db)}")
        print(f"   - Extra in database (not in dataset): {len(extra_in_db)}")

        if len(missing_in_db) > 0:
            print(f"\n⚠️  First 10 missing IDs: {list(missing_in_db)[:10]}")

        if len(extra_in_db) > 0:
            print(f"\n⚠️  First 10 extra IDs: {list(extra_in_db)[:10]}")

        # Validate title matching
        print(f"\n📝 Title Validation (checking first 10):")
        for i in range(min(10, len(self.metadata))):
            db_id = str(self.metadata[i].get("id"))
            db_title = self.metadata[i].get("title", "N/A")

            # Find in dataset
            dataset_row = self.df[self.df["id"] == int(db_id)]
            if not dataset_row.empty:
                dataset_title = dataset_row.iloc[0]["title"]
                match = "✅" if db_title == dataset_title else "❌"
                print(
                    f"   {match} ID {db_id}: DB='{db_title}' | Dataset='{dataset_title}'"
                )
            else:
                print(f"   ❌ ID {db_id}: Not found in dataset! Title='{db_title}'")

        return {
            "db_ids": db_ids,
            "db_titles": db_titles,
            "missing_in_db": missing_in_db,
            "extra_in_db": extra_in_db,
        }

    def check_embedding_quality(self, embeddings):
        """Check for common embedding issues"""
        print("\n" + "=" * 80)
        print("EMBEDDING QUALITY CHECKS")
        print("=" * 80)

        # Check for all-zeros embeddings
        zero_embeddings = np.all(embeddings == 0, axis=1)
        print(f"\n🔍 Zero embeddings: {zero_embeddings.sum()} / {len(embeddings)}")

        # Check for duplicate embeddings
        unique_embeddings = np.unique(embeddings, axis=0)
        print(f"🔍 Unique embeddings: {len(unique_embeddings)} / {len(embeddings)}")
        print(f"   Duplicates: {len(embeddings) - len(unique_embeddings)}")

        # Check embedding norms
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"\n📏 Embedding norms:")
        print(f"   Mean: {norms.mean():.6f}")
        print(f"   Std:  {norms.std():.6f}")
        print(f"   Min:  {norms.min():.6f}")
        print(f"   Max:  {norms.max():.6f}")

        if isinstance(self.index, faiss.IndexFlatIP):
            # For cosine similarity, norms should be ~1.0
            if not np.allclose(norms, 1.0, atol=0.01):
                print(
                    f"   ⚠️  WARNING: Using cosine similarity but embeddings not normalized!"
                )

        # Check for NaN or Inf
        has_nan = np.isnan(embeddings).any()
        has_inf = np.isinf(embeddings).any()
        print(f"\n🚨 NaN values: {has_nan}")
        print(f"🚨 Inf values: {has_inf}")

        # Distribution of values
        print(f"\n📊 Value distribution:")
        print(f"   Mean: {embeddings.mean():.6f}")
        print(f"   Std:  {embeddings.std():.6f}")
        print(f"   Min:  {embeddings.min():.6f}")
        print(f"   Max:  {embeddings.max():.6f}")

    def plot_embedding_space(self, embeddings, save_path="embedding_visualization.png"):
        """Visualize embeddings in 2D using t-SNE"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATION")
        print("=" * 80)

        # Sample if too many (t-SNE is slow)
        max_samples = 5000
        if len(embeddings) > max_samples:
            print(f"⏳ Sampling {max_samples} points for visualization...")
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings_subset = embeddings[indices]
            metadata_subset = [self.metadata[i] for i in indices]
        else:
            embeddings_subset = embeddings
            metadata_subset = self.metadata
            indices = np.arange(len(embeddings))

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

        # 1. PCA visualization
        print("📊 Computing PCA...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings_subset)

        axes[0, 0].scatter(
            embeddings_pca[:, 0],
            embeddings_pca[:, 1],
            alpha=0.5,
            s=10,
            c=range(len(embeddings_pca)),
            cmap="viridis",
        )
        axes[0, 0].set_title("PCA Projection", fontsize=16, fontweight="bold")
        axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. t-SNE visualization
        print("📊 Computing t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_tsne = tsne.fit_transform(embeddings_subset)

        axes[0, 1].scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            alpha=0.5,
            s=10,
            c=range(len(embeddings_tsne)),
            cmap="viridis",
        )
        axes[0, 1].set_title("t-SNE Projection", fontsize=16, fontweight="bold")
        axes[0, 1].set_xlabel("t-SNE 1")
        axes[0, 1].set_ylabel("t-SNE 2")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Embedding norm distribution
        norms = np.linalg.norm(embeddings, axis=1)
        axes[1, 0].hist(norms, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(
            norms.mean(), color="red", linestyle="--", label=f"Mean: {norms.mean():.4f}"
        )
        axes[1, 0].set_title(
            "Embedding Norm Distribution", fontsize=16, fontweight="bold"
        )
        axes[1, 0].set_xlabel("L2 Norm")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Pairwise similarity heatmap (sample)
        sample_size = min(100, len(embeddings_subset))
        sample_indices = np.random.choice(
            len(embeddings_subset), sample_size, replace=False
        )
        sample_embeddings = embeddings_subset[sample_indices]

        # Normalize for cosine similarity
        sample_embeddings_norm = sample_embeddings / np.linalg.norm(
            sample_embeddings, axis=1, keepdims=True
        )
        similarity_matrix = np.dot(sample_embeddings_norm, sample_embeddings_norm.T)

        im = axes[1, 1].imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, 1].set_title(
            f"Pairwise Cosine Similarity ({sample_size} samples)",
            fontsize=16,
            fontweight="bold",
        )
        axes[1, 1].set_xlabel("Sample Index")
        axes[1, 1].set_ylabel("Sample Index")
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Visualization saved to {save_path}")
        plt.show()

    def create_id_mapping_log(self, output_path="database_id_mapping.csv"):
        """Create a detailed CSV log of all database entries"""
        print("\n" + "=" * 80)
        print("CREATING ID MAPPING LOG")
        print("=" * 80)

        records = []

        for idx, meta in enumerate(self.metadata):
            db_id = str(meta.get("id"))
            db_title = meta.get("title", "N/A")

            # Find in dataset
            dataset_row = self.df[self.df["id"] == int(db_id)]

            if not dataset_row.empty:
                dataset_title = dataset_row.iloc[0]["title"]
                title_match = db_title == dataset_title
                in_dataset = True
            else:
                dataset_title = "NOT FOUND"
                title_match = False
                in_dataset = False

            records.append(
                {
                    "db_index": idx,
                    "db_id": db_id,
                    "db_title": db_title,
                    "dataset_title": dataset_title,
                    "in_dataset": in_dataset,
                    "title_match": title_match,
                    "genre": meta.get("genre", "N/A"),
                    "synopsis_preview": str(meta.get("sypnopsis", "N/A"))[:100],
                }
            )

        log_df = pd.DataFrame(records)
        log_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ ID mapping log saved to {output_path}")
        print(f"\n📊 Summary:")
        print(f"   - Total entries: {len(log_df)}")
        print(f"   - In dataset: {log_df['in_dataset'].sum()}")
        print(f"   - Title matches: {log_df['title_match'].sum()}")
        print(f"   - Mismatches: {(~log_df['title_match']).sum()}")

        return log_df

    def test_specific_ids(self, test_ids):
        """Test specific anime IDs"""
        print("\n" + "=" * 80)
        print("TESTING SPECIFIC IDs")
        print("=" * 80)

        for test_id in test_ids:
            print(f"\n🔍 Testing ID: {test_id}")

            # Check in metadata
            found_in_db = False
            db_index = None
            for idx, meta in enumerate(self.metadata):
                if str(meta.get("id")) == str(test_id):
                    found_in_db = True
                    db_index = idx
                    print(f"   ✅ Found in database at index {idx}")
                    print(f"      Title: {meta.get('title')}")
                    print(f"      Genre: {meta.get('genre')}")

                    # Get embedding
                    embedding = self.index.reconstruct(idx)
                    norm = np.linalg.norm(embedding)
                    print(f"      Embedding norm: {norm:.6f}")
                    print(f"      First 5 values: {embedding[:5]}")
                    break

            if not found_in_db:
                print(f"   ❌ NOT found in database")

            # Check in dataset
            dataset_row = self.df[self.df["id"] == int(test_id)]
            if not dataset_row.empty:
                print(f"   ✅ Found in original dataset")
                print(f"      Title: {dataset_row.iloc[0]['title']}")
            else:
                print(f"   ❌ NOT found in original dataset")

    def run_full_diagnostic(self):
        """Run all diagnostics"""
        print("\n" + "=" * 80)
        print("VECTOR DATABASE FULL DIAGNOSTIC")
        print("=" * 80)

        # Extract embeddings
        print("\n⏳ Extracting embeddings from FAISS index...")
        embeddings = self.extract_all_embeddings()

        # Run all checks
        id_validation = self.validate_id_mapping()
        self.check_embedding_quality(embeddings)
        log_df = self.create_id_mapping_log()

        # Visualizations
        self.plot_embedding_space(embeddings)

        return {
            "embeddings": embeddings,
            "id_validation": id_validation,
            "log_df": log_df,
        }


# ==================== USAGE ====================
if __name__ == "__main__":
    # Initialize debugger
    debugger = VectorDatabaseDebugger(
        index_path="./Embeddings/Attention_AnimeVecDb.index",
        metadata_path="./Embeddings/Attention_AnimeVecDb.pkl",
        dataset_path="./Dataset/AnimeList.csv",
    )

    # Run full diagnostic
    results = debugger.run_full_diagnostic()

    # Test specific problematic IDs
    debugger.test_specific_ids(["877", "23055", "5114"])  # NGE, JoJo, FMA:B

    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - embedding_visualization.png")
    print("  - database_id_mapping.csv")
