import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from mal import client
from PIL import Image

from Libs.indexing_db import Indexing


class AnimeDBManager:
    """
    Manages the addition of new anime to the vector database.
    Handles API fetching, image downloading, and database integration.
    """

    def __init__(self, api_id: str, indexing: Optional[Indexing] = None):
        """
        Initialize the AnimeDBManager

        Args:
            api_id: Your MAL API ID
            indexing: Existing Indexing instance (creates new if None)
        """
        self.api = client.Client(api_id)
        self.indexing = indexing if indexing else Indexing()
        self.image_dir = Path("./Dataset/images/")
        self.dataset_path = Path("./Dataset/AnimeList.csv")

        # Ensure directories exist
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Load the indexing database if not already loaded
        if self.indexing.vector_db is None:
            self.indexing.load_vector_database()

    def search_anime(self, query: str, limit: int = 10) -> List:
        """
        Search for anime using MAL API

        Args:
            query: Search query string (3-64 characters)
            limit: Maximum number of results

        Returns:
            List of anime search results
        """
        try:
            results = self.api.anime_search(
                query=query,
                limit=limit,
                fields=[
                    "id",
                    "title",
                    "main_picture",
                    "synopsis",
                    "genres",
                    "mean",
                    "num_scoring_users",
                    "popularity",
                    "rating",
                    "studios",
                    "alternative_titles",
                ],
                include_nsfw=True,
            )
            return list(results)
        except ValueError as e:
            raise ValueError(f"Search query must be between 3 and 64 characters: {e}")

    def get_anime_details(self, anime_id: int) -> Dict:
        """
        Get detailed information for a specific anime

        Args:
            anime_id: The MAL anime ID

        Returns:
            Dictionary with anime details
        """
        anime = self.api.get_anime(
            id=anime_id,
            fields=[
                "id",
                "title",
                "main_picture",
                "synopsis",
                "genres",
                "mean",
                "num_scoring_users",
                "popularity",
                "rating",
                "studios",
                "alternative_titles",
                "media_type",
                "status",
            ],
        )
        return anime

    def extract_anime_data(self, anime_obj) -> Dict:
        """
        Extract and format anime data from MAL API response to match CSV structure

        Args:
            anime_obj: Anime object from MAL API

        Returns:
            Formatted dictionary with anime data matching CSV columns
        """
        # Extract basic info
        data = {
            "id": anime_obj.id,
            "title": anime_obj.title,
            "title_english": getattr(anime_obj.alternative_titles, "en", "")
            if hasattr(anime_obj, "alternative_titles")
            else "",
            "title_japanese": getattr(anime_obj.alternative_titles, "ja", "")
            if hasattr(anime_obj, "alternative_titles")
            else "",
            "title_synonyms": ", ".join(
                getattr(anime_obj.alternative_titles, "synonyms", [])
            )
            if hasattr(anime_obj, "alternative_titles")
            else "",
            # Image
            "image_url": anime_obj.main_picture_url
            if anime_obj.main_picture_url
            else "",
            # Type and source
            "type": anime_obj.media_type
            if hasattr(anime_obj, "media_type") and anime_obj.media_type
            else "",
            "source": anime_obj.source
            if hasattr(anime_obj, "source") and anime_obj.source
            else "",
            # Episodes and status
            "episodes": anime_obj.num_episodes
            if hasattr(anime_obj, "num_episodes") and anime_obj.num_episodes
            else 0,
            "status": anime_obj.status
            if hasattr(anime_obj, "status") and anime_obj.status
            else "",
            "airing": anime_obj.is_airing if hasattr(anime_obj, "is_airing") else False,
            # Dates
            "aired_string": self._format_aired_string(anime_obj),
            "aired": self._format_aired_dict(anime_obj),
            # Duration and rating
            "duration": anime_obj.average_episode_duration
            if hasattr(anime_obj, "average_episode_duration")
            and anime_obj.average_episode_duration
            else 0,
            "rating": anime_obj.rating
            if hasattr(anime_obj, "rating") and anime_obj.rating
            else "",
            # Scores
            "score": anime_obj.mean
            if hasattr(anime_obj, "mean") and anime_obj.mean
            else 0.0,
            "scored_by": anime_obj.num_scoring_users
            if hasattr(anime_obj, "num_scoring_users") and anime_obj.num_scoring_users
            else 0,
            "rank": anime_obj.rank
            if hasattr(anime_obj, "rank") and anime_obj.rank
            else 0,
            "popularity": anime_obj.popularity
            if hasattr(anime_obj, "popularity") and anime_obj.popularity
            else 0,
            # Members and favorites
            "members": anime_obj.num_list_users
            if hasattr(anime_obj, "num_list_users") and anime_obj.num_list_users
            else 0,
            "favorites": anime_obj.num_favorites
            if hasattr(anime_obj, "num_favorites") and anime_obj.num_favorites
            else 0,
            # Background
            "background": anime_obj.background
            if hasattr(anime_obj, "background") and anime_obj.background
            else "",
            # Premiered and broadcast
            "premiered": anime_obj.start_season
            if hasattr(anime_obj, "start_season") and anime_obj.start_season
            else "",
            "broadcast": self._format_broadcast(anime_obj),
            # Related (needs special handling)
            "related": self._format_related(anime_obj),
            # Production info - these might not be available in the API response
            "producer": "",  # MAL API doesn't provide this separately
            "licensor": "",  # MAL API doesn't provide this separately
            "studio": anime_obj.studios
            if hasattr(anime_obj, "studios") and anime_obj.studios
            else "",
            "studios": anime_obj.studios
            if hasattr(anime_obj, "studios") and anime_obj.studios
            else "",
            # Genres
            "genre": ", ".join([str(g) for g in anime_obj.genres])
            if hasattr(anime_obj, "genres") and anime_obj.genres
            else "",
            # Themes
            "opening_theme": ", ".join(anime_obj.openings)
            if hasattr(anime_obj, "openings") and anime_obj.openings
            else "",
            "ending_theme": ", ".join(anime_obj.endings)
            if hasattr(anime_obj, "endings") and anime_obj.endings
            else "",
            # Synopsis
            "sypnopsis": anime_obj.synopsis
            if hasattr(anime_obj, "synopsis") and anime_obj.synopsis
            else "",
        }

        return data

    def _format_aired_string(self, anime_obj) -> str:
        """Format the aired date range as a string"""
        try:
            start = (
                anime_obj.start_date
                if hasattr(anime_obj, "start_date") and anime_obj.start_date
                else None
            )
            end = (
                anime_obj.end_date
                if hasattr(anime_obj, "end_date") and anime_obj.end_date
                else None
            )

            if start and end:
                return f"{start} to {end}"
            elif start:
                return f"{start} to ?"
            else:
                return "Not available"
        except:
            return "Not available"

    def _format_aired_dict(self, anime_obj) -> str:
        """Format aired information as a dict-like string"""
        try:
            start = (
                anime_obj.start_date
                if hasattr(anime_obj, "start_date") and anime_obj.start_date
                else None
            )
            end = (
                anime_obj.end_date
                if hasattr(anime_obj, "end_date") and anime_obj.end_date
                else None
            )

            return str(
                {"from": str(start) if start else None, "to": str(end) if end else None}
            )
        except:
            return str({"from": None, "to": None})

    def _format_broadcast(self, anime_obj) -> str:
        """Format broadcast information"""
        try:
            if hasattr(anime_obj, "broadcast_day") and anime_obj.broadcast_day:
                day = anime_obj.broadcast_day
                time = (
                    anime_obj.broadcast_time
                    if hasattr(anime_obj, "broadcast_time") and anime_obj.broadcast_time
                    else ""
                )
                return f"{day}s at {time}" if time else f"{day}s"
            return ""
        except:
            return ""

    def _format_related(self, anime_obj) -> str:
        """Format related anime information"""
        try:
            if hasattr(anime_obj, "related_anime") and anime_obj.related_anime:
                related_list = []
                for rel in anime_obj.related_anime:
                    rel_type = getattr(rel, "relation_type", "Related")
                    rel_title = getattr(rel.node, "title", "Unknown")
                    related_list.append(f"{rel_type}: {rel_title}")
                return ", ".join(related_list)
            return ""
        except:
            return ""

    def download_anime_image(self, anime_obj, anime_id: int) -> str:
        """
        Download anime poster image from anime object

        Args:
            anime_obj: Anime object from MAL API
            anime_id: Anime ID for filename

        Returns:
            Path to saved image
        """
        try:
            # Get the main picture URL using the property
            image_url = anime_obj.main_picture_url

            if not image_url:
                raise ValueError(f"No image URL available for anime ID {anime_id}")

            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Open image and save as jpg
            img = Image.open(BytesIO(response.content))

            # Convert to RGB if necessary (handles PNG transparency)
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background

            # Save image
            image_path = self.image_dir / f"{anime_id}.jpg"
            img.save(image_path, "JPEG", quality=95)

            print(f"✅ Image downloaded: {image_path}")
            return str(image_path)

        except Exception as e:
            print(f"❌ Failed to download image: {e}")
            raise

    def add_anime_to_csv(self, data: Dict):
        """
        Add anime data to the AnimeList.csv file

        Args:
            data: Dictionary with anime data
        """
        try:
            # Load existing CSV
            if self.dataset_path.exists():
                df = pd.read_csv(self.dataset_path)
            else:
                df = pd.DataFrame()

            # Check if anime already exists
            if "id" in df.columns and data["id"] in df["id"].values:
                print(f"⚠️  Anime ID {data['id']} already exists in CSV, updating...")
                df.loc[df["id"] == data["id"], list(data.keys())] = list(data.values())
            else:
                # Add new row
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

            # Save back to CSV
            df.to_csv(self.dataset_path, index=False)
            print(f"✅ Anime added to CSV: {data['title']}")

        except Exception as e:
            print(f"❌ Failed to add anime to CSV: {e}")
            raise

    def add_anime_by_id(self, anime_id: int) -> Dict:
        """
        Complete pipeline: fetch anime by ID, download image, add to DB

        Args:
            anime_id: MAL anime ID

        Returns:
            Dictionary with anime data
        """
        print(f"\n{'=' * 60}")
        print(f"Adding anime ID: {anime_id}")
        print(f"{'=' * 60}\n")

        # 1. Get anime details from API
        print("📡 Fetching anime details from MAL API...")
        anime_obj = self.get_anime_details(anime_id)

        # 2. Extract and format data
        print("📝 Extracting anime data...")
        data = self.extract_anime_data(anime_obj)

        # 3. Download image - pass the whole anime object
        print("🖼️  Downloading anime poster...")
        image_path = self.download_anime_image(anime_obj, anime_id)

        # 4. Add to CSV dataset
        print("💾 Adding to CSV dataset...")
        self.add_anime_to_csv(data)

        # 5. Add to vector database (embeddings + fusion)
        print("🧮 Encoding and adding to vector database...")
        self.indexing.add_new_anime_to_db(data, image_path=image_path)

        print(f"\n{'=' * 60}")
        print(f"✅ Successfully added '{data['title']}' to the database!")
        print(f"{'=' * 60}\n")

        return data

    def add_anime_by_search(self, query: str, index: int = 0) -> Dict:
        """
        Search for anime and add the result at specified index

        Args:
            query: Search query
            index: Which result to add (default: 0, the first result)

        Returns:
            Dictionary with anime data
        """
        print(f"🔍 Searching for: '{query}'...")
        results = self.search_anime(query, limit=10)

        if not results:
            raise ValueError(f"No results found for query: {query}")

        if index >= len(results):
            raise ValueError(
                f"Index {index} out of range. Found {len(results)} results."
            )

        # Display results
        print(f"\n📋 Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  {i}. {result.title} (ID: {result.id})")

        print(f"\n✅ Selecting result #{index}: {results[index].title}\n")

        # Add the selected anime
        return self.add_anime_by_id(results[index].id)

    def batch_add_anime(self, anime_ids: List[int]) -> List[Dict]:
        """
        Add multiple anime to the database

        Args:
            anime_ids: List of MAL anime IDs

        Returns:
            List of added anime data
        """
        added_anime = []
        failed_ids = []

        print(f"\n{'=' * 60}")
        print(f"Batch adding {len(anime_ids)} anime")
        print(f"{'=' * 60}\n")

        for i, anime_id in enumerate(anime_ids, 1):
            try:
                print(f"[{i}/{len(anime_ids)}] Processing anime ID: {anime_id}")
                data = self.add_anime_by_id(anime_id)
                added_anime.append(data)
            except Exception as e:
                print(f"❌ Failed to add anime ID {anime_id}: {e}")
                failed_ids.append(anime_id)

        print(f"\n{'=' * 60}")
        print(f"✅ Successfully added: {len(added_anime)}/{len(anime_ids)}")
        if failed_ids:
            print(f"❌ Failed IDs: {failed_ids}")
        print(f"{'=' * 60}\n")

        return added_anime


# Example usage
if __name__ == "__main__":
    API_ID = "d79a8a3b8f42750e317b0b7abc47adf2"

    # Initialize manager
    manager = AnimeDBManager(API_ID)

    # Example 1: Add anime by ID
    # manager.add_anime_by_id(anime_id=1)  # Cowboy Bebop

    # Example 2: Add anime by search
    manager.add_anime_by_search("Steins Gate", index=0)

    # Example 3: Batch add multiple anime
    # anime_ids = [1, 5, 20, 30, 40]
    # manager.batch_add_anime(anime_ids)
