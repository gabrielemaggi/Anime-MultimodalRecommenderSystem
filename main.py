import os
from math import nan
from pathlib import Path

import pandas as pd
import streamlit as st

from Libs.indexing_db import *
from Libs.User import User

# ---------------------------------------------------------------------------
# Configurazione Pagina
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Anime Recommendation System", layout="wide")

st.markdown(
    """
<style>
    /* Force columns to equal height */
    div[data-testid="column"] {
        display: flex;
        align-items: stretch;
    }
    /* Force the container inside the column to take full height */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    /* Fixed height for images to prevent vertical jumping */
    .stImage img {
        height: 250px;
        object-fit: cover;
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    syn_model = SynopsisEncoder().getModel()
    vis_model = VisualEncoder().getModel()
    index = Indexing()
    index.load_vector_database()
    if hasattr(vis_model, "eval"):
        vis_model.eval()
    return syn_model, vis_model, index


@st.cache_data
def get_anime_lookup():
    df = pd.read_csv("./Dataset/AnimeList.csv")
    return df.set_index("id").to_dict(orient="index")


@st.cache_data
def get_available_genres_studios():
    df = pd.read_csv("./Dataset/AnimeList.csv")
    all_genres = set()
    if "genre" in df.columns:
        for genres_str in df["genre"].dropna():
            if isinstance(genres_str, str):
                all_genres.update([g.strip() for g in genres_str.split(",")])
    all_studios = set()
    if "studio" in df.columns:
        for studio_str in df["studio"].dropna():
            if isinstance(studio_str, str):
                all_studios.update([s.strip() for s in studio_str.split(",")])
    return sorted(list(all_genres)), sorted(list(all_studios))


prompt_encoder_model, image_encoder_model, index = load_models()

# Session Status Init
if "results" not in st.session_state:
    st.session_state.results = []
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "user_object" not in st.session_state:
    st.session_state.user_object = None
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []
if "selected_studios" not in st.session_state:
    st.session_state.selected_studios = []
if "filter_mode" not in st.session_state:
    st.session_state.filter_mode = "append"
if "filter_magnitude" not in st.session_state:
    st.session_state.filter_magnitude = 1.0
if "cards_per_row" not in st.session_state:
    st.session_state.cards_per_row = 4
if "top_k_val" not in st.session_state:
    st.session_state.top_k_val = 12  # Your default value


def find_anime_image(anime_id: str) -> str | None:
    EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    for ext in EXTENSIONS:
        candidate = Path(f"./Dataset/images/{anime_id}{ext}")
        if candidate.is_file():
            return str(candidate)
    folder = Path(f"./Dataset/images/{anime_id}")
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in EXTENSIONS:
                return str(f)
    return None


def apply_filtering(
    user_obj, indexer, genres=None, studios=None, mode="append", magnitude=1.0
):
    if not genres and not studios:
        return False
    try:
        results = indexer.encode_tabular_genre_studio(
            genres=genres or [], studios=studios or []
        )
        embedding = None
        if genres and results.get("genres"):
            embedding = results.get("genres").get(genres[0])
        elif studios and results.get("studios"):
            embedding = results.get("studios").get(studios[0])
        if embedding is not None:
            query = indexer.align_embedding(embedding, modality="tab")
            user_obj.add_filtering(query, mode, magnitude)
            return True
    except Exception:
        return False
    return False


def get_user_object():
    """Helper function to get or create the user object from session state"""
    if st.session_state.user_object is None:
        u_name = st.session_state.logged_in_user
        if u_name:
            st.session_state.user_object = User(
                int(u_name) if u_name.isdigit() else u_name
            )
    return st.session_state.user_object


# ---------------------------------------------------------------------------
# Sidebar Logic
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 Anime RecSys")
    st.divider()

    if st.session_state.logged_in_user is None:
        # Login View
        st.subheader("👤 Login")
        user_input = st.text_input(
            "Username or ID", placeholder="e.g. MrPeanut02", key="login_input"
        )
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            if user_input:
                st.session_state.logged_in_user = user_input
                with st.spinner("🔄 Loading..."):
                    # Create and save user object in session state
                    user = User(int(user_input) if user_input.isdigit() else user_input)
                    user.findCentersOfClusters()
                    st.session_state.user_object = user
                    st.session_state.results = user.get_nearest_anime_from_clusters(
                        index, 12
                    )
                st.rerun()
    else:
        # 1. User Header
        st.markdown(f"### 👤 **{st.session_state.logged_in_user}**")

        # 2. Navigation
        page_selection = st.radio("Navigate to:", ["Recommendations", "User History"])
        st.divider()

        # 3. Contextual Controls (Filters only for Recommendations)
        if page_selection == "Recommendations":
            st.subheader("🔍 Refine Results")
            genres, studios = get_available_genres_studios()
            st.session_state.selected_genres = st.multiselect(
                "Genres",
                genres,
                key="genre_selector",
                default=st.session_state.selected_genres,
            )
            st.session_state.selected_studios = st.multiselect(
                "Studios",
                studios,
                key="studio_selector",
                default=st.session_state.selected_studios,
            )

            if st.session_state.selected_genres or st.session_state.selected_studios:
                with st.expander("Advanced Tuning"):
                    st.session_state.filter_mode = st.radio(
                        "Filter Mode",
                        ["append", "move"],
                        index=0 if st.session_state.filter_mode == "append" else 1,
                    )
                    st.session_state.filter_magnitude = st.slider(
                        "Magnitude", 0.0, 5.0, st.session_state.filter_magnitude, 0.1
                    )

            st.session_state.top_k_val = st.number_input(
                "Results limit",
                min_value=1,
                max_value=100,
                value=st.session_state.top_k_val,
                key="top_k_input",
            )

            if st.button("🚀 Update Results", type="primary", use_container_width=True):
                with st.spinner("🔄 Refreshing..."):
                    # Use the saved user object
                    user = get_user_object()
                    user.findCentersOfClusters()
                    apply_filtering(
                        user,
                        index,
                        st.session_state.selected_genres,
                        st.session_state.selected_studios,
                        st.session_state.filter_mode,
                        st.session_state.filter_magnitude,
                    )
                    st.session_state.results = user.get_nearest_anime_from_clusters(
                        index, st.session_state.top_k_val
                    )

                    st.rerun()
            st.divider()

        # 4. Global Visualization Settings (Always at the bottom)
        st.subheader("🖼️ Visualization")
        st.session_state.cards_per_row = st.slider(
            "Cards per Row", 2, 6, st.session_state.cards_per_row
        )

        # 5. Exit Section
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        if st.button("🚪 Change User", use_container_width=True):
            st.session_state.logged_in_user = None
            st.session_state.user_object = None
            st.session_state.results = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
if st.session_state.logged_in_user is not None:
    u_name = st.session_state.logged_in_user

    if page_selection == "Recommendations":
        st.title(f"🌟 Your Recommendations")
        if st.session_state.results:
            grid = st.session_state.cards_per_row
            for i in range(0, len(st.session_state.results), grid):
                cols = st.columns(grid)
                for j in range(grid):
                    idx = i + j
                    if idx < len(st.session_state.results):
                        anime = st.session_state.results[idx]
                        anime_id = str(anime.get("id", ""))
                        with cols[j]:
                            with st.container(border=True):
                                img = find_anime_image(anime_id)

                                # Clickable image with expander for details
                                if st.button(
                                    "🔍",
                                    key=f"view_{anime_id}_{idx}",
                                    use_container_width=True,
                                ):
                                    st.session_state[
                                        f"show_details_{anime_id}_{idx}"
                                    ] = not st.session_state.get(
                                        f"show_details_{anime_id}_{idx}", False
                                    )

                                st.image(
                                    img
                                    if img
                                    else "https://via.placeholder.com/200x300",
                                    use_container_width=True,
                                )

                                st.markdown(
                                    f"""
                                <div style="height: 80px; overflow: hidden; margin-top: 10px;">
                                    <div style="font-weight: bold; font-size: 16px; line-height: 1.2;">{anime.get("title", "Unknown")}</div>
                                    <div style="color: gray; font-size: 14px; margin-top: 5px;">
                                        🎯 Similarity: {float(anime.get("similarity", 0)) * 100:.1f}%
                                    </div>
                                </div>""",
                                    unsafe_allow_html=True,
                                )

                                # Show details if button was clicked
                                if st.session_state.get(
                                    f"show_details_{anime_id}_{idx}", False
                                ):
                                    with st.expander("📋 Details", expanded=True):
                                        lookup = get_anime_lookup()
                                        anime_info = (
                                            lookup.get(int(anime_id))
                                            if anime_id.isdigit()
                                            else None
                                        )

                                        if anime_info:
                                            # Title
                                            if anime_info.get(
                                                "title_english"
                                            ) and not pd.isna(
                                                anime_info.get("title_english")
                                            ):
                                                st.markdown(
                                                    f"**English Title:** {anime_info.get('title_english')}"
                                                )
                                            if anime_info.get(
                                                "title_japanese"
                                            ) and not pd.isna(
                                                anime_info.get("title_japanese")
                                            ):
                                                st.markdown(
                                                    f"**Japanese Title:** {anime_info.get('title_japanese')}"
                                                )

                                            st.divider()

                                            # Metadata
                                            if anime_info.get("genre") and not pd.isna(
                                                anime_info.get("genre")
                                            ):
                                                st.markdown(
                                                    f"**🎭 Genres:** {anime_info.get('genre')}"
                                                )

                                            if anime_info.get("studio") and not pd.isna(
                                                anime_info.get("studio")
                                            ):
                                                st.markdown(
                                                    f"**🎬 Studio:** {anime_info.get('studio')}"
                                                )

                                            if anime_info.get("type") and not pd.isna(
                                                anime_info.get("type")
                                            ):
                                                st.markdown(
                                                    f"**📺 Type:** {anime_info.get('type')}"
                                                )

                                            if anime_info.get(
                                                "episodes"
                                            ) and not pd.isna(
                                                anime_info.get("episodes")
                                            ):
                                                st.markdown(
                                                    f"**📊 Episodes:** {anime_info.get('episodes')}"
                                                )

                                            if anime_info.get("score") and not pd.isna(
                                                anime_info.get("score")
                                            ):
                                                st.markdown(
                                                    f"**⭐ Score:** {anime_info.get('score')}/10"
                                                )

                                            # Synopsis
                                            if anime_info.get(
                                                "synopsis"
                                            ) and not pd.isna(
                                                anime_info.get("synopsis")
                                            ):
                                                st.divider()
                                                st.markdown("**📖 Synopsis:**")
                                                st.write(anime_info.get("synopsis"))
                                        else:
                                            st.info(
                                                "No additional information available for this anime."
                                            )

                                with st.form(key=f"f_{anime_id}_{idx}"):
                                    rating = st.number_input(
                                        "Rate", 1, 10, 5, key=f"v_{anime_id}_{idx}"
                                    )
                                    if st.form_submit_button(
                                        "Submit Rating", use_container_width=True
                                    ):
                                        # We use a spinner to show the system is processing the new data
                                        with st.spinner(
                                            "🔄 Updating your profile and recommendations..."
                                        ):
                                            # 1. Use the saved user object and save rating
                                            u = get_user_object()
                                            u.add_anime(anime_id, rating)

                                            # 2. Re-calculate user clusters based on the new rating
                                            u.findCentersOfClusters()

                                            # 3. Apply current filters if any are selected
                                            if (
                                                st.session_state.selected_genres
                                                or st.session_state.selected_studios
                                            ):
                                                apply_filtering(
                                                    u,
                                                    index,
                                                    st.session_state.selected_genres,
                                                    st.session_state.selected_studios,
                                                    st.session_state.filter_mode,
                                                    st.session_state.filter_magnitude,
                                                )

                                            # 4. Refresh the session state results
                                            st.session_state.results = (
                                                u.get_nearest_anime_from_clusters(
                                                    index, st.session_state.top_k_val
                                                )
                                            )

                                        # Success message appears briefly before the rerun
                                        st.rerun()

    elif page_selection == "User History":
        st.title("📚 Your Watch History")
        # Use the saved user object
        user = get_user_object()
        raw_watchlist = user.get_watchList()

        if not raw_watchlist:
            st.info("Your watchlist is currently empty.")
        else:
            lookup = get_anime_lookup()
            clean_history = []
            for item in raw_watchlist:
                a_id, score = item[0], item[1]
                info = lookup.get(a_id)
                title = (
                    info.get("title_english")
                    if info and not pd.isna(info.get("title_english"))
                    else (info.get("title_japanese") if info else f"ID: {a_id}")
                )
                clean_history.append({"id": a_id, "title": title, "rating": score})

            grid = st.session_state.cards_per_row
            for i in range(0, len(clean_history), grid):
                cols = st.columns(grid)
                for j in range(grid):
                    idx = i + j
                    if idx < len(clean_history):
                        item = clean_history[idx]
                        with cols[j]:
                            with st.container(border=True):
                                # Immagine con altezza fissa (gestita dal CSS sopra)
                                img = find_anime_image(str(item["id"]))

                                # Clickable button for details
                                if st.button(
                                    "🔍",
                                    key=f"history_view_{item['id']}_{idx}",
                                    use_container_width=True,
                                ):
                                    st.session_state[
                                        f"show_history_details_{item['id']}_{idx}"
                                    ] = not st.session_state.get(
                                        f"show_history_details_{item['id']}_{idx}",
                                        False,
                                    )

                                st.image(
                                    img
                                    if img
                                    else "https://via.placeholder.com/200x300",
                                    use_container_width=True,
                                )

                                # Wrapper a dimensione fissa per Titolo e Rating
                                # Usiamo 80px come nelle recommendations per coerenza
                                rating_val = item["rating"]
                                rating_display = (
                                    f"⭐ {rating_val}/10"
                                    if (rating_val and rating_val > 0)
                                    else "📝 Not Rated"
                                )

                                st.markdown(
                                    f"""
                                    <div style="height: 80px; overflow: hidden; margin-top: 10px;">
                                        <div style="font-weight: bold; font-size: 16px; line-height: 1.2;">{item["title"]}</div>
                                        <div style="color: #ffaa00; font-size: 14px; margin-top: 5px; font-weight: bold;">
                                            {rating_display}
                                        </div>
                                    </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                                # Show details if button was clicked
                                if st.session_state.get(
                                    f"show_history_details_{item['id']}_{idx}", False
                                ):
                                    with st.expander("📋 Details", expanded=True):
                                        anime_info = lookup.get(item["id"])

                                        if anime_info:
                                            # Title
                                            if anime_info.get(
                                                "title_english"
                                            ) and not pd.isna(
                                                anime_info.get("title_english")
                                            ):
                                                st.markdown(
                                                    f"**English Title:** {anime_info.get('title_english')}"
                                                )
                                            if anime_info.get(
                                                "title_japanese"
                                            ) and not pd.isna(
                                                anime_info.get("title_japanese")
                                            ):
                                                st.markdown(
                                                    f"**Japanese Title:** {anime_info.get('title_japanese')}"
                                                )

                                            st.divider()

                                            # Metadata
                                            if anime_info.get("genre") and not pd.isna(
                                                anime_info.get("genre")
                                            ):
                                                st.markdown(
                                                    f"**🎭 Genres:** {anime_info.get('genre')}"
                                                )

                                            if anime_info.get("studio") and not pd.isna(
                                                anime_info.get("studio")
                                            ):
                                                st.markdown(
                                                    f"**🎬 Studio:** {anime_info.get('studio')}"
                                                )

                                            if anime_info.get("type") and not pd.isna(
                                                anime_info.get("type")
                                            ):
                                                st.markdown(
                                                    f"**📺 Type:** {anime_info.get('type')}"
                                                )

                                            if anime_info.get(
                                                "episodes"
                                            ) and not pd.isna(
                                                anime_info.get("episodes")
                                            ):
                                                st.markdown(
                                                    f"**📊 Episodes:** {anime_info.get('episodes')}"
                                                )

                                            if anime_info.get("score") and not pd.isna(
                                                anime_info.get("score")
                                            ):
                                                st.markdown(
                                                    f"**⭐ Score:** {anime_info.get('score')}/10"
                                                )

                                            # Synopsis
                                            if anime_info.get(
                                                "synopsis"
                                            ) and not pd.isna(
                                                anime_info.get("synopsis")
                                            ):
                                                st.divider()
                                                st.markdown("**📖 Synopsis:**")
                                                st.write(anime_info.get("synopsis"))
                                        else:
                                            st.info(
                                                "No additional information available for this anime."
                                            )
else:
    st.title("👋 Welcome")
    st.info("Enter your username or ID in the sidebar to start.")
