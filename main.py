import os
from math import nan
from pathlib import Path

import pandas as pd
import streamlit as st

from Libs import GoalParsing
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
    /* Responsive images with fixed aspect ratio */
    .stImage img {
        width: 100%;
        height: auto;
        aspect-ratio: 2/3;
        object-fit: cover;
        border-radius: 5px;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stImage img:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
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


@st.cache_resource
def load_goal_parser():
    """Load the goal parsing system"""
    return GoalParsing()


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
goal_parser = load_goal_parser()

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
    st.session_state.top_k_val = 12
if "text_goal" not in st.session_state:
    st.session_state.text_goal = ""
if "synopsis_text" not in st.session_state:
    st.session_state.synopsis_text = ""
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "filter_type" not in st.session_state:
    st.session_state.filter_type = "manual"  # "manual", "text", "synopsis", or "image"


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


def apply_text_goal(user_obj, indexer, goal_text, parser):
    """Apply text-based goal parsing"""
    if not goal_text or not goal_text.strip():
        return False
    try:
        parser.process_request(goal_text, user_obj, indexer)
        return True
    except Exception as e:
        st.error(f"Error processing text goal: {str(e)}")
        return False


def apply_synopsis_filter(user_obj, indexer, synopsis_text, parser):
    """Apply synopsis-based filtering"""
    if not synopsis_text or not synopsis_text.strip():
        return False
    try:
        parser.process_sypnopsis(synopsis_text, user_obj, indexer)
        return True
    except Exception as e:
        st.error(f"Error processing synopsis: {str(e)}")
        return False


def apply_image_filter(user_obj, indexer, image, parser):
    """Apply image-based filtering"""
    if image is None:
        return False
    try:
        parser.process_image(image, user_obj, indexer)
        return True
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
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
                    user.findCentersOfClusters(index)
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

            # Filter Type Selection - simpler approach
            filter_options = [
                "Manual Selection",
                "Text Description",
                "Synopsis Search",
                "Image Search",
            ]
            current_index = 0
            if st.session_state.filter_type == "text":
                current_index = 1
            elif st.session_state.filter_type == "synopsis":
                current_index = 2
            elif st.session_state.filter_type == "image":
                current_index = 3

            selected_filter = st.radio(
                "Filter Method:",
                filter_options,
                index=current_index,
                key="filter_type_radio",
            )

            # Update internal state based on selection
            if selected_filter == "Manual Selection":
                st.session_state.filter_type = "manual"
            elif selected_filter == "Text Description":
                st.session_state.filter_type = "text"
            elif selected_filter == "Synopsis Search":
                st.session_state.filter_type = "synopsis"
            elif selected_filter == "Image Search":
                st.session_state.filter_type = "image"

            st.divider()

            # Manual Filter Mode
            if st.session_state.filter_type == "manual":
                st.markdown("**🎯 Manual Filters**")
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

                if (
                    st.session_state.selected_genres
                    or st.session_state.selected_studios
                ):
                    with st.expander("Advanced Tuning"):
                        st.session_state.filter_mode = st.radio(
                            "Filter Mode",
                            ["append", "move"],
                            index=0 if st.session_state.filter_mode == "append" else 1,
                        )
                        st.session_state.filter_magnitude = st.slider(
                            "Magnitude",
                            0.0,
                            5.0,
                            st.session_state.filter_magnitude,
                            0.1,
                        )

            # Text Goal Mode
            elif st.session_state.filter_type == "text":
                st.markdown("**💬 Describe Your Preferences**")
                st.caption(
                    "Example: 'I want a Romance anime, in the style of mappa studio'"
                )
                st.session_state.text_goal = st.text_area(
                    "What are you looking for?",
                    value=st.session_state.text_goal,
                    placeholder="Describe what kind of anime you want...",
                    height=100,
                    key="text_goal_input",
                )

                if st.session_state.text_goal:
                    st.info(
                        "🤖 AI will parse your request and apply appropriate filters"
                    )

            # Synopsis Mode
            elif st.session_state.filter_type == "synopsis":
                st.markdown("**📖 Search by Synopsis**")
                st.caption("Describe the plot or story you're looking for")
                st.session_state.synopsis_text = st.text_area(
                    "Synopsis/Plot Description",
                    value=st.session_state.synopsis_text,
                    placeholder="E.g., 'A story about time travel and parallel universes...'",
                    height=120,
                    key="synopsis_input",
                )

                if st.session_state.synopsis_text:
                    st.info("📚 AI will find anime with similar storylines")

            # Image Mode
            elif st.session_state.filter_type == "image":
                st.markdown("**🖼️ Search by Image**")
                st.caption("Upload an anime image to find similar styles")
                uploaded_file = st.file_uploader(
                    "Upload Image",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="image_uploader",
                )

                if uploaded_file is not None:
                    st.session_state.uploaded_image = uploaded_file
                    # Show preview
                    st.image(
                        uploaded_file,
                        caption="Uploaded Image",
                        use_container_width=True,
                    )
                    st.info("🎨 AI will find anime with similar visual style")

            st.divider()

            # Results Limit
            st.session_state.top_k_val = st.number_input(
                "Results limit",
                min_value=1,
                max_value=100,
                value=st.session_state.top_k_val,
                key="top_k_input",
            )

            # Update Button
            if st.button("🚀 Update Results", type="primary", use_container_width=True):
                with st.spinner("🔄 Refreshing..."):
                    # Use the saved user object
                    user = get_user_object()
                    user.findCentersOfClusters(index)

                    # Apply appropriate filtering based on mode
                    if st.session_state.filter_type == "manual":
                        apply_filtering(
                            user,
                            index,
                            st.session_state.selected_genres,
                            st.session_state.selected_studios,
                            st.session_state.filter_mode,
                            st.session_state.filter_magnitude,
                        )
                    elif st.session_state.filter_type == "text":
                        apply_text_goal(
                            user, index, st.session_state.text_goal, goal_parser
                        )
                    elif st.session_state.filter_type == "synopsis":
                        apply_synopsis_filter(
                            user, index, st.session_state.synopsis_text, goal_parser
                        )
                    elif st.session_state.filter_type == "image":
                        if st.session_state.uploaded_image:
                            apply_image_filter(
                                user,
                                index,
                                st.session_state.uploaded_image,
                                goal_parser,
                            )
                        else:
                            st.warning("⚠️ Please upload an image first")
                            st.stop()

                    results = user.get_nearest_anime_from_clusters(
                        index, st.session_state.top_k_val
                    )

                    st.session_state.results = results[: st.session_state.top_k_val]
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
            st.session_state.text_goal = ""
            st.session_state.synopsis_text = ""
            st.session_state.uploaded_image = None
            st.session_state.selected_genres = []
            st.session_state.selected_studios = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
if st.session_state.logged_in_user is not None:
    u_name = st.session_state.logged_in_user

    if page_selection == "Recommendations":
        st.title(f"🌟 Your Recommendations")

        # Show active filter info
        if st.session_state.filter_type == "manual" and (
            st.session_state.selected_genres or st.session_state.selected_studios
        ):
            filter_info = []
            if st.session_state.selected_genres:
                filter_info.append(
                    f"Genres: {', '.join(st.session_state.selected_genres)}"
                )
            if st.session_state.selected_studios:
                filter_info.append(
                    f"Studios: {', '.join(st.session_state.selected_studios)}"
                )
            st.info(f"🎯 Active Filters: {' | '.join(filter_info)}")
        elif st.session_state.filter_type == "text" and st.session_state.text_goal:
            st.info(f"💬 Goal: {st.session_state.text_goal}")
        elif (
            st.session_state.filter_type == "synopsis"
            and st.session_state.synopsis_text
        ):
            st.info(
                f"📖 Synopsis Filter: {st.session_state.synopsis_text[:100]}{'...' if len(st.session_state.synopsis_text) > 100 else ''}"
            )
        elif (
            st.session_state.filter_type == "image" and st.session_state.uploaded_image
        ):
            st.info(f"🖼️ Image Filter: {st.session_state.uploaded_image.name}")

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

                                # Clickable image - clicking toggles details
                                if st.button(
                                    "Click to view details",
                                    key=f"img_btn_{anime_id}_{idx}",
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
                                            u.findCentersOfClusters(index)

                                            # 3. Apply current filters if any are selected
                                            if st.session_state.filter_type == "manual":
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
                                            elif st.session_state.filter_type == "text":
                                                if st.session_state.text_goal:
                                                    apply_text_goal(
                                                        u,
                                                        index,
                                                        st.session_state.text_goal,
                                                        goal_parser,
                                                    )
                                            elif (
                                                st.session_state.filter_type
                                                == "synopsis"
                                            ):
                                                if st.session_state.synopsis_text:
                                                    apply_synopsis_filter(
                                                        u,
                                                        index,
                                                        st.session_state.synopsis_text,
                                                        goal_parser,
                                                    )
                                            elif (
                                                st.session_state.filter_type == "image"
                                            ):
                                                if st.session_state.uploaded_image:
                                                    apply_image_filter(
                                                        u,
                                                        index,
                                                        st.session_state.uploaded_image,
                                                        goal_parser,
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
                                img = find_anime_image(str(item["id"]))

                                # Clickable button for details
                                if st.button(
                                    "Click to view details",
                                    key=f"history_img_btn_{item['id']}_{idx}",
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
