import os
from math import nan
from pathlib import Path

import pandas as pd
import streamlit as st
from indexing_db import *
from User import User

# ---------------------------------------------------------------------------
# Configurazione Pagina (Deve essere la prima istruzione Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Anime Recommendation System", layout="wide")


# caching of models
@st.cache_resource
def load_models():
    syn_model = SynopsisEncoder().getModel()
    vis_model = VisualEncoder().getModel()

    index = Indexing()
    index.load_vector_database()

    # evaluation mode for DinoV2
    if hasattr(vis_model, "eval"):
        vis_model.eval()

    return syn_model, vis_model, index


@st.cache_data
def get_anime_lookup():
    """
    Crea un dizionario veloce per cercare i titoli.
    Chiave: ID Anime -> Valore: {Dati Anime}
    """
    df = pd.read_csv("dataset/AnimeList.csv")
    return df.set_index('id').to_dict(orient='index')


@st.cache_data
def get_available_genres_studios():
    """
    Estrae tutti i generi e gli studi disponibili dal dataset
    """
    df = pd.read_csv("./dataset/AnimeList.csv")

    # Estrai generi unici (assumendo che siano in una colonna separata da virgole)
    all_genres = set()
    if 'genre' in df.columns:
        for genres_str in df['genre'].dropna():
            if isinstance(genres_str, str):
                all_genres.update([g.strip() for g in genres_str.split(',')])

    # Estrai studi unici
    all_studios = set()
    if 'studio' in df.columns:
        for studio_str in df['studio'].dropna():
            if isinstance(studio_str, str):
                all_studios.update([s.strip() for s in studio_str.split(',')])

    return sorted(list(all_genres)), sorted(list(all_studios))


# load models
prompt_encoder_model, image_encoder_model, index = load_models()

# css for rounded buttons and responsive layout
st.markdown(
    """
<style>
    div[data-testid="column"] {
        display: flex;
        align-items: stretch;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
    }
    /* Responsive container for cards */
    .anime-card {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    /* Ensure images maintain aspect ratio */
    .stImage {
        flex-shrink: 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# session status
if "results" not in st.session_state:
    st.session_state.results = []
if "filter_mode" not in st.session_state:
    st.session_state.filter_mode = "append"  # or "move"
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []
if "selected_studios" not in st.session_state:
    st.session_state.selected_studios = []
if "cards_per_row" not in st.session_state:
    st.session_state.cards_per_row = 4
if "filter_magnitude" not in st.session_state:
    st.session_state.filter_magnitude = 1.0


# ---------------------------------------------------------------------------
# Helper: trova il file immagine (gestisce cartelle o file diretti)
# ---------------------------------------------------------------------------
def find_anime_image(anime_id: str) -> str | None:
    EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    # Caso 1: file diretto
    for ext in EXTENSIONS:
        candidate = Path(f"dataset/images/{anime_id}{ext}")
        if candidate.is_file():
            return str(candidate)
    # Caso 2: cartella con ID
    folder = Path(f"dataset/images/{anime_id}")
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in EXTENSIONS:
                return str(f)
    return None


def apply_filtering(user_obj, indexer, genres=None, studios=None, mode="append", magnitude=1.0):
    """
    Applica il filtro basato su generi e studi con una magnitudine specificata

    Args:
        user_obj: User object
        indexer: Indexing object
        genres: List of genre strings
        studios: List of studio strings
        mode: "append" or "move"
        magnitude: Float value controlling filter strength (0.0 to n)
    """
    if not genres and not studios:
        return  # Nessun filtro da applicare

    try:
        # Codifica i metadati tabellari
        results = indexer.encode_tabular_genre_studio(
            genres=genres if genres else [],
            studios=studios if studios else []
        )

        # Estrai l'embedding (priorità a generi, poi studi)
        embedding = None
        if genres and results.get('genres'):
            # Usa il primo genere selezionato
            embedding = results.get('genres').get(genres[0])
        elif studios and results.get('studios'):
            # Usa il primo studio selezionato
            embedding = results.get('studios').get(studios[0])

        if embedding is not None:
            # Allinea l'embedding
            query = indexer.align_embedding(embedding, modality='tab')
            # Applica il filtro all'utente con magnitudine
            user_obj.add_filtering(query, mode, magnitude)
            return True
    except Exception as e:
        st.error(f"Error applying filter: {e}")
        return False

    return False


# ---------------------------------------------------------------------------
# Sidebar: Navigazione e Impostazioni Utente Globali
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 Anime RecSys")

    # Navigazione tra le pagine
    page_selection = st.radio("Navigate to:", ["Recommendations", "User History"], label_visibility="visible")

    st.markdown("---")
    st.subheader("⚙️ User Settings")

    # Input Utente
    user_name = st.text_input("User Name", placeholder="e.g., MrPeanut02", key="sidebar_user_name")
    top_k = st.number_input("Top-K Results", min_value=1, max_value=100, value=10, step=1)

    # Responsive layout setting
    st.markdown("---")
    st.subheader("📐 Display Settings")
    cards_per_row = st.slider("Cards per Row", min_value=2, max_value=6, value=4, step=1)
    st.session_state.cards_per_row = cards_per_row

    # Filtering Section
    st.markdown("---")
    st.subheader("🔍 Content Filters")

    # Get available genres and studios
    available_genres, available_studios = get_available_genres_studios()

    # Genre selection
    selected_genres = st.multiselect(
        "Filter by Genres",
        options=available_genres,
        default=st.session_state.selected_genres,
        help="Select one or more genres to filter recommendations"
    )
    st.session_state.selected_genres = selected_genres

    # Studio selection
    selected_studios = st.multiselect(
        "Filter by Studios",
        options=available_studios,
        default=st.session_state.selected_studios,
        help="Select one or more studios to filter recommendations"
    )
    st.session_state.selected_studios = selected_studios

    # Filter mode
    filter_mode = st.radio(
        "Filter Mode",
        options=["append", "move"],
        index=0 if st.session_state.filter_mode == "append" else 1,
        help="Append: Add filter preferences | Move: Shift cluster centers"
    )
    st.session_state.filter_mode = filter_mode

    # Filter magnitude slider
    filter_magnitude = st.slider(
        "Filter Magnitude",
        min_value=0.0,
        max_value=5.0,
        value=st.session_state.filter_magnitude,
        step=0.1,
        help="Controls the strength of the filter. Higher values = stronger influence"
    )
    st.session_state.filter_magnitude = filter_magnitude

    # Visual indicator for magnitude
    if filter_magnitude > 0:
        magnitude_bars = "█" * int(filter_magnitude * 2)
        st.caption(f"Strength: {magnitude_bars} ({filter_magnitude:.1f})")

    # Apply filter button
    if st.button("🎯 Apply Filters", type="primary", use_container_width=True):
        if user_name:
            with st.spinner("Applying filters..."):
                user = User(int(user_name) if user_name.isdigit() else user_name)
                user.findCentersOfClusters()

                success = apply_filtering(
                    user,
                    index,
                    genres=selected_genres if selected_genres else None,
                    studios=selected_studios if selected_studios else None,
                    mode=filter_mode,
                    magnitude=filter_magnitude
                )

                if success:
                    results = user.get_nearest_anime_from_clusters(index, top_k)
                    st.session_state.results = results
                    st.success("✅ Filters applied successfully!")
                    st.rerun()
        else:
            st.warning("⚠️ Please enter a user name first")

    # Clear filters button
    if st.button("🗑️ Clear Filters", use_container_width=True):
        st.session_state.selected_genres = []
        st.session_state.selected_studios = []
        st.session_state.filter_magnitude = 1.0
        st.rerun()

    st.markdown("---")
    st.caption(f"📍 Current Page: {page_selection}")


# ---------------------------------------------------------------------------
# PAGINA 1: RECOMMENDATIONS
# ---------------------------------------------------------------------------
if page_selection == "Recommendations":
    st.title("🌟 Anime Recommendations")

    # Show active filters
    if st.session_state.selected_genres or st.session_state.selected_studios:
        filter_info = []
        if st.session_state.selected_genres:
            filter_info.append(f"Genres: {', '.join(st.session_state.selected_genres)}")
        if st.session_state.selected_studios:
            filter_info.append(f"Studios: {', '.join(st.session_state.selected_studios)}")

        magnitude_display = f"Magnitude: {st.session_state.filter_magnitude:.1f}"
        st.info(f"🔍 Active Filters ({st.session_state.filter_mode}) | {' | '.join(filter_info)} | {magnitude_display}")

    # results container
    results_container = st.container()

    # Display results with responsive grid
    with results_container:
        if st.session_state.results:
            st.subheader(f"✨ Suggestions for: {user_name or 'Unknown User'}")

            IMG_PER_ROW = st.session_state.cards_per_row
            total_results = len(st.session_state.results)

            for i in range(0, total_results, IMG_PER_ROW):
                cols = st.columns(IMG_PER_ROW)
                for j in range(IMG_PER_ROW):
                    idx = i + j
                    if idx < total_results:
                        anime = st.session_state.results[idx]
                        anime_id = str(anime.get("id", ""))
                        with cols[j]:
                            with st.container(border=True):
                                img_path = find_anime_image(anime_id)
                                if img_path:
                                    st.image(img_path, use_container_width=True)
                                else:
                                    st.image(
                                        "https://via.placeholder.com/200x300?text=No+Image",
                                        use_container_width=True,
                                    )

                                title = anime.get("title", "Unknown Title")
                                st.markdown(f"**{title}**")

                                similarity = anime.get("similarity")
                                if similarity:
                                    st.caption(
                                        f"🎯 Similarity: {float(similarity) * 100:.1f}%"
                                    )

                                # Rating Input
                                with st.form(key=f"form_rec_{anime_id}_{idx}"):
                                    rating = st.number_input(
                                        "Rate (1-10)",
                                        min_value=1,
                                        max_value=10,
                                        value=5,
                                        key=f"val_rec_{anime_id}_{idx}",
                                    )
                                    submit_rating = st.form_submit_button("Submit Rating")

                                if submit_rating:
                                    if user_name:
                                        u = User(
                                            int(user_name)
                                            if user_name.isdigit()
                                            else user_name
                                        )
                                        u.add_anime(anime_id, rating)

                                        with st.spinner("Updating preferences..."):
                                            u.findCentersOfClusters()

                                            # Re-apply filters if active
                                            if st.session_state.selected_genres or st.session_state.selected_studios:
                                                apply_filtering(
                                                    u,
                                                    index,
                                                    genres=st.session_state.selected_genres if st.session_state.selected_genres else None,
                                                    studios=st.session_state.selected_studios if st.session_state.selected_studios else None,
                                                    mode=st.session_state.filter_mode,
                                                    magnitude=st.session_state.filter_magnitude
                                                )

                                            new_results = u.get_nearest_anime_from_clusters(
                                                index, top_k
                                            )
                                            st.session_state.results = new_results

                                        st.success(f"✅ Added {title} to your list!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Please enter a user name first.")
        else:
            st.info("👋 Welcome! Enter your user name and click 'Get Recommendations' to start.")

    st.markdown("---")

    # Input area (bottom)
    with st.container(border=True):
        col_clip, col_input, col_btn = st.columns([0.05, 0.85, 0.1], gap="small")

        with col_clip:
            with st.popover("📎", help="Load an image"):
                uploaded_file = st.file_uploader(
                    "Upload image", type=["png", "jpg", "jpeg"]
                )
                if uploaded_file:
                    st.success("✅ Image loaded!")

        with col_input:
            prompt_text = st.text_input(
                "Insert a prompt...", label_visibility="collapsed", key="search_input"
            )

        with col_btn:
            run_search = st.button("🚀 Get Recommendations", type="primary", help="Get personalized suggestions")

    # Execution logic for Recommendation Page
    if run_search:
        if not user_name or user_name.strip() == "":
            st.warning("⚠️ User name not specified. Please enter your name in the sidebar.")
        else:
            user = User(int(user_name) if user_name.isdigit() else user_name)

            with st.spinner("🔄 Calculating preferences..."):
                user.findCentersOfClusters()

                # Apply filters if any are selected
                if st.session_state.selected_genres or st.session_state.selected_studios:
                    apply_filtering(
                        user,
                        index,
                        genres=st.session_state.selected_genres if st.session_state.selected_genres else None,
                        studios=st.session_state.selected_studios if st.session_state.selected_studios else None,
                        mode=st.session_state.filter_mode,
                        magnitude=st.session_state.filter_magnitude
                    )

                results = user.get_nearest_anime_from_clusters(index, top_k)
                st.session_state.results = results

            st.rerun()


# ---------------------------------------------------------------------------
# PAGINA 2: USER HISTORY
# ---------------------------------------------------------------------------
elif page_selection == "User History":
    st.title("📚 Your Watch History")

    if user_name:
        try:
            user = User(int(user_name) if user_name.isdigit() else user_name)
            raw_watchlist = user.get_watchList()

            if not raw_watchlist:
                st.info("📭 No watch history yet. Start rating some anime!")
            else:
                anime_lookup = get_anime_lookup()
                clean_history = []

                for item in raw_watchlist:
                    anime_id = item[0]
                    score = item[1]

                    anime_info = anime_lookup.get(anime_id)

                    if anime_info:
                        title = anime_info.get('title_english')
                        if pd.isna(title):
                            title = anime_info.get('title_japanese')
                        if pd.isna(title):
                            title = f"Unknown Title (ID: {anime_id})"
                    else:
                        title = f"Unknown ID: {anime_id}"

                    clean_history.append({
                        'id': anime_id,
                        'title': title,
                        'rating': score
                    })

                # Display stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Watched", len(clean_history))
                with col2:
                    avg_rating = sum([item['rating'] for item in clean_history]) / len(clean_history)
                    st.metric("⭐ Average Rating", f"{avg_rating:.1f}")
                with col3:
                    top_rated = max([item['rating'] for item in clean_history])
                    st.metric("🏆 Highest Rating", top_rated)

                st.markdown("---")

                # Responsive grid display
                IMG_PER_ROW = st.session_state.cards_per_row
                total_items = len(clean_history)

                for i in range(0, total_items, IMG_PER_ROW):
                    cols = st.columns(IMG_PER_ROW)
                    for j in range(IMG_PER_ROW):
                        idx = i + j
                        if idx < total_items:
                            anime_data = clean_history[idx]

                            current_id = str(anime_data['id'])
                            current_title = anime_data['title']
                            current_score = anime_data['rating']

                            with cols[j]:
                                with st.container(border=True):
                                    img_path = find_anime_image(current_id)

                                    if img_path:
                                        st.image(img_path, use_container_width=True)
                                    else:
                                        st.image(
                                            "https://via.placeholder.com/200x300?text=No+Image",
                                            use_container_width=True,
                                        )

                                    st.markdown(f"**{current_title}**")

                                    # Star rating display
                                    star_display = "⭐" * int(current_score)
                                    st.write(f"{star_display} **{current_score}/10**")

        except Exception as e:
            st.error(f"❌ Error loading history: {e}")
            st.caption("💡 Debug Info: Check CSV path and column names.")
    else:
        st.warning("⚠️ Please enter your user name in the sidebar to view your history.")
