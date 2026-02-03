import os
import streamlit as st
from pathlib import Path
from indexing_db import *
from User import User


# caching of models
@st.cache_resource
def load_models():

    syn_model = SynopsisEncoder().getModel()
    vis_model = VisualEncoder().getModel()

    index = Indexing()
    index.load_vector_database()

    # evaluation mode for DinoV2
    if hasattr(vis_model, 'eval'):
        vis_model.eval()

    return syn_model, vis_model, index


# load models
prompt_encoder_model, image_encoder_model, index = load_models()

# setting page
st.set_page_config(page_title="Anime Reccomendation System", layout="wide")

# css for rounded buttons
st.markdown("""
<style>
    div[data-testid="column"] {
        display: flex;
        align-items: center;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
    }
</style>
""", unsafe_allow_html=True)

# session status
if "results" not in st.session_state:
    st.session_state.results = []


# ---------------------------------------------------------------------------
# Helper: find the image file inside Imgs/{id}  (folder or direct file)
# ---------------------------------------------------------------------------
def find_anime_image(anime_id: str) -> str | None:
    """
    Looks for an image in two places:
      1. Imgs/<id>.<ext>          (file directly)
      2. Imgs/<id>/<anything>.<ext>  (folder containing an image)
    Returns the path string if found, else None.
    """
    EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")

    # --- case 1: Imgs/123.jpg ---
    for ext in EXTENSIONS:
        candidate = Path(f"dataset/images/{anime_id}{ext}")
        if candidate.is_file():
            return str(candidate)

    # --- case 2: Imgs/123/poster.jpg ---
    folder = Path(f"dataset/images/{anime_id}")
    if folder.is_dir():
        for f in sorted(folder.iterdir()):          # sorted for determinism
            if f.is_file() and f.suffix.lower() in EXTENSIONS:
                return str(f)

    return None


# ---------------------------------------------------------------------------
# Render a single anime card
# ---------------------------------------------------------------------------
def render_card(anime: dict):
    anime_id   = str(anime.get("id", "unknown"))
    title      = anime.get("title", "Unknown Title")
    title_en   = anime.get("title_english", None)
    synopsis   = anime.get("sypnopsis", "No synopsis available.")
    similarity = anime.get("similarity", None)

    with st.container(border=True):

        # ── image ─────────────────────────────────────────────
        img_path = find_anime_image(anime_id)

        st.image(img_path, use_container_width=True)


        # ── title ─────────────────────────────────────────────
        # only append English title when it exists and is not NaN
        if title_en and str(title_en).lower() != "nan":
            display_title = f"{title} — {title_en}"
        else:
            display_title = title
        st.markdown(f"**{display_title}**")

        # ── similarity badge ──────────────────────────────────
        if similarity is not None:
            pct = float(similarity) * 100
            st.markdown(f"🎯 *Similarity: {pct:.1f}%*")

        # ── synopsis (truncated, with expander for full text) ─
        if synopsis and len(synopsis) > 150:
            st.caption(synopsis[:150] + "…")
            with st.expander("Read more", expanded=False):
                st.write(synopsis)
        else:
            st.caption(synopsis)


# ---------------------------------------------------------------------------
# PAGE LAYOUT
# ---------------------------------------------------------------------------

# header
col_top_left, col_spacer, col_top_right = st.columns([3, 4, 2])

with col_top_left:
    user_name = st.text_input("Insert user name", placeholder="Es. Leonardo_Di_Caprio")

with col_top_right:
    top_k = st.number_input("Top-K", min_value=1, max_value=100, value=5, step=1)

st.markdown("---")

# ---------------------------------------------------------------------------
# Results grid  (3 cards per row)
# ---------------------------------------------------------------------------
results_container = st.container()

with results_container:
    if st.session_state.results:
        st.subheader(f"Suggestions for: {user_name or 'Unknown User'}")

        CARDS_PER_ROW = 3
        total = len(st.session_state.results)

        for i in range(0, total, CARDS_PER_ROW):
            cols = st.columns(CARDS_PER_ROW)
            for j in range(CARDS_PER_ROW):
                idx = i + j
                if idx < total:
                    with cols[j]:
                        render_card(st.session_state.results[idx])
    else:
        st.info("Insert a prompt and/or an image and press suggest")
        for _ in range(5):
            st.write("")

st.markdown("---")

# ---------------------------------------------------------------------------
# Input bar
# ---------------------------------------------------------------------------
with st.container(border=True):
    col_clip, col_input, col_btn = st.columns([0.05, 0.85, 0.1], gap="small")

    with col_clip:
        with st.popover("📎", help="Load an image"):
            uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                st.success("Loaded!")

    with col_input:
        prompt_text = st.text_input("Insert a prompt...", label_visibility="collapsed", key="search_input")

    with col_btn:
        run_search = st.button("➤", type="primary", help="Run Suggestion")

# ---------------------------------------------------------------------------
# Execution logic
# ---------------------------------------------------------------------------
if run_search:

    if not user_name or user_name.strip() == "":
        st.warning("User name not specified, suggestions not based on user preferences")
    else:

        user = User(int(user_name))
        print(int(user_name))
        st.session_state.current_user_name = user_name

        user.findCentersOfClusters()

        # get_nearest_anime_from_clusters returns list of anime dicts
        results = user.get_nearest_anime_from_clusters(index, top_k)
        print("results: ", results)
        # store the raw anime dicts (not image URLs)
        st.session_state.results = results

        st.rerun()
