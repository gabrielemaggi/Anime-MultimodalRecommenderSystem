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
st.set_page_config(page_title="Anime Recommendation System", layout="centered")


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
    # Crea un dizionario indicizzato per ID per accesso istantaneo
    # Orient='index' crea un dict tipo: {123: {'title_english': '...', ...}, 456: {...}}
    return df.set_index('id').to_dict(orient='index') # Osa 'id' o 'MAL_ID' in base al tuo CSV


# load models
prompt_encoder_model, image_encoder_model, index = load_models()

# css for rounded buttons
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# session status - Utilizziamo "results" per memorizzare i dizionari completi degli anime
if "results" not in st.session_state:
    st.session_state.results = []


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


# ---------------------------------------------------------------------------
# Sidebar: Navigazione e Impostazioni Utente Globali
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Anime RecSys")

    # Navigazione tra le pagine
    page_selection = st.radio("Go to:", ["Recommendations", "User History"])

    st.markdown("---")
    st.subheader("User Settings")

    # Input Utente (Spostati qui per essere visibili in tutte le pagine)
    user_name = st.text_input("Insert user name", placeholder="Es. Leonardo_Di_Caprio", key="sidebar_user_name")
    top_k = st.number_input("Top-K", min_value=1, max_value=100, value=5, step=1)

    st.info(f"Current Mode: {page_selection}")

# ---------------------------------------------------------------------------
# PAGINA 1: RECOMMENDATIONS (Logica Originale)
# ---------------------------------------------------------------------------
if page_selection == "Recommendations":

    # results container
    results_container = st.container()

    # add images (Visualizzazione a griglia con card dettagliate)
    with results_container:
        if st.session_state.results:
            st.subheader(f"Suggestions for: {user_name or 'Unknown User'}")

            IMG_PER_ROW = 3
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
                                # print(type(img_path)) # Debug rimosso per pulizia UI
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

                                # --- Rating Input ---
                                with st.form(key=f"form_rec_{anime_id}"):
                                    rating = st.number_input(
                                        "Rate (1-10)",
                                        min_value=1,
                                        max_value=10,
                                        value=5,
                                        key=f"val_rec_{anime_id}",
                                    )
                                    submit_rating = st.form_submit_button("Submit Rating")

                                if submit_rating:
                                    if user_name:
                                        # 1. Initialize user and add the anime
                                        u = User(
                                            int(user_name)
                                            if user_name.isdigit()
                                            else user_name
                                        )
                                        u.add_anime(
                                            anime_id, rating
                                        )  # Assuming add_anime(id, score)

                                        # 2. Recalculate Clusters and Suggestions
                                        with st.spinner("Updating preferences..."):
                                            u.findCentersOfClusters()
                                            new_results = u.get_nearest_anime_from_clusters(
                                                index, top_k
                                            )
                                            st.session_state.results = new_results

                                        st.success(f"Added {title} to your list!")
                                        st.rerun()
                                    else:
                                        st.error("Please enter a user name first.")
        else:
            st.info("Insert a prompt and/or an image and press suggest")
            for _ in range(5):
                st.write("")

    st.markdown("---")

    # input area (bottom)
    with st.container(border=True):
        col_clip, col_input, col_btn = st.columns([0.05, 0.85, 0.1], gap="small")

        with col_clip:
            with st.popover("📎", help="Load an image"):
                uploaded_file = st.file_uploader(
                    "Upload image", type=["png", "jpg", "jpeg"]
                )
                if uploaded_file:
                    st.success("Loaded!")

        with col_input:
            prompt_text = st.text_input(
                "Insert a prompt...", label_visibility="collapsed", key="search_input"
            )

        with col_btn:
            run_search = st.button("➤", type="primary", help="Run Suggestion")

    # execution logic for Recommendation Page
    if run_search:
        if not user_name or user_name.strip() == "":
            st.warning("User name not specified, suggestions not based on user preferences")
        else:
            # Codice per inizializzazione utente e cluster
            user = User(int(user_name) if user_name.isdigit() else user_name)
            # st.session_state.user_watchlist = user.get_watchList() # Mantenuto commentato come richiesto

            with st.spinner("Calculating preferences..."):
                user.findCentersOfClusters()
                # Otteniamo i risultati completi (dizionari)
                results = user.get_nearest_anime_from_clusters(index, top_k)
                st.session_state.results = results

        # Logica per prompt/immagine (Mantenuta commentata come richiesto)
        # if not prompt_text and not uploaded_file:
        #     st.warning("Please insert a text prompt or upload an image.")
        # else:
        #     with st.spinner("Analyzing and retrieving suggestions..."):
        #         ... (logica encoder) ...

        st.rerun()

    # --- Commented Block for Encoders ---
    # if not prompt_text and not uploaded_file:
    #    st.warning("Please insert a text prompt or upload an image.")
    # else:
    #    with st.spinner("Analyzing and retrieving suggestions..."):

    # prompt encode
    #        prompt_encoded = None
    #        if prompt_text:
    #            prompt_encoded = prompt_encoder_model.encode(
    #                prompt_text,
    #                show_progress_bar=False,
    #                convert_to_numpy=True
    #            )

    # image encode
    #        image_encoded = None
    #        if uploaded_file:

    # loading image
    #            pil_image = Image.open(uploaded_file).convert('RGB')

    # transfrom
    #            transform = T.Compose([
    #                T.Resize((224, 224)),
    #                T.ToTensor(),
    #                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #            ])

    # input to tensor and add batch size.
    #            input_tensor = transform(pil_image).unsqueeze(0)

    # CPU or GPU run
    #            device = next(image_encoder_model.parameters()).device
    #            input_tensor = input_tensor.to(device)

    # encoding
    #            with torch.no_grad():
    #                image_encoded = image_encoder_model(input_tensor).cpu().numpy()

    # if uploaded_file:
    # fused_embedding =

    # db.load_vector_db()
    # final_results = db.search_similar_anime(fused_embedding, top_k=top_k)

    # gemini simulation
    # simulated_results = []
    # for i in range(top_k):
    #    simulated_results.append(f"https://picsum.photos/200/200?random={np.random.randint(0, 1000)}")

    # st.session_state.results_images = simulated_results

    # st.rerun()


# ---------------------------------------------------------------------------
# PAGINA 2: USER HISTORY
# ---------------------------------------------------------------------------
elif page_selection == "User History":
        # ... (codice header utente) ...

        if user_name:
            try:
                user = User(int(user_name) if user_name.isdigit() else user_name)
                raw_watchlist = user.get_watchList()

                if not raw_watchlist:
                    st.info("No history.")
                else:
                    # 1. Carica il dizionario veloce (lo fa una volta sola e lo mette in cache)
                    anime_lookup = get_anime_lookup()

                    clean_history = []

                    # 2. Ciclo ora velocissimo
                    for item in raw_watchlist:
                        anime_id = item[0]
                        score = item[1]

                        # Lookup istantaneo nel dizionario (niente scansione righe!)
                        anime_info = anime_lookup.get(anime_id)

                        if anime_info:
                            # Gestione Titolo (Priorità: Inglese -> Giapponese -> Sconosciuto)
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

                # 4. Visualizzazione a Griglia
                IMG_PER_ROW = 3
                total_items = len(clean_history)

                for i in range(0, total_items, IMG_PER_ROW):
                    cols = st.columns(IMG_PER_ROW)
                    for j in range(IMG_PER_ROW):
                        idx = i + j
                        if idx < total_items:
                            anime_data = clean_history[idx]

                            # Estrazione dati per la card
                            current_id = str(anime_data['id'])
                            current_title = anime_data['title']
                            current_score = anime_data['rating']

                            with cols[j]:
                                with st.container(border=True):
                                    # Trova immagine usando l'ID
                                    img_path = find_anime_image(current_id)

                                    if img_path:
                                        st.image(img_path, use_container_width=True)
                                    else:
                                        st.image(
                                            "https://via.placeholder.com/200x300?text=No+Image",
                                            use_container_width=True,
                                        )
                                        st.markdown(f"no image found")

                                    # Mostra Titolo recuperato dal CSV
                                    st.markdown(f"**{current_title}**")

                                    # Mostra Score recuperato dalla Watchlist
                                    st.write(f"🌟 Your Rating: **{current_score}**")

            except Exception as e:
                st.error(f"Error loading history: {e}")
                st.caption("Debug Info: Check CSV path and column names (anime_id vs id).")