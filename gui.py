import streamlit as st
from indexing import *


# caching of models
@st.cache_resource
def load_models():

    syn_model = SynopsisEncoder().getModel()
    vis_model = VisualEncoder().getModel()

    # evaluation mode for DinoV2
    if hasattr(vis_model, 'eval'):
        vis_model.eval()

    return syn_model, vis_model


# load models
prompt_encoder_model, image_encoder_model = load_models()

# setting page
st.set_page_config(page_title="Anime Reccomendation System", layout="centered")

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
        height: 48px; /* Altezza simile all'input text */
    }
</style>
""", unsafe_allow_html=True)

# session status
if "results_images" not in st.session_state:
    st.session_state.results_images = []

# header
col_top_left, col_spacer, col_top_right = st.columns([3, 4, 2])

with col_top_left:
    user_name = st.text_input("Insert user name", placeholder="Es. Leonardo_Di_Caprio")

with col_top_right:
    top_k = st.number_input("Top-K", min_value=1, max_value=100, value=5, step=1)

st.markdown("---")

# results container
results_container = st.container()

# add images
with results_container:
    if st.session_state.results_images:
        st.subheader(f"Suggestions for: {user_name or 'Unknown User'}")

        # total results
        total_results = min(top_k, len(st.session_state.results_images))

        # imgs
        IMG_PER_ROW = 5

        # images in table layout
        # row
        for i in range(0, total_results, IMG_PER_ROW):

            cols = st.columns(IMG_PER_ROW)

            # columns
            for j in range(IMG_PER_ROW):
                idx = i + j

                if idx < total_results:
                    with cols[j]:
                        img_data = st.session_state.results_images[idx]
                        st.image(img_data, caption=f"Result {idx + 1}", use_container_width=True)

    else:
        st.info("Insert a prompt and/or an image and press suggest")
        for _ in range(5): st.write("")

st.markdown("---")

# input
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

# execution logic
if run_search:
    if not prompt_text and not uploaded_file:
        st.warning("Please insert a text prompt or upload an image.")
    else:
        with st.spinner("Analyzing and retrieving suggestions..."):

            # prompt encode
            prompt_encoded = None
            if prompt_text:
                prompt_encoded = prompt_encoder_model.encode(
                    prompt_text,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

            # image encode
            image_encoded = None
            if uploaded_file:

                # loading image
                pil_image = Image.open(uploaded_file).convert('RGB')

                # transfrom
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                # input to tensor and add batch size.
                input_tensor = transform(pil_image).unsqueeze(0)

                # CPU or GPU run
                device = next(image_encoder_model.parameters()).device
                input_tensor = input_tensor.to(device)

                # encoding
                with torch.no_grad():
                    image_encoded = image_encoder_model(input_tensor).cpu().numpy()

            db = Indexing()

            fused_embedding = prompt_encoded

            #if uploaded_file:
                #fused_embedding =

            #db.load_vector_db()
            #final_results = db.search_similar_anime(fused_embedding, top_k=top_k)

            # gemini simulation
            simulated_results = []
            for i in range(top_k):
                simulated_results.append(f"https://picsum.photos/200/200?random={np.random.randint(0, 1000)}")

            st.session_state.results_images = simulated_results

            st.rerun()