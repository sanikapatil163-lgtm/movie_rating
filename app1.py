
import streamlit as st
import numpy as np
import pickle
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="CineRate | Movie Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10091f, #211036, #10091f);
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    text-align: center;
    padding: 30px 10px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #d8b4fe;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #c4b5fd;
    font-size: 1.1rem;
}

.section-title {
    color: #d8b4fe;
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 8px;
}

div[data-testid="stForm"] {
    background: rgba(43, 23, 72, 0.65);
    border: 1px solid #553477;
    border-radius: 18px;
    padding: 25px;
}

.stTextInput label,
.stTextArea label,
.stMultiSelect label,
.stFileUploader label {
    color: #e9d5ff !important;
}

.stTextInput input,
.stTextArea textarea {
    background-color: #1b102b !important;
    color: white !important;
    border: 1px solid #67458b !important;
    border-radius: 10px !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(43, 23, 72, 0.5);
    border: 1px dashed #8b5fbf;
    border-radius: 12px;
    padding: 12px;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #9333ea, #c084fc);
    color: white;
    border: none;
}

.result-card {
    background: linear-gradient(135deg, #2b1748, #412064);
    border: 1px solid #9563c7;
    border-radius: 18px;
    padding: 35px 20px;
    text-align: center;
    margin-top: 10px;
}

.result-card h2 {
    color: #e9d5ff;
    font-size: 1.5rem;
}

.result-card h1 {
    color: #f0abfc;
    font-size: 3.5rem;
    margin: 15px 0;
}

.result-card p {
    color: #d8b4fe;
}

.poster-placeholder {
    background: linear-gradient(145deg, #2b1748, #171027);
    border: 1px solid #604080;
    border-radius: 16px;
    padding: 55px 15px;
    text-align: center;
    color: #cbb7e8;
}

.footer {
    text-align: center;
    color: #a78bca;
    margin-top: 45px;
    font-size: 0.9rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    with open("movie_rating_model.pkl", "rb") as file:
        xgb_model = pickle.load(file)

    with open("sentence_model.pkl", "rb") as file:
        sentence_model = pickle.load(file)

    with open("genre_columns.pkl", "rb") as file:
        genres_columns = pickle.load(file)

    with open("avg_actor_rating.pkl", "rb") as file:
        avg_actor_rating = pickle.load(file)

    with open("avg_director_rating.pkl", "rb") as file:
        avg_director_rating = pickle.load(file)

    return (
        xgb_model,
        sentence_model,
        genres_columns,
        avg_actor_rating,
        avg_director_rating
    )


try:
    (
        xgb_model,
        sentence_model,
        genres_columns,
        avg_actor_rating,
        avg_director_rating
    ) = load_models()

except Exception as e:
    st.error(f"Unable to load the prediction models: {e}")
    st.info(
        "Make sure all five .pkl files are present "
        "in the same folder as app.py."
    )
    st.stop()


# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 CineRate</h1>
    <p>Discover your movie's predicted rating with AI</p>
</div>
""", unsafe_allow_html=True)


# ---------------- MOVIE POSTER UPLOAD ----------------
st.markdown(
    '<div class="section-title">🎞️ Movie Poster</div>',
    unsafe_allow_html=True
)

st.caption("Upload a poster to personalize your movie prediction.")

uploaded_poster = st.file_uploader(
    "Choose a movie poster",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG, JPG, or JPEG image."
)

poster_image = None

if uploaded_poster is not None:
    try:
        poster_image = Image.open(uploaded_poster).convert("RGB")
        st.image(
            poster_image,
            caption="Your Movie Poster",
            width=220
        )
    except Exception:
        st.error("The uploaded file could not be opened as an image.")


# ---------------- MOVIE DETAILS FORM ----------------
st.markdown(
    '<div class="section-title">✨ Movie Details</div>',
    unsafe_allow_html=True
)

with st.form("movie_prediction_form"):

    actors = st.text_input(
        "Actors",
        placeholder="Example: Actor One, Actor Two"
    )

    director = st.text_input(
        "Director",
        placeholder="Enter the director's name"
    )

    genres = st.text_input(
        "Genres",
        placeholder="Example: Drama, Thriller, Action"
    )

    synopsis = st.text_area(
        "Movie Synopsis",
        placeholder="Describe the movie story...",
        height=150
    )

    submitted = st.form_submit_button("✨ Predict Movie Rating")


# ---------------- PREDICTION LOGIC ----------------
if submitted:

    if not actors.strip() or not director.strip() or not genres.strip() or not synopsis.strip():
        st.warning("Please fill in all movie details before predicting.")

    else:
        try:
            # Calculate average rating for the entered actors
            input_actor_avg = np.mean([
                avg_actor_rating.get(actor.strip(), np.nan)
                for actor in actors.split(",")
            ])

            # Calculate average rating for the entered director
            input_director_avg = np.mean([
                avg_director_rating.get(director.strip(), np.nan)
                for director in director.split(",")
            ])

            # Calculate fallback rating for unknown names
            overall_mean_rating = np.mean(
                list(avg_actor_rating.values()) +
                list(avg_director_rating.values())
            )

            if np.isnan(input_actor_avg):
                input_actor_avg = overall_mean_rating

            if np.isnan(input_director_avg):
                input_director_avg = overall_mean_rating

            # Convert the synopsis into an embedding
            synopsis_embedding = sentence_model.encode([synopsis])

            # Create the genre feature vector
            input_genres = [
                genre.strip()
                for genre in genres.split(",")
            ]

            genres_vector = np.zeros(
                (1, len(genres_columns))
            )

            for genre in input_genres:
                if genre in genres_columns:
                    genres_vector[
                        0, genres_columns.index(genre)
                    ] = 1

            # Combine all features for the model
            features = np.hstack((
                synopsis_embedding,
                genres_vector,
                [[input_actor_avg, input_director_avg]]
            ))

            # Predict the movie rating
            predicted_rating = float(
                xgb_model.predict(features)[0]
            )

            # Keep the displayed score within the 0–10 range
            displayed_rating = max(
                0.0, min(10.0, predicted_rating)
            )

            st.markdown("---")
            st.markdown(
                '<div class="section-title">🍿 Prediction Result</div>',
                unsafe_allow_html=True
            )

            # Show poster and rating side by side
            col1, col2 = st.columns([1, 2], gap="large")

            with col1:
                if poster_image is not None:
                    st.image(
                        poster_image,
                        caption="Movie Poster",
                        use_container_width=True
                    )
                else:
                    st.markdown("""
                    <div class="poster-placeholder">
                        <div style="font-size: 55px;">🎬</div>
                        <p>Upload a poster to display it here</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <h2>✨ Predicted Movie Rating</h2>
                    <h1>{displayed_rating:.1f} / 10</h1>
                    <p>Your movie's predicted rating</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(displayed_rating / 10)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    🎬 CineRate — Explore stories. Predict ratings.
</div>
""", unsafe_allow_html=True)