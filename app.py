import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit Config ---
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🍿", layout="centered")

# --- Improved Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f4f4f4;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 900;
        color: #d92e2e;
        margin-top: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .rec-box {
        background-color: white;
        padding: 15px 25px;
        border-radius: 12px;
        margin: 8px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: 0.3s ease;
        font-size: 17px;
    }
    .rec-box:hover {
        background-color: #fce4ec;
        transform: scale(1.01);
    }
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type a movie you love and discover similar ones instantly 🍿</div>', unsafe_allow_html=True)

# --- Data Loader ---
@st.cache_data
def load_movies():
    df = pd.read_csv("movies.csv")
    df.fillna('', inplace=True)
    return df

@st.cache_resource
def build_model(df):
    combined = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined)
    similarity = cosine_similarity(vectors)
    return similarity

# --- Recommendation Logic ---
def get_recommendations(movie_name, df, similarity):
    titles = df['title'].tolist()
    match = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.5)
    if not match:
        return [], None
    index = df[df.title == match[0]].index[0]
    similarity_scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended = [df.iloc[i[0]]['title'] for i in sorted_scores[1:21]]
    return recommended, match[0]

# --- Load Data and Model ---
movies_df = load_movies()
similarity_matrix = build_model(movies_df)

# --- Input Field ---
movie_input = st.text_input("🔍 Enter a movie title", "")

# --- Recommend Button ---
if st.button("🎯 Show Recommendations"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations, matched = get_recommendations(movie_input, movies_df, similarity_matrix)
        if matched:
            st.success(f"Top 20 movies similar to **{matched}**:")
            for i, movie in enumerate(recommendations, 1):
                st.markdown(f'<div class="rec-box">🍿 {i}. {movie}</div>', unsafe_allow_html=True)
        else:
            st.error("No similar movie found. Try a different title.")

# --- Footer ---
st.markdown('<div class="footer">Built with ❤️ using Streamlit · Designed for movie lovers 🎥</div>', unsafe_allow_html=True)
