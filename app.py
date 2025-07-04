import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit Config ---
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🍿", layout="centered")

# --- Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f4f4; }
    .title { text-align: center; font-size: 40px; font-weight: 900; color: #d92e2e; margin-top: 20px; }
    .subtitle { text-align: center; font-size: 18px; color: #555; margin-bottom: 30px; }
    .rec-box { background-color: white; padding: 15px 25px; border-radius: 12px;
               margin: 8px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.05); font-size: 17px; }
    .rec-box:hover { background-color: #fce4ec; transform: scale(1.01); }
    .footer { text-align: center; font-size: 14px; margin-top: 50px; color: #888; }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type a movie you love and discover similar ones instantly 🍿</div>', unsafe_allow_html=True)

# --- Load & Cache Data ---
@st.cache_data
def load_movies():
    df = pd.read_csv("movies.csv")
    df.fillna('', inplace=True)
    df['title'] = df['title'].str.strip()  # remove extra spaces
    return df

@st.cache_resource
def build_model(df):
    combined = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined)
    similarity = cosine_similarity(vectors)
    return similarity

# --- Recommendation Logic ---
def get_recommendations(user_input, df, similarity):
    titles = df['title'].tolist()
    titles_lower = [t.lower() for t in titles]
    user_input = user_input.strip().lower()

    matches = difflib.get_close_matches(user_input, titles_lower, n=1, cutoff=0.5)

    if not matches:
        return [], None

    matched_title = titles[titles_lower.index(matches[0])]
    idx = df[df.title == matched_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommendations = [df.iloc[i[0]]['title'] for i in sorted_scores[1:21]]
    return recommendations, matched_title

# --- Run App ---
movies_df = load_movies()
similarity_matrix = build_model(movies_df)

movie_input = st.text_input("🔍 Enter a movie title", "")

if st.button("🎯 Show Recommendations"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations, matched = get_recommendations(movie_input, movies_df, similarity_matrix)

        st.write("Debug:", {"You typed": movie_input, "Matched title": matched})  # REMOVE this later if you want

        if matched:
            st.success(f"Top 20 movies similar to **{matched}**:")
            for i, movie in enumerate(recommendations, 1):
                st.markdown(f'<div class="rec-box">🍿 {i}. {movie}</div>', unsafe_allow_html=True)
        else:
            st.error("No similar movie found. Try a different title.")

# --- Footer ---
st.markdown('<div class="footer">Built with ❤️ using Streamlit · Designed for movie lovers 🎥</div>', unsafe_allow_html=True)
