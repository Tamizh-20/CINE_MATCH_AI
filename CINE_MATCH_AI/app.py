import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
movies = pd.read_csv("movies.csv")

# TF-IDF processing
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['Genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Title lookup
movie_indices = pd.Series(movies.index, index=movies['Title'])

# Recommend function
def get_recommendations(title, top_n=10):
    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices_top = [i[0] for i in sim_scores]
    return movies['Title'].iloc[movie_indices_top]

# --- Streamlit UI ---
st.set_page_config(page_title="🎬 MovieMatch AI", layout="centered")

st.title("🎬 MovieMatch AI")
st.markdown("""
Welcome to **MovieMatch AI** — your personal movie recommender system powered by artificial intelligence!  
Enter a movie title you like, and we'll suggest similar ones just for you.  
""")

# Movie input (instead of selectbox)
selected_movie = st.text_input("Enter the name of a movie you like:")

if st.button("🎥 Recommend"):
    if selected_movie:
        st.subheader("You might also enjoy:")
        try:
            recommendations = get_recommendations(selected_movie)
            for i, title in enumerate(recommendations, 1):
                st.write(f"{i}. {title}")
        except KeyError:
            st.error("Sorry, we couldn't find the movie. Please try again with a different title.")
    else:
        st.error("Please enter a movie name.")
