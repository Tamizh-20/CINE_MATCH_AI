from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset/movies.csv")
df['overview'] = df['overview'].fillna('')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])

# Recommend function
def recommend(title):
    title = title.lower()
    df['title_lower'] = df['title'].str.lower()

    if title not in df['title_lower'].values:
        return []

    idx = df[df['title_lower'] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10 excluding the input movie
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices].tolist()

# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    recs = []
    if request.method == "POST":
        movie = request.form["movie"]
        print("User input:", movie)
        recs = recommend(movie)
        print("Recommendations:", recs)
    return render_template("index.html", recommendations=recs)
# Run the app
if __name__ == "__main__":
    app.run(debug=True)