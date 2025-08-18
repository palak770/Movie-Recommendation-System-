import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load datasets
# ----------------------------
movies_path = r"C:\Users\dpand\Dropbox\PC\Downloads\archive (1)\tmdb_5000_movies.csv"
credits_path = r"C:\Users\dpand\Dropbox\PC\Downloads\archive (1)\tmdb_5000_credits.csv"

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

# Strip spaces from column names
movies.columns = movies.columns.str.strip()
credits.columns = credits.columns.str.strip()

# Check columns
st.write("Columns in movies dataset:", movies.columns.tolist())

# ----------------------------
# Handle missing values safely
# ----------------------------
# Adjust column names if necessary
genre_col = 'genres' if 'genres' in movies.columns else 'Genre'
title_col = 'title' if 'title' in movies.columns else 'Title'

movies[genre_col] = movies[genre_col].fillna("")
movies[title_col] = movies[title_col].fillna("")

# ----------------------------
# Create tags (combine Genre + Title)
# ----------------------------
movies['tags'] = movies[genre_col].astype(str) + " " + movies[title_col].astype(str)

# ----------------------------
# Vectorize tags
# ----------------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies['tags']).toarray()

# ----------------------------
# Compute similarity
# ----------------------------
similarity = cosine_similarity(vectors)

# ----------------------------
# Recommendation function
# ----------------------------
def recommend(movie):
    if movie not in movies[title_col].values:
        return ["Movie not found in dataset!"]
    movie_index = movies[movies[title_col] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]][title_col] for i in movie_list]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", movies[title_col].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("### Recommended Movies:")
    for i in recommendations:
        st.write(i)
