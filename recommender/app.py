'''Streamlit app for Movie Lens hybrid filtering recommendations'''

import sys
from pathlib import Path

import pickle
import subprocess
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data/matricies.pkl'
GENERATOR_PATH = Path(__file__).with_name('generate_data.py')

@st.cache_resource
def get_data():

    if Path(DATA_PATH).exists():
        with open(DATA_PATH, 'rb') as file:
            data = pickle.load(file)

        if 'movies' in data:
            return data

    with st.spinner("Generating data artifacts...", show_time=True):
        subprocess.run([sys.executable, str(GENERATOR_PATH)], cwd=PROJECT_ROOT, check=True)

    with open(DATA_PATH, 'rb') as file:
        return pickle.load(file)


def normalize(series):
    '''Scale a series to [0, 1].'''

    lo, hi = series.min(), series.max()

    return (series - lo) / (hi - lo) if hi > lo else series * 0


def get_title(movie_id):
    '''Return the movie title for a given ID.'''

    result = movies.loc[movies['movie_id'] == movie_id, 'title']
    return result.iloc[0] if not result.empty else f'Unknown ({movie_id})'


def find_movies(query):
    '''Return movies whose titles contain the query, ignoring case.'''

    return movies.loc[
        movies['title'].str.contains(query, case=False, regex=False, na=False),
        ['movie_id', 'title']
    ]


def hybrid_recommendations(movie_id, n=5, alpha=0.5):
    '''Return the top n movies by weighted combination of collab and content scores.'''

    collab = item_similarity_df[movie_id] if movie_id in item_similarity_df.columns else pd.Series(dtype=float)
    content = genre_similarity_df[movie_id]

    common = collab.index.intersection(content.index)
    common = common[common != movie_id]

    hybrid = alpha * normalize(collab[common]) + (1 - alpha) * normalize(content[common])
    top = hybrid.sort_values(ascending=False).iloc[:n]

    results = [{'title': get_title(mid), 'hybrid_score': round(score, 4)} for mid, score in top.items()]
    print(f'Hybrid recommendations for "{get_title(movie_id)}" (alpha={alpha}):')

    return pd.DataFrame(results)

data = get_data()
item_similarity_df = data['item_similarity_df']
genre_similarity_df = data['genre_similarity_df']
movies = data['movies']

st.title('MovieLens Recommender')
query = st.text_input('Movie title', placeholder='Toy Story')
alpha = st.slider('Alpha: (0 = content based, 1 = collaborative)', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

if query.strip():
    matches = find_movies(query.strip())

    if matches.empty:
        st.warning('No matching movies found.')
    else:
        selected_movie = st.selectbox(
            'Choose a movie',
            matches.itertuples(index=False),
            format_func=lambda movie: movie.title
        )
        result = hybrid_recommendations(selected_movie.movie_id, alpha=alpha)
        st.dataframe(result)
