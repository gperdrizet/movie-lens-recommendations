'''Streamlit app for Movie Lens hybrid filtering recommendations'''

import pickle
import pandas as pd
import streamlit as st

from generate_data import DATA_PATH, generate_data


@st.cache_resource
def get_data():
    # Loading the matrices is expensive, so Streamlit keeps this resource for
    # the lifetime of the server instead of unpickling it on every rerun.

    if DATA_PATH.exists():
        with open(DATA_PATH, 'rb') as file:
            data = pickle.load(file)

        # Older artifacts did not store movie titles. Regenerate those files so
        # title selection and result labels always have the required metadata.
        if 'movies' in data:
            return data

    # The generator owns the shared output path and creates data/ if needed.
    with st.spinner("Generating data artifacts...", show_time=True):
        return generate_data()


def normalize(series):
    '''Scale a series to [0, 1].'''

    lo, hi = series.min(), series.max()

    # A constant series has no useful range. Returning zero prevents a divide
    # by zero and lets the other recommender signal determine the ranking.
    return (series - lo) / (hi - lo) if hi > lo else series * 0


def get_title(movie_id):
    '''Return the movie title for a given ID.'''

    result = movies.loc[movies['movie_id'] == movie_id, 'title']
    return result.iloc[0] if not result.empty else f'Unknown ({movie_id})'


def hybrid_recommendations(movie_id, n=5, alpha=0.5):
    '''Return the top n movies by weighted combination of collab and content scores.'''

    # Collaborative scores measure similar rating patterns. Content scores
    # measure similar genre vectors. Both matrices use MovieLens IDs as labels.
    collab = item_similarity_df[movie_id] if movie_id in item_similarity_df.columns else pd.Series(dtype=float)
    content = genre_similarity_df[movie_id]

    # Score only movies available to both approaches, excluding the selected
    # movie before it can be returned as its own recommendation.
    common = collab.index.intersection(content.index)
    common = common[common != movie_id]

    # The matrices have different score ranges, so normalize them before alpha
    # sets the balance: 0 is content-only and 1 is collaborative-only.
    hybrid = alpha * normalize(collab[common]) + (1 - alpha) * normalize(content[common])
    top = hybrid.sort_values(ascending=False).iloc[:n]

    # Convert internal IDs back to titles for the table shown in the app.
    results = [{'title': get_title(mid), 'hybrid_score': round(score, 4)} for mid, score in top.items()]
    print(f'Hybrid recommendations for "{get_title(movie_id)}" (alpha={alpha}):')

    return pd.DataFrame(results)

data = get_data()
# Unpack the artifact once so recommendation functions can reuse the matrices
# across Streamlit reruns.
item_similarity_df = data['item_similarity_df']
genre_similarity_df = data['genre_similarity_df']
movies = data['movies']

st.title('MovieLens Recommender')
# A select box limits input to MovieLens records and supports title search.
movie_options = list(movies[['movie_id', 'title']].itertuples(index=False))
selected_movie = st.selectbox(
    'Movie title',
    movie_options,
    index=None,
    placeholder='Start typing a movie title',
    format_func=lambda movie: movie.title
)
alpha = st.slider('Alpha: (0 = content based, 1 = collaborative)', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

if selected_movie:
    # The picker returns the record, including its numeric matrix identifier.
    result = hybrid_recommendations(selected_movie.movie_id, alpha=alpha)
    st.dataframe(result)
