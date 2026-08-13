'''Helper script to generate hybrid filtering recommendation engine
data artifacts'''

import io
import pickle
import zipfile
from pathlib import Path

import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep the artifact beside the application, regardless of the working directory
# used to launch this script or the Streamlit server.
DATA_PATH = PROJECT_ROOT / 'data/matrices.pkl'


def download():
    '''Get Movie Lens dataset'''

    # Download the archive into memory so pandas can read its files directly
    # without extracting the full MovieLens distribution to disk.
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    response = requests.get(url)

    # Create a read-only zipfile object over the downloaded bytes.
    return zipfile.ZipFile(io.BytesIO(response.content))


def load(z):
    '''Loads data from zip archive'''

    # Ratings describe explicit user preference. Timestamps are unnecessary for
    # this item-similarity demonstration, so remove them after loading.
    ratings = pd.read_csv(
        z.open('ml-100k/u.data'),
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    ).drop(columns=['timestamp'])

    # The genre file defines the order of binary genre columns in u.item.
    genre_names = pd.read_csv(
        z.open('ml-100k/u.genre'),
        sep='|',
        names=['genre', 'genre_id'],
        encoding='latin-1'
    )['genre'].tolist()

    # Keep each movie's ID, display title, and genre flags for later lookup and
    # content-based similarity calculation.
    movie_cols = ['movie_id', 'title', 'release_date', 'video_release', 'imdb_url'] + genre_names

    movies = pd.read_csv(
        z.open('ml-100k/u.item'),
        sep='|',
        names=movie_cols,
        encoding='latin-1'
    )[['movie_id', 'title'] + genre_names]

    print(f'Ratings: {len(ratings):,} rows')
    print(f'Movies:  {len(movies):,} rows')
    print(f'Users:   {ratings["user_id"].nunique():,}')
    print(f'Rating scale: {ratings["rating"].min()}-{ratings["rating"].max()}')

    return ratings, genre_names, movies


def build(ratings, genre_names, movies):
    '''Builds user-item and item-item matrices'''

    # Pivot ratings into the matrix where every row is a user and every column
    # is a movie. Missing values mean the user has not rated that movie.
    print('Building user-item matrix... ', end='')
    user_item_matrix = ratings.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating'
    )
    print('Done.')

    # Cosine similarity requires numeric vectors. Zero represents an unrated
    # movie while preserving the original NaN matrix for other use cases.
    user_item_filled = user_item_matrix.fillna(0)

    # Transpose so each movie is a vector of user ratings. The resulting matrix
    # lets the app look up collaborative similarity by movie ID.
    print('Building item-item similarity matrix... ', end='')
    item_similarity = cosine_similarity(user_item_filled.T)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    print('Done.')

    # Each movie is also represented by its binary genre flags. Cosine
    # similarity over these vectors supplies the content-based signal.
    print('Building genre similarity matrix... ', end='')
    genre_matrix = movies[genre_names].values
    genre_similarity = cosine_similarity(genre_matrix)

    genre_similarity_df = pd.DataFrame(
        genre_similarity,
        index=movies['movie_id'],
        columns=movies['movie_id']
    )
    print('Done.')

    return item_similarity_df, genre_similarity_df


def main():
    '''Runs the data generation'''

    # Build the inputs once, then package only what the web app needs at run
    # time. This keeps the Streamlit startup path independent of the download.
    z = download()
    ratings, genre_names, movies = load(z)
    item_similarity_df, genre_similarity_df = build(
        ratings,
        genre_names,
        movies
    )

    result = {
        'item_similarity_df': item_similarity_df,
        'genre_similarity_df': genre_similarity_df,
        'movies': movies
    }

    return result


def generate_data():
    '''Build and save the recommendation data artifact.'''

    result = main()
    # A fresh clone has no data directory, so create it before opening the file.
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Pickle preserves pandas indexes and labels used for movie-ID lookups.
    with open(DATA_PATH, 'wb') as file:
        pickle.dump(result, file)

    return result


if __name__ == '__main__':
    generate_data()
