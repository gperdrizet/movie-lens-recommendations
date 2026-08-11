'''Helper script to generate hybrid filtering recommendation engine
data aftifacts'''

import io
import pickle
import zipfile

import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def download():
    '''Get Movie Lens dataset'''

    # Use Python's requests library to get zip archive of data
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    response = requests.get(url)

    # Create read-only zipfile object in memory from response content
    return zipfile.ZipFile(io.BytesIO(response.content))


def load(z):
    '''Loads data from zip archive'''

    # Load ratings
    ratings = pd.read_csv(
        z.open('ml-100k/u.data'),
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    ).drop(columns=['timestamp'])

    # Load genre/name list
    genre_names = pd.read_csv(
        z.open('ml-100k/u.genre'),
        sep='|',
        names=['genre', 'genre_id'],
        encoding='latin-1'
    )['genre'].tolist()

    # Load movie data
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
    '''Builds user-item and item-item matricies'''

    # Make user-item matrix (rows = users, columns = movies)
    print('Building user-item matrix... ', end='')
    user_item_matrix = ratings.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating'
    )
    print('Done.')

    # Fill unrated entries with 0 for cosine similarity
    user_item_filled = user_item_matrix.fillna(0)

    # Make item-item similarity matrix
    # Transpose so items are rows, then compute pairwise cosine similarity
    print('Building item-item similarity matrix... ', end='')
    item_similarity = cosine_similarity(user_item_filled.T)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    print('Done.')

    # Build genre feature matrix (rows = movies, columns = genres)
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

    z = download()
    ratings, genre_names, movies = load(z)
    item_similarity_df, genre_similarity_df = build(
        ratings,
        genre_names,
        movies
    )

    result = {
        'item_similarity_df': item_similarity_df,
        'genre_similarity_df': genre_similarity_df
    }

    return result

if __name__ == '__main__':
    
    # Do the data asset build
    result = main()

    # Save the result
    with open('data/matricies.pkl', 'wb') as f:
        pickle.dump(result, f)