# MovieLens recommendations

A Streamlit movie recommender built with the MovieLens 100K dataset. Choose a
movie title, then adjust the alpha slider to blend collaborative filtering with
genre-based content filtering.

## Run locally

Fork [the repository](https://github.com/gperdrizet/movie-lens-recommendations)
on GitHub, then clone your fork and enter the project directory:

```bash
git clone git@github.com:YOUR-GITHUB-USERNAME/movie-lens-recommendations.git
cd movie-lens-recommendations
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run recommender/app.py
```

Streamlit will display the local URL in the terminal. The first run downloads
the MovieLens data and creates the recommendation artifacts in `data/`.
