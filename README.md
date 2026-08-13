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

## Deploy to Render.com

You will need a free [Render.com](https://render.com/) account. I recommend registering via your GitHub account, this is how we will deploy the app.

### Create new web service

From your Render dashboard, select '+New' at the top left and choose 'Web service'. Then choose 'Public Git Repository' and paste in the public GitHub URL for the repository and click connect.

### Configure the service

Set the following values:

1. **Name**: whatever you want
2. **Language**: Python 3
3. **Branch**: main (can be whatever branch you want to deploy)
4. **Root directory**: set to repo root `/`
5. **Build command**: install Python dependencies with `pip install -r requirements.txt`
6. **Start command**: start the Streamlit app with: `streamlit run recommender/app.py`
7. **Instance type**: use free, low compute resources, but should be enough for this demo.

Click deploy!