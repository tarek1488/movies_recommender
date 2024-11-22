from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models import Word2Vec
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_average_words, process_data

app = FastAPI()

# Load pre-trained models and data
model = Word2Vec.load('train_model.model')
movies_vectors = joblib.load('movies_vectors.pkl')
df = pd.read_csv('netflix_titles.csv')


# Pydantic model for request validation
class RecommendationRequest(BaseModel):
    show_id: str


@app.get("/")
def home():
    return {"message": "Welcome to the Movie Recommender API!"}


@app.post("/recommend")
def get_recommendations(data: RecommendationRequest):
    try:
        # Fetch the movie details
        movie = df.loc[df['show_id'] == data.show_id]
        if movie.empty:
            raise HTTPException(status_code=404, detail=f"No movie found with show_id: {data.show_id}")

        # Process the movie details
        processed_movie = process_data(movie.iloc[0])
        tokenized_movie = processed_movie.split()
        movie_vector = get_average_words(tokenized_movie, model, 128, model.wv.key_to_index)

        # Calculate similarity scores
        similarity_scores = cosine_similarity([movie_vector], movies_vectors)
        similar_movies = list(enumerate(similarity_scores[0]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]

        # Get recommended movies
        recommended_movies = df.iloc[[i for i, _ in sorted_similar_movies]]
        recommendation = recommended_movies[['title', 'description', 'type']].to_dict(orient='records')

        return recommendation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


