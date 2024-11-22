
from gensim.models import Word2Vec
from utils import prepare_train_data, get_sentense_vector
import joblib

#model parameters
VECTOR_SIZE = 128
WINDOW = 5
NEGATIVE_SAMPLES = 50
DATA_PATH = 'netflix_titles.csv'

train_data = prepare_train_data(DATA_PATH)
model =  Word2Vec(train_data, vector_size= VECTOR_SIZE, window=WINDOW, negative= NEGATIVE_SAMPLES, min_count=1, workers=4)
movies_vectors = get_sentense_vector(train_data, model, VECTOR_SIZE)
joblib.dump(movies_vectors, 'movies_vectors.pkl')
model.save('train_model.model')

# def recommender(movie_id):
#     # Check if the movie exists
#     movie = meta_data.loc[meta_data['show_id'] == movie_id]
#     if movie.empty:
#         print("Movie not found")
#         return

#     # Process the movie
#     processed_movie = process_data(movie.iloc[0])
#     tokenized_movie = processed_movie.split()
#     movie_vector = get_average_words(tokenized_movie, model, vector_size, model.wv.key_to_index)

#     # Calculate similarity scores
#     similarity_scores = cosine_similarity([movie_vector], sentense_vectors)
#     similar_movies = list(enumerate(similarity_scores[0]))
#     sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]

#     # Display the top 10 similar movies
#     print("Top 10 similar movies:")
#     recommended_movies = df.iloc[[i for i, _ in sorted_similar_movies]]
#     return recommended_movies[['title', 'description', 'type']]





  






