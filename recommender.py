
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





  






