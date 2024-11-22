import pandas as pd
import numpy as np
def process_data(row):
  return f"{row['type']} {row['listed_in']} {row['description']}".replace(",", "")

def prepare_train_data(data_paths):
  df = pd.read_csv(data_paths)
  meta_data = df[['show_id','type', 'listed_in', 'description']].reset_index(drop=True) 
  combined_meta_data = meta_data.apply(process_data, axis=1).tolist()
  train_data = [row.split() for row in combined_meta_data]
  return train_data
  
def get_average_words(tokens, model, vector_size, vocab):
  n = 0
  final_sentense_vector = np.zeros((vector_size,))
  for token in tokens:
    if token in vocab:
      n += 1
      final_sentense_vector += model.wv[token]
  if n > 0:
    final_sentense_vector /= n
  return final_sentense_vector


def get_sentense_vector(data, model, vector_size):
  sentenses_vectors = []
  for row in data:
    vocab = model.wv.key_to_index
    sentense_vector= get_average_words(row, model, vector_size, vocab)
    sentenses_vectors.append(sentense_vector)
  return sentenses_vectors