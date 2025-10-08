import pandas as pd
import re
import string
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

# splits = {'train': 'train.parquet', 'validation': 'validation.parquet', 'test': 'test.parquet'}
# df = pd.read_parquet("hf://datasets/cornell-movie-review-data/rotten_tomatoes/" + splits["train"])
splits = {'train': 'train.parquet', 'validation': 'validation.parquet', 'test': 'test.parquet'}
df_train = pd.read_parquet("hf://datasets/cornell-movie-review-data/rotten_tomatoes/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/cornell-movie-review-data/rotten_tomatoes/" + splits["test"])
0
0# print(df.head())
# def bag_of_word(corpus: string) -> List:

#50k w drugim
0
def prepocess_data(sentence: str) -> str:
        sentence = sentence.lower()
        sentence = re.sub(r'\W', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = re.sub(r'\d', ' ', sentence)
        # text = text.translate(str.maketrans("", "", string.punctuation))
        sentence = sentence.strip()
        return sentence.split()

def create_vocab_index(corpus: list) -> dict:
    vocab = {}
    idx = 0
    for sentence in corpus:
        for token in prepocess_data(sentence):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def create_bow_vector(vocab: set, sentence: str) -> list:
     vec = [0] * len(vocab)
     for token in sentence:
          if token in vocab:
               index = vocab[token]
               vec[index] += 1
     return vec



vocab = defaultdict(int)
if __name__ == "__main__":
    processed_corpus = [prepocess_data(sentence=sentence) for sentence in df['text'][:5]]
    # vocabulary = create_vocab(df['text'][:5])
    vocabulary = create_vocab_index(df['text'][:5])
    bow_vectors = [create_bow_vector(vocabulary, sentence) for sentence in processed_corpus]
    print(vocabulary)
    print(bow_vectors[:1])
    X_train = [create_bow_vector(vocabulary, sentence) for sentence in df_train['text']]
    y_train = df_train['label'].tolist()
    X_test = [create_bow_vector(vocabulary, sentence) for sentence in df_test['text']]
    y_test = df_test['label'].tolist()

    X_train_tensor = torch.tensor(bow_vectors, dtype=torch.float32)
        