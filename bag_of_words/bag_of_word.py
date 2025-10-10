import re
from collections import Counter
from datasets import load_dataset


def preprocess_data(sentence: str) -> str:
    sentence = sentence.lower()
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\d', ' ', sentence)
    return sentence.strip()


def create_vocab(corpus: list) -> dict:
    vocab = {}
    idx = 0
    for sentence in corpus:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def create_bow_vector(vocab: dict, sentence: str) -> list:
    vec = [0] * len(vocab)
    for token in sentence.split():
        if token in vocab:
            index = vocab[token]
            vec[index] += 1
    return vec



if __name__ == "__main__":
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    train_texts = [preprocess_data(text) for text in ds['train']['text']]
    vocab = create_vocab(train_texts)
    bow_vectors = [create_bow_vector(vocab, sentence) for sentence in train_texts]
    print(bow_vectors[0])
    print(len(vocab))
        