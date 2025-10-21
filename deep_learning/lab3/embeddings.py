import numpy as np
import torch

def load_embeddings(path):
    embeddings_dict = dict()
    f = open(path, "r")
    for line in f:
        parts = line.strip().split(" ")
        word = parts[0]
        embeddings = parts[1:]
        embeddings_dict[word] = embeddings
    f.close()
    return embeddings_dict

def matrix(vocab, dict, no_normal, dim=300):
    matrix = np.random.normal(0,1,(len(vocab.itos), dim))
    matrix[0] = np.zeros(dim)
    if no_normal:
        for token, index in vocab.stoi.items():
            if token in dict:
                matrix[index] = dict[token]

    return torch.tensor(matrix)
