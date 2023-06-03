import copy

import numpy as np
import torch
import pickle
import os
from nltk.corpus import wordnet as wn
import torchtext.vocab as vocab
import scipy.sparse as sp

# data_file = './data/R8/r8.pkl'
# with open(data_file, 'rb') as f:
#     data = pickle.load(f)

def cosine_simlarity(A, B):
    # A: N*D, B: N*D
    realmin = 2.2e-10
    [N, D] = A.shape
    inter_product = np.matmul(A, np.transpose(B))  # N*N
    len_A = np.sqrt(np.sum(A * A, axis=1, keepdims=True))
    len_B = np.sqrt(np.sum(B * B, axis=1, keepdims=True))
    len_AB = np.matmul(len_A, np.transpose(len_B))
    cos_AB = inter_product / (len_AB + realmin)
    cos_AB[(np.arange(N), np.arange(N))] = 1
    return cos_AB

def get_word_embedding_from_glove(cache_dir):
    glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)

    return glove


def build_adj_matrix(dataset_dir, adj_save_dir, word_emb):
    m = [2, 4, 6, 8]
    adj = {}
    threshold = [0.6, 0.5, 0.35, 0.2]

    with open(dataset_dir, 'rb') as f:
        data = pickle.load(f)

    voc = data['voc20000']
    voc_id = []
    ignore_id = []
    for i in range(len(voc)):
        if voc[i] not in word_emb.stoi:
            voc_id.append(word_emb.stoi['.'])
            print(voc[i])
            ignore_id.append(i)
            continue
        voc_id.append(word_emb.stoi[voc[i]])

    voc_vector = word_emb.vectors[voc_id].numpy()
    voc_vector[ignore_id] = np.zeros([len(ignore_id), 300])
    voc_adj = cosine_simlarity(voc_vector, voc_vector)

    for layer in range(len(m)):
        adj_t = np.zeros_like(voc_adj)
        c = 0
        for i in range(len(voc)):
            if i in ignore_id:
                adj_t[i][i] = 1
                continue
            v = np.sort(voc_adj[i])[::-1][m[layer]]
            if v < threshold[layer]:
                v = threshold[layer]
            for j in range(len(voc)):
                if voc_adj[i][j] >= v:
                    c += 1
                    adj_t[i][j] = 1
                else:
                    adj_t[i][j] = 0

        adj['layer_' + str(layer + 1)] = sp.csr_matrix(adj_t)
        print('====> Layer {} has been built.'.format(layer + 1))

    f = open(adj_save_dir, 'wb')
    pickle.dump(adj, f)
    f.close()

if __name__ == '__main__':
    cache_dir = './data/vector/glove/'
    dataset_dir = './data/20ng/20ng.pkl'
    adj_save_dir = './data/20ng/20ng_adj.pkl'
    print('===> Load word embedding from glove')
    word_emb = get_word_embedding_from_glove(cache_dir=cache_dir)
    print('===> Build adj matrix from data')
    build_adj_matrix(dataset_dir=dataset_dir, adj_save_dir=adj_save_dir, word_emb=word_emb)
    print('===> Build over!!!')



