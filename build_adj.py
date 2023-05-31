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

cache_dir = './.vector_cache/glove/'
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
print(glove.vectors.size())

# data_file = './dataset/20ng.pkl'
# data_file = './dataset/rcv1v2_bow_8000.pkl'
# data_file = './dataset/wiki_bow_10000.pkl'

data_file = './dataset/r8.pkl'
# data_file = './data/20NG/20ng.pkl'
# data_file = './data/TMN/tag_my_news.pkl'
with open(data_file, 'rb') as f:
    data = pickle.load(f)

# voc = data['voc2000']
voc = data['voc']
# voc = data[1]
# voc = data[1]
print(len(voc))

voc_id = []
ignore_id = []
for i in range(len(voc)):
    if voc[i] not in glove.stoi:
        voc_id.append(glove.stoi['.'])
        print(voc[i])
        ignore_id.append(i)
        continue
    voc_id.append(glove.stoi[voc[i]])

voc_vector = glove.vectors[voc_id].numpy()
print('voc vector shape: ', voc_vector.shape)
voc_vector[ignore_id] = np.zeros([len(ignore_id), 300])
voc_adj = cosine_simlarity(voc_vector, voc_vector)
# m = [1, 2, 3, 4]
m = [4, 8, 16, 32]
# m = [4, 8, 16, 32]
adj = {}
threshold = [0.6, 0.5, 0.35, 0.2]
for layer in range(len(m)):
    adj_t = np.zeros_like(voc_adj)
    print('layer = ', layer)
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
    print('c = ', c)
    print(adj_t.sum())
    # adj_t = adj_t / np.sum(adj_t, axis=1)

    adj['layer_' + str(layer + 1)] = sp.csr_matrix(adj_t)
    print('build layer ', layer + 1)


# f = open('../Sawtooth_1/data/r8_spadj_2468_7.pkl', 'wb')
# f = open('../Sawtooth_1/data/20ng2w_spadj_2468_7_1.pkl', 'wb')
# f = open('../Sawtooth_1/data/tmn_spadj_2468_7.pkl', 'wb')
# f = open('./dataset/rcv1_spadj_2468.pkl', 'wb')
# f = open('./dataset/wiki_adj_2468.pkl', 'wb')
f = open('dataset/r8_spadj_final.pkl', 'wb')
pickle.dump(adj, f)
f.close()

print('build over!')




