import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio
import scipy.sparse as sp
import torch
from sampler import Basic_Sampler

def normalize(A):
    rowsum = np.array(A.sum(1))
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_A_inv = sp.diags(r_inv)
    A = r_A_inv.dot(A)
    return A

def gen_ppl_doc(x, ratio=0.8):
    """
    inputs:
        x: N x V, np array,
        ratio: float or double,
    returns:
        x_1: N x V, np array, the first half docs whose length equals to ratio * doc length,
        x_2: N x V, np array, the second half docs whose length equals to (1 - ratio) * doc length,
    """
    import random
    x_1, x_2 = np.zeros_like(x), np.zeros_like(x)
    # indices_x, indices_y = np.nonzero(x)
    for doc_idx, doc in enumerate(x):
        indices_y = np.nonzero(doc)[0]
        l = []
        for i in range(len(indices_y)):
            value = doc[indices_y[i]]
            for _ in range(int(value)):
                l.append(indices_y[i])
        random.seed(2020)
        random.shuffle(l)
        l_1 = l[:int(len(l) * ratio)]
        l_2 = l[int(len(l) * ratio):]
        for l1_value in l_1:
            x_1[doc_idx][l1_value] += 1
        for l2_value in l_2:
            x_2[doc_idx][l2_value] += 1
    return x_1, x_2

def gen_ppl_doc_train(x, ratio=0.8):
    """
    inputs:
        x: N x V, np array,
        ratio: float or double,
    returns:
        x_1: N x V, np array, the first half docs whose length equals to ratio * doc length,
        x_2: N x V, np array, the second half docs whose length equals to (1 - ratio) * doc length,
    """
    print('start build train！')
    import random
    x_1 = np.zeros_like(x)
    # indices_x, indices_y = np.nonzero(x)
    for doc_idx, doc in enumerate(x):
        indices_y = np.nonzero(doc)[0]
        l = []
        for i in range(len(indices_y)):
            value = doc[indices_y[i]]
            for _ in range(int(value)):
                l.append(indices_y[i])
        random.seed(2020)
        random.shuffle(l)
        l_1 = l[:int(len(l) * ratio)]
        for l1_value in l_1:
            x_1[doc_idx][l1_value] += 1
    return x_1, 0

class CustomTrainDataset_txt_ppl(Dataset):
    def __init__(self, data_file, adj_file):
        self.sampler = Basic_Sampler('gpu')
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.voc = data['voc2000']
        data_train = data['data_2000'].toarray()
        self.train_data, self.test_data = gen_ppl_doc(data_train.astype("int32"))

        del data

        with open(adj_file, 'rb') as f:
            adj = pickle.load(f)

        self.adj = []
        self.adj.append(adj['layer_1'])
        self.adj.append(adj['layer_2'])
        self.adj.append(adj['layer_3'])
        self.adj.append(adj['layer_4'])

        self.prob = [0] * 4
        self.prob[0] = self.train_data @ self.adj[0]
        self.distribution = []
        self.train_data_h = [0] * 4
        for layer in range(4):
            adj_norm = self.adj[layer] / np.sum(self.adj[layer], axis=1)
            if layer == 0:
                self.prob[0] = self.train_data @ adj_norm
            else:
                self.prob[layer] = self.prob[layer - 1] @ adj_norm
            self.train_data_h[layer] = torch.from_numpy(self.prob[layer])
            # self.train_data_h[layer] = torch.from_numpy(self.sampler.poisson(self.prob[layer]))
            # self.distribution.append(torch.distributions.poisson.Poisson(torch.from_numpy(self.prob[layer])))
            # self.train_data_h[layer] = self.distribution[-1].sample()

        del adj


        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        # self.train_data_h = [0] * 4
        # for layer in range(4):
        #     self.train_data_h[layer] = torch.from_numpy(self.sampler.poisson(self.prob[layer]))
        #     # self.train_data_h[layer] = self.distribution[-1].sample()
        return [torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.squeeze(self.train_data_h[0][index]).float(), torch.squeeze(self.train_data_h[1][index]).float(),
                torch.squeeze(self.train_data_h[2][index]).float(), torch.squeeze(self.train_data_h[3][index]).float()], torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

def get_train_loader_txt_ppl(topic_data_file, adj_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTrainDataset_txt_ppl(topic_data_file, adj_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc, dataset.adj

class CustomTrainDataset_txt_ppl_rcv1(Dataset):
    def __init__(self, data_file, adj_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.voc = data[1]
        # self.data_all = data[0]
        self.train_data, self.test_data = gen_ppl_doc(data[0].astype("int8").toarray())
        print('step 1 over!')
        self.train_data = sp.csr_matrix(self.train_data)
        self.test_data = sp.csr_matrix(self.test_data)
        print('step 2 over!')
        del data

        with open(adj_file, 'rb') as f:
            adj = pickle.load(f)

        self.adj = []
        self.adj.append(adj['layer_1'])
        self.adj.append(adj['layer_2'])
        self.adj.append(adj['layer_3'])
        self.adj.append(adj['layer_4'])

        print('step 3 over!')
        self.train_data_h = [0] * 4
        for layer in range(4):
            print('layer ', layer)
            adj_norm = normalize(self.adj[layer])
            if layer == 0:
                self.train_data_h[0] = self.train_data @ adj_norm
            else:
                self.train_data_h[layer] = self.train_data_h[layer - 1] @ adj_norm

        print('step 4 over!')
            # self.train_data_h[layer] = torch.from_numpy(self.sampler.poisson(self.prob[layer]))
            # self.distribution.append(torch.distributions.poisson.Poisson(torch.from_numpy(self.prob[layer])))
            # self.train_data_h[layer] = self.distribution[-1].sample()

        del adj

        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        ret = self.train_data[index].toarray()
        ret_1 = self.train_data_h[0][index].toarray()
        ret_2 = self.train_data_h[1][index].toarray()
        ret_3 = self.train_data_h[2][index].toarray()
        ret_4 = self.train_data_h[3][index].toarray()

        # train_data, test_data = gen_ppl_doc(ret.astype("int32"))
        # train_data_1, _ = gen_ppl_doc(ret_1.astype("int32"))
        # train_data_2, _ = gen_ppl_doc(ret_2.astype("int32"))
        # train_data_3, _  = gen_ppl_doc(ret_3.astype("int32"))
        # train_data_4, _ = gen_ppl_doc(ret_4.astype("int32"))

        return [torch.from_numpy(np.squeeze(ret)).float(), torch.from_numpy(np.squeeze(ret_1)).float(),
                torch.from_numpy(np.squeeze(ret_2)).float(), torch.from_numpy(np.squeeze(ret_3)).float(),
                torch.from_numpy(np.squeeze(ret_4)).float()], torch.from_numpy(np.squeeze(self.test_data[index].toarray())).float()

    def __len__(self):
        return self.N

def get_train_loader_txt_ppl_rcv1(topic_data_file, adj_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTrainDataset_txt_ppl_rcv1(topic_data_file, adj_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc, dataset.adj

class CustomDataset_txt_ppl_r8(Dataset):
    def __init__(self, data_file, adj_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        # self.data_all = data['fea']
        self.voc = data['voc']

        self.train_data, self.test_data = gen_ppl_doc(data['fea'])
        del data

        print('step 1 over!')
        self.train_data = sp.csr_matrix(self.train_data)
        self.test_data = sp.csr_matrix(self.test_data)
        print('step 2 over!')

        with open(adj_file, 'rb') as f:
            adj = pickle.load(f)

        self.adj = []
        self.adj.append(adj['layer_1'])
        self.adj.append(adj['layer_2'])
        self.adj.append(adj['layer_3'])
        self.adj.append(adj['layer_4'])

        print('step 3 over!')
        self.train_data_h = [0] * 4
        for layer in range(4):
            print('layer ', layer)
            adj_norm = normalize(self.adj[layer])
            if layer == 0:
                self.train_data_h[0] = self.train_data @ adj_norm
            else:
                self.train_data_h[layer] = self.train_data_h[layer - 1] @ adj_norm

        print('step 4 over!')

        del adj

        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        ret = self.train_data[index].toarray()
        ret_1 = self.train_data_h[0][index].toarray()
        ret_2 = self.train_data_h[1][index].toarray()
        ret_3 = self.train_data_h[2][index].toarray()
        ret_4 = self.train_data_h[3][index].toarray()

        return [torch.from_numpy(np.squeeze(ret)).float(), torch.from_numpy(np.squeeze(ret_1)).float(),
                torch.from_numpy(np.squeeze(ret_2)).float(), torch.from_numpy(np.squeeze(ret_3)).float(),
                torch.from_numpy(np.squeeze(ret_4)).float()], torch.from_numpy(np.squeeze(self.test_data[index].toarray())).float()

    def __len__(self):
        return self.N

def get_train_loader_txt_ppl_r8(topic_data_file, adj_file, batch_size=200, shuffle=True, num_workers=100):
    dataset = CustomDataset_txt_ppl_r8(topic_data_file, adj_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc, dataset.adj


# with open('./dataset/r8.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print('')