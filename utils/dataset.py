import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.sparse as sp
import torch

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
    for doc_idx, doc in enumerate(x):
        indices_y = np.nonzero(doc)[0]
        l = []
        for i in range(len(indices_y)):
            value = doc[indices_y[i]]
            for _ in range(int(value)):
                l.append(indices_y[i])
        random.seed(2023)
        random.shuffle(l)
        l_1 = l[:int(len(l) * ratio)]
        l_2 = l[int(len(l) * ratio):]
        for l1_value in l_1:
            x_1[doc_idx][l1_value] += 1
        for l2_value in l_2:
            x_2[doc_idx][l2_value] += 1
    return x_1, x_2

class CustomDataset_ppl_20ng(Dataset):
    def __init__(self, data_file, adj_file):
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

        del adj

        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        return [torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.squeeze(self.train_data_h[0][index]).float(), torch.squeeze(self.train_data_h[1][index]).float(),
                torch.squeeze(self.train_data_h[2][index]).float(), torch.squeeze(self.train_data_h[3][index]).float()], torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

def get_loader_ppl_20ng(topic_data_file, adj_file, batch_size=200, shuffle=True, num_workers=4):
    dataset = CustomDataset_ppl_20ng(topic_data_file, adj_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc, dataset.adj

class CustomDataset_ppl_rcv1(Dataset):
    def __init__(self, data_file, adj_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.voc = data[1]
        self.train_data, self.test_data = gen_ppl_doc(data[0].astype("int8").toarray())
        self.train_data = sp.csr_matrix(self.train_data)
        self.test_data = sp.csr_matrix(self.test_data)
        del data

        with open(adj_file, 'rb') as f:
            adj = pickle.load(f)

        self.adj = []
        self.adj.append(adj['layer_1'])
        self.adj.append(adj['layer_2'])
        self.adj.append(adj['layer_3'])
        self.adj.append(adj['layer_4'])

        self.train_data_h = [0] * 4
        for layer in range(4):
            print('layer ', layer)
            adj_norm = normalize(self.adj[layer])
            if layer == 0:
                self.train_data_h[0] = self.train_data @ adj_norm
            else:
                self.train_data_h[layer] = self.train_data_h[layer - 1] @ adj_norm
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

def get_loader_ppl_rcv1(topic_data_file, adj_file, batch_size=200, shuffle=True, num_workers=4):
    dataset = CustomDataset_ppl_rcv1(topic_data_file, adj_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc, dataset.adj

class CustomDataset_ppl_r8(Dataset):
    def __init__(self, data_file, adj_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.voc = data['voc']
        self.train_data, self.test_data = gen_ppl_doc(data['fea'])
        del data

        self.train_data = sp.csr_matrix(self.train_data)
        self.test_data = sp.csr_matrix(self.test_data)

        with open(adj_file, 'rb') as f:
            adj = pickle.load(f)

        self.adj = []
        self.adj.append(adj['layer_1'])
        self.adj.append(adj['layer_2'])
        self.adj.append(adj['layer_3'])
        self.adj.append(adj['layer_4'])

        self.train_data_h = [0] * 4
        for layer in range(4):
            print('layer ', layer)
            adj_norm = normalize(self.adj[layer])
            if layer == 0:
                self.train_data_h[0] = self.train_data @ adj_norm
            else:
                self.train_data_h[layer] = self.train_data_h[layer - 1] @ adj_norm

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

def get_loader_ppl_r8(topic_data_file, adj_file, batch_size=200, shuffle=True, num_workers=4):
    dataset = CustomDataset_ppl_r8(topic_data_file, adj_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc, dataset.adj

class CustomDataset_clustering(torch.utils.data.Dataset):
    def __init__(self, name, path, adj_file, mode='train'):
        super(CustomDataset_clustering, self).__init__()
        self.mode = mode
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if name in ['20ng']:
            train_id = data['train_id']
            test_id = data['test_id']
            label = np.squeeze(np.array(data['label']))
            train_bows = data['data_2000'][train_id]
            train_labels = label[train_id]
            test_bows = data['data_2000'][test_id]
            test_labels = label[test_id]
            vocab = data['voc2000']

        elif name in ['tmn']:
            vocab = data['vocab']
            train_bows = data['train_data']
            train_labels = data['train_labels']
            test_bows = data['test_data']
            test_labels = data['test_labels']

        elif name in ['r8']:
            vocab = data['voc']
            train_bows = data['fea'][:5185]
            train_labels = data['gnd'][:5185]
            test_bows = data['fea'][5485:]
            test_labels = data['gnd'][5485:]

        elif name in ['rcv', 'wiki']:
            vocab = data['vocab']
            train_bows = data['data']
            train_labels = None
            test_bows = None
            test_labels = None
        else:
            raise NotImplementedError(f'unknown dataset: {name}')

        if mode == 'train':
            self.data = train_bows
            self.labels = train_labels
        elif mode == 'test':
            self.data = test_bows
            self.labels = test_labels
        else:
            raise ValueError("argument 'mode' must be either train or test")
        self.vocab = vocab
        del data

        if mode == 'train':
            with open(adj_file, 'rb') as f:
                adj = pickle.load(f)

            self.adj = []
            self.adj.append(adj['layer_1'])
            self.adj.append(adj['layer_2'])
            self.adj.append(adj['layer_3'])
            self.adj.append(adj['layer_4'])

            self.data_h = [0] * 4
            self.distribution = []
            for layer in range(4):
                adj_norm = np.asarray(self.adj[layer] / np.sum(self.adj[layer], axis=1))
                if layer == 0:
                    self.data_h[0] = np.asarray(self.data @ adj_norm)
                else:
                    self.data_h[layer] = self.data_h[layer - 1] @ adj_norm

            del adj

        if self.labels is not None:
            assert self.data.shape[0] == len(self.labels)

    def __getitem__(self, index):
        try:
            data_cur = self.data[index].toarray()
        except:
            data_cur = self.data[index]
        if self.labels is not None:
            if self.mode == 'train':
                return [data_cur.squeeze(), self.data_h[0][index].squeeze(), self.data_h[1][index].squeeze(),
                    self.data_h[2][index].squeeze(), self.data_h[3][index].squeeze()], self.labels[index]
            else:
                return [data_cur.squeeze(), 0, 0, 0, 0], self.labels[index]
        else:
            return [self.data[index].squeeze(), self.data_h[0][index].squeeze(), self.data_h[1][index].squeeze(),
                self.data_h[2][index].squeeze(), self.data_h[3][index].squeeze()], 0

    def __len__(self):
        return self.data.shape[0]

def get_loader_clustering(data_name, data_path, adj_path, mode='train', batch_size=200, shuffle=True, drop_last=True, num_workers=4):
    dataset = CustomDataset_clustering(name=data_name, path=data_path, adj_file=adj_path, mode=mode)
    if mode == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        ), dataset.vocab, dataset.adj
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )