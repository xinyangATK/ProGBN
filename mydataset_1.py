import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio

class CustomDataset(Dataset):
    def __init__(self, data_file):
        # with open(data_file, 'rb') as f:
        #     data = pickle.load(f)
        # self.train_data = data['doc_bow'].toarray()
        # self.N, self.vocab_size = self.train_data.shape
        # self.voc = data['word2id']
        data = sio.loadmat('mnist_data/mnist')
        self.train_data = np.array(np.ceil(data['train_mnist'] * 5), order='C')  # 0-1
        self.test_data = np.array(np.ceil(data['test_mnist'] * 5), order='C')  # 0-1
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomDataset(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size

class CustomDataset_txt(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.train_data = data_all[data['train_id']].astype("int32")
        #self.train_data = data_all.astype("int32")
        self.voc = data['voc2000']
        train_label = [data['label'][i] for i in data['train_id']]
        test_label = [data['label'][i] for i in data['test_id']]
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_loader_txt(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomDataset_txt(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc


class CustomTestDataset_txt(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.test_data = data_all[data['test_id']].astype("int32")
        self.voc = data['voc2000']
        train_label = [data['label'][i] for i in data['train_id']]
        test_label = [data['label'][i] for i in data['test_id']]
        self.N, self.vocab_size = self.test_data.shape

    def __getitem__(self, index):
        topic_data = self.test_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_test_loader_txt(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTestDataset_txt(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio
import torch

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

class CustomTrainDataset_txt_ppl(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.voc = data['voc2000']
        data_train = data['data_2000'].toarray()
        # data_test = data['data_2000'][data['test_id']].toarray()

        self.train_data, self.test_data = gen_ppl_doc(data_train.astype("int32"))
        self.train_data_1, _ = gen_ppl_doc(data['data2000_1'].toarray().astype("int32"))
        self.train_data_2, _ = gen_ppl_doc(data['data2000_2'].toarray().astype("int32"))
        self.train_data_3, _  = gen_ppl_doc(data['data2000_3'].toarray().astype("int32"))
        self.train_data_4, _ = gen_ppl_doc(data['data2000_4'].toarray().astype("int32"))

        del data

        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        return [torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(np.squeeze(self.train_data_1[index])).float(),
                torch.from_numpy(np.squeeze(self.train_data_2[index])).float(), torch.from_numpy(np.squeeze(self.train_data_3[index])).float(),
                torch.from_numpy(np.squeeze(self.train_data_4[index])).float()], torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

def get_train_loader_txt_ppl(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTrainDataset_txt_ppl(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc

class CustomTrainDataset_txt_ppl_rcv1(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.voc = data['voc']
        self.data_all = data['bow']
        self.data_1 = data['data8000_1']
        self.data_2 = data['data8000_2']
        self.data_3 = data['data8000_3']
        self.data_4 = data['data8000_4']

        del data

        self.N, self.vocab_size = self.data_all.shape

    def __getitem__(self, index):
        ret = self.data_all[index].toarray()
        ret_1 = self.data_1[index].toarray()
        ret_2 = self.data_2[index].toarray()
        ret_3 = self.data_3[index].toarray()
        ret_4 = self.data_4[index].toarray()

        train_data, test_data = gen_ppl_doc(ret.astype("int32"))
        train_data_1, _ = gen_ppl_doc(ret_1.astype("int32"))
        train_data_2, _ = gen_ppl_doc(ret_2.astype("int32"))
        train_data_3, _  = gen_ppl_doc(ret_3.astype("int32"))
        train_data_4, _ = gen_ppl_doc(ret_4.astype("int32"))

        return [torch.from_numpy(np.squeeze(train_data)).float(), torch.from_numpy(np.squeeze(train_data_1)).float(),
                torch.from_numpy(np.squeeze(train_data_2)).float(), torch.from_numpy(np.squeeze(train_data_3)).float(),
                torch.from_numpy(np.squeeze(train_data_4)).float()], torch.from_numpy(np.squeeze(test_data)).float()

    def __len__(self):
        return self.N

def get_train_loader_txt_ppl_rcv1(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTrainDataset_txt_ppl_rcv1(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc

class CustomDataset_txt_ppl_wiki(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.data_all = data['test_bow_10000']
        self.voc = data['voc_10000']
        del data
        self.N, self.vocab_size = self.data_all.shape

    def __getitem__(self, index):
        ret = self.data_all[index].toarray()
        train_data, test_data = gen_ppl_doc(ret.astype("int32"))
        return torch.from_numpy(np.squeeze(train_data)).float(), torch.from_numpy(np.squeeze(test_data)).float()

    def __len__(self):
        return self.N

def get_train_loader_txt_ppl_wiki(topic_data_file, batch_size=200, shuffle=True, num_workers=100):
    dataset = CustomDataset_txt_ppl_wiki(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc


class CustomTestDataset_txt_ppl(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_train = data['data_2000'][data['train_id']].toarray()
        data_test = data['data_2000'][data['test_id']].toarray()
        self.train_data, self.test_data = gen_ppl_doc(data_test.astype("int32"))
        self.voc = data['voc2000']
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        return torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

def get_test_loader_txt_ppl(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTestDataset_txt_ppl(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc

# class CustomDataset(Dataset):
#     def __init__(self, data_file):
#         # with open(data_file, 'rb') as f:
#         #     data = pickle.load(f)
#         # self.train_data = data['doc_bow'].toarray()
#         # self.N, self.vocab_size = self.train_data.shape
#         # self.voc = data['word2id']
#         data = sio.loadmat('mnist_data/mnist')
#         self.train_data = np.array(np.ceil(data['train_mnist'] * 5), order='C')  # 0-1
#         self.test_data = np.array(np.ceil(data['test_mnist'] * 5), order='C')  # 0-1
#         self.N, self.vocab_size = self.train_data.shape
#
#     def __getitem__(self, index):
#         topic_data = self.train_data[index, :]
#         return np.squeeze(topic_data), 1
#
#     def __len__(self):
#         return self.N
#
# def get_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
#     dataset = CustomDataset(topic_data_file)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
#                       drop_last=True), dataset.vocab_size
#


# class CustomDataset(Dataset):
#     def __init__(self, data_file):
#         with open(data_file, 'rb') as f:
#             data = pickle.load(f)
#         train_id = data['train_id']
#         test_id = data['test_id']
#         train_data = data['data_2000']
#         test_data = data['data_2000'][test_id]
#         train_label = np.array(data['label'])[train_id]
#         test_label = np.array(data['label'])[test_id]
#         voc = data['voc2000']
#         self.train_data = train_data
#         self.N, self.vocab_size = self.train_data.shape
#         self.voc = voc
#
#     def __getitem__(self, index):
#         topic_data = self.train_data[index].toarray()
#         return np.squeeze(topic_data), 1
#
#     def __len__(self):
#         return self.N