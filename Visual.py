import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from mydataset import *
from model import *
import numpy as np
import matplotlib.pyplot as plt


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=200, help="models used.")
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--n_updates', type=int, default=100, help='parameter of gbn')
parser.add_argument('--MBratio', type=int, default=100, help='parameter of gbn')
parser.add_argument('--topic_size', type=list, default=[128, 64, 32], help='Number of units in hidden layer 1.')
parser.add_argument('--hidden_size', type=list, default=[128, 64, 32], help='Number of units in hidden layer 1.')
parser.add_argument('--vocab_size', type=int, default=2000, help='Number of vocabulary')
parser.add_argument('--embed_size', type=list, default=[50, 50, 50], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-dir', type=str, default='../mnist_data/mnist', help='type of dataset.')
parser.add_argument('--output-dir', type=str, default='torch_phi_output_etm_hier_share', help='type of dataset.')
parser.add_argument('--save-path', type=str, default='saves/gbn_model_weibull_etm_256_128_64_kl_1', help='type of dataset.')
parser.add_argument('--word-vector-path', type=str, default='../process_data/20ng_word_embedding.pkl', help='type of dataset.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, vocab_size = get_loader(args.dataset_dir, batch_size=args.batch_size)
args.vocab_size = vocab_size

args.n_updates = len(train_loader) * args.epochs
args.MBratio = len(train_loader)
# with open(args.word_vector_path, 'rb') as f:
#     data = pickle.load(f)
# word_embedding = torch.from_numpy(data['embedding'])

model = GBN_ETM(args)
model = model.cuda()

if os.path.isfile(args.save_path):
    print('Loading {}'.format(args.save_path))
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("=> no checkpoint found at '{}'".format(args.save_path))
    checkpoint = {
        "epoch": 0,
        "best": float("inf")
    }

# word_embed = model.decoder[0].rho.cpu().detach().numpy()
# topic_embed_1 = model.decoder[0].alphas.cpu().detach().numpy()
# topic_embed_2 = model.decoder[1].alphas.cpu().detach().numpy()
# topic_embed_3 = model.decoder[2].alphas.cpu().detach().numpy()

w_1 = torch.mm(model.decoder[0].rho, torch.transpose(model.decoder[0].alphas, 0, 1))
phi_1 = torch.softmax(w_1, dim=0).cpu().detach().numpy()

w_2 = torch.mm(model.decoder[1].rho, torch.transpose(model.decoder[1].alphas, 0, 1))
phi_2 = torch.softmax(w_2, dim=0).cpu().detach().numpy()

w_3 = torch.mm(model.decoder[2].rho, torch.transpose(model.decoder[2].alphas, 0, 1))
phi_3 = torch.softmax(w_3, dim=0).cpu().detach().numpy()

phi = [phi_1, phi_2, phi_3]


## load data
# data = sio.loadmat('C:\\Users\DELL\Desktop\SAR\mstar_data\mstar_3_cut_onehot.mat')
# data = sio.loadmat('C:\\Users\DELL\Desktop\SAR\SIMU\pairdata_all_half.mat')
# test_data = data['simu_test']
# data = sio.loadmat('C:\\Users\DELL\Desktop\\pgbn_divide_class_500-450-50_phi_simu.mat')
# Phi = data['Phi']
# # data = sio.loadmat('C:\\Users\DELL\Desktop\\whai_divide_angle_200-100-50_theta_simu_loadallphi.mat')
# Theta_t = data['Theta']
# fig = 200

## visualization of theta
# fig4 = plt.figure()
# for i in range(3):
#     ax = fig4.add_subplot(1, 3, i + 1)
#     ax.plot(Theta_t[0, i][:, 100])
#
# fig5 = plt.figure()
#
# #original data
# test_data_i = [np.reshape(test_data[:, 0], [64, 64]),
#                np.reshape(test_data[:, 50], [64, 64]),
#                np.reshape(test_data[:, 100], [64, 64])]
# for i in range(3):
#     ax = fig5.add_subplot(3, 3, i + 1)
#     ax.imshow(test_data_i[i])
#     ax.axis('off')
#
# # sampling
#
# sample = np.random.poisson(np.matmul(Phi[0,0],Theta_t[0,0]))
# sample = [np.reshape(sample[:, 0], [64, 64]),
#           np.reshape(sample[:, 50], [64, 64]),
#           np.reshape(sample[:, 100], [64, 64])]
# for i in range(3):
#     ax = fig5.add_subplot(3, 3, i + 7)
#     ax.imshow(sample[i])
#     ax.axis('off')
#
# # ## mean
# #
# # data_recon = np.matmul(Phi[0,0],Theta_t[0,0])
# # data_recon = [np.reshape(data_recon[:, 1], [64, 64]),
# #               np.reshape(data_recon[:, 588], [64, 64]),
# #               np.reshape(data_recon[:, 783], [64, 64])]
# # for i in range(3):
# #     ax = fig5.add_subplot(3, 3, i + 4)
# #     ax.imshow(data_recon[i])
# #     ax.axis('off')


## visualization of dictionary
index1 = range(100)
index2 = range(49)
index3 = range(49)
# index1 = np.argsort(-np.sum(theta[0], axis=1))

print(index1)
# layer1
dic1 = phi[0][:, index1[0:49]]
fig7 = plt.figure()
for i in range(dic1.shape[1]):
    tmp = dic1[:, i].reshape(28, 28)
    ax = fig7.add_subplot(7, 7, i + 1)
    ax.axis('off')
    ax.set_title(str(index1[i] + 1))
    ax.imshow(tmp)

# index2 = np.argsort(-np.sum(theta[1], axis=1))
# layer2
dic2 = np.matmul(phi[0], phi[1][:, index2[0:49]])
fig8 = plt.figure()
for i in range(dic2.shape[1]):
    tmp = dic2[:, i].reshape(28, 28)
    ax = fig8.add_subplot(7, 7, i + 1)
    ax.axis('off')
    ax.set_title(str(index2[i] + 1))
    ax.imshow(tmp)

# index3 = np.argsort(-np.sum(theta[2], axis=1))

# layer3
dic3 = np.matmul(np.matmul(phi[0], phi[1]), phi[2][:, index3[0:32]])
fig9 = plt.figure()
for i in range(dic3.shape[1]):
    tmp = dic3[:, i].reshape(28, 28)
    ax = fig9.add_subplot(7, 7, i + 1)
    ax.axis('off')
    ax.set_title(str(index3[i] + 1))
    ax.imshow(tmp)

plt.show()