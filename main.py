import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from mydataset import *
from model import *
import pickle
from trainer import GBN_trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=200, help="models used.")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=450, help='Number of epochs to train.')
parser.add_argument('--n_updates', type=int, default=1, help='parameter of gbn')
parser.add_argument('--MBratio', type=int, default=100, help='parameter of gbn')
parser.add_argument('--topic_size', type=list, default=[256, 128, 64, 32, 16], help='Number of units in hidden layer 1.') # [512, 256, 128, 64, 32, 16, 8], 32, 16, 12,  8, 4, 2
parser.add_argument('--hidden_size', type=int, default=512, help='Number of units in hidden layer 1.') #, 128, 64, 32, 16
parser.add_argument('--vocab_size', type=int, default=2000, help='Number of vocabulary')
parser.add_argument('--embed_size', type=int, default=100, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--dataset', type=str, default='rcv1', help='name of dataset')
# parser.add_argument('--dataset-dir', type=str, default='./dataset/rcv1v2_bow_8000.pkl', help='type of dataset.')
# parser.add_argument('--dataset', type=str, default='r8', help='name of dataset')
# parser.add_argument('--dataset-dir', type=str, default='./dataset/r8.pkl', help='type of dataset.')
parser.add_argument('--dataset', type=str, default='20ng', help='name of dataset')
parser.add_argument('--dataset-dir', type=str, default='./dataset/20ng.pkl', help='type of dataset.')
parser.add_argument('--output-dir', type=str, default='torch_phi_output_etm_hier_share', help='type of dataset.')
parser.add_argument('--save-path', type=str, default='saves/20ng_vis', help='type of dataset.')
# parser.add_argument('--save-path', type=str, default='saves/r8', help='type of dataset.')
# parser.add_argument('--save-path', type=str, default='saves/rcv1', help='type of dataset.')
parser.add_argument('--word-vector-path', type=str, default='../process_data/20ng_word_embedding.pkl', help='type of dataset.')
parser.add_argument('--pc', type=bool, default=True, help='type of dataset.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# train_loader, vocab_size, voc = get_loader_txt(args.dataset_dir, batch_size=args.batch_size)
# test_loader, vocab_size, voc = get_test_loader_txt(args.dataset_dir, batch_size=args.batch_size)

if args.dataset == '20ng':
    train_loader, vocab_size, voc, adj = get_train_loader_txt_ppl(args.dataset_dir, adj_file='./dataset/20ng_spadj_2468_7.pkl', batch_size=args.batch_size)
elif args.dataset == 'rcv1':
    train_loader, vocab_size, voc, adj = get_train_loader_txt_ppl_rcv1(args.dataset_dir, adj_file='./dataset/rcv1_spadj_2468.pkl', batch_size=args.batch_size)
else:
    train_loader, vocab_size, voc, adj = get_train_loader_txt_ppl_r8(args.dataset_dir, adj_file='./dataset/r8_spadj_final.pkl', batch_size=args.batch_size)


args.vocab_size = vocab_size
args.adj = adj

args.n_updates = len(train_loader) * args.epochs
args.MBratio = len(train_loader)

trainer = GBN_trainer(args,  voc_path=voc)
trainer.train(train_loader, train_loader)
trainer.vis_txt()
