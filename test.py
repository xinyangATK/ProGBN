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
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--n_updates', type=int, default=100, help='parameter of gbn')
parser.add_argument('--MBratio', type=int, default=100, help='parameter of gbn')
parser.add_argument('--topic_size', type=list, default=[512, 256, 128, 64, 32, 16, 8], help='Number of units in hidden layer 1.')
parser.add_argument('--hidden_size', type=list, default=[512, 256, 128, 64, 32, 16, 8], help='Number of units in hidden layer 1.')
parser.add_argument('--vocab_size', type=int, default=2000, help='Number of vocabulary')
parser.add_argument('--embed_size', type=int, default=50, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-dir', type=str, default='./dataset/20ng.pkl', help='type of dataset.')
parser.add_argument('--output-dir', type=str, default='torch_phi_output_etm_hier_share', help='type of dataset.')
parser.add_argument('--save-path', type=str, default='saves/gbn_model_weibull_etm_share_50_7_kl_0.1', help='type of dataset.')
parser.add_argument('--word-vector-path', type=str, default='../process_data/20ng_word_embedding.pkl', help='type of dataset.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, vocab_size, voc = get_loader_txt(args.dataset_dir, batch_size=args.batch_size)
args.vocab_size = vocab_size

args.n_updates = len(train_loader) * args.epochs
args.MBratio = len(train_loader)

trainer = GBN_trainer(args, voc_path=voc)
# trainer.test(data_loader=train_loader)
trainer.vis_txt()

# train_log = 0
# val_log = 0
# for epoch in range(args.epochs):
#     # train_bar = tqdm(iterable=train_loader, desc="DAT Training Epoch [{}/{}]".format(epoch, n_epoch))
#     model.train()
#     loss_t = 0.0
#     likelihood_t = 0.0
#     lb_t = 0.0
#     num_data = len(train_loader)
#     for i, (train_data, train_label) in enumerate(train_loader):
#
#         train_data = torch.tensor(train_data, dtype=torch.float).cuda()
#         train_label = torch.tensor(train_label, dtype=torch.long).cuda()
#         theta, loss, likelihood, lb = model(train_data)
#
#         for para in model.parameters():
#             flag = torch.sum(torch.isnan(para))
#
#         if (flag == 0) :
#             # if epoch < 10:
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
#                 model_opt.step()
#                 model_opt.zero_grad()
#                 loss_t += loss.item()/num_data
#                 likelihood_t += likelihood.item()/num_data
#                 lb_t += lb/num_data
#             # else:
#             #     loss.backward()
#             #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
#             #     model_opt_2.step()
#             #     model_opt_2.zero_grad()
#             #     loss_t += loss.item()
#             #     likelihood_t += likelihood.item()
#             #     lb_t += lb
#
#
#     info = {
#         'loss': -loss_t,
#         'likelihood': likelihood_t,
#         'lb': lb_t
#     }
#     state = loss_t
#     is_best = False
#     if state < checkpoint['best']:
#         checkpoint['best'] = state
#         is_best = True
#     save_checkpoint({
#         'state_dict': model.state_dict(),
#         'best': state,
#         'epoch': epoch
#     }, args.save_path,
#         is_best)
#
#     # for tag,value in info.items():
#     #     logger.scalar_summary(tag,value,epoch)
#
#     if epoch % 1 == 0:
#         print('epoch {}|{}, loss: {}, likelihood: {}, lb: {}'.format(epoch, args.epochs, -loss_t, likelihood_t, lb_t))
#
# print('pre_train is doneeeeeeeeee')
