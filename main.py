import numpy as np
from utils.dataset import *
from progbn import *
from trainer import GBN_trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='ppl', help='task clustering')
parser.add_argument('--batch-size', type=int, default=200, help="models used.")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--n_updates', type=int, default=1, help='parameter of gbn')
parser.add_argument('--MBratio', type=int, default=100, help='parameter of gbn')
parser.add_argument('--topic_size', type=list, default=[256, 128, 64, 32, 16], help='Number of units in hidden layer 1.')
parser.add_argument('--hidden_size', type=int, default=512, help='Number of units in hidden layer 1.')
parser.add_argument('--vocab_size', type=int, default=2000, help='Number of vocabulary')
parser.add_argument('--embed_size', type=int, default=100, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='20ng', help='name of data')
parser.add_argument('--data-dir', type=str, default='./data/20ng/20ng.pkl', help='type of data.')
parser.add_argument('--adj_path', type=str, default='./data/20ng/20ng_adj.pkl', help='path of adj')
parser.add_argument('--save-path', type=str, default='saves/20ng', help='type of data.')
parser.add_argument('--save_freq', type=int, default=5, help='freq')
parser.add_argument('--pc', type=bool, default=True, help='type of data.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Load Dataset: {}".format(args.dataset))
if args.task == 'ppl':
    #### args.data in ['20ng', 'rcv1', 'r8']
    if args.dataset == '20ng':
        train_loader, vocab_size, voc, adj = get_loader_ppl_20ng(args.dataset_dir, adj_file=args.adj_path, batch_size=args.batch_size)
    elif args.dataset == 'rcv1':
        train_loader, vocab_size, voc, adj = get_loader_ppl_rcv1(args.dataset_dir, adj_file=args.adj_path, batch_size=args.batch_size)
    else:
        train_loader, vocab_size, voc, adj = get_loader_ppl_r8(args.dataset_dir, adj_file=args.adj_path, batch_size=args.batch_size)
else:
    #### args.data in ['20ng', 'r8', 'tmn']
    train_loader, voc, adj = get_loader_clustering(args.dataset, args.dataset_dir, args.adj_path,  'train', args.batch_size)
    test_loader = get_loader_clustering(args.dataset, args.dataset_dir, args.adj_path, 'test', args.batch_size, shuffle=False, drop_last=False)

args.vocab_size = len(voc)
args.adj = adj

print("Build Model with ProGBN")
trainer = GBN_trainer(args,  voc_path=voc)
print("=======================  Start Training  =======================")
trainer.train(train_loader, train_loader)
