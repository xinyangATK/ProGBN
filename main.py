from mydataset import *
from model import *
from utils.data_util import get_data_loader
from trainer import GBN_trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='clustering', help='task')
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
parser.add_argument('--dataset', type=str, default='20ng', help='name of dataset')
parser.add_argument('--dataset-dir', type=str, default='./dataset/20ng.pkl', help='type of dataset.')
parser.add_argument('--adj_path', type=str, default='./dataset/20ng_spadj.pkl', help='path of adj')
parser.add_argument('--save-path', type=str, default='saves/20ng_vis', help='type of dataset.')
parser.add_argument('--save_freq', type=int, default=5, help='freq')
parser.add_argument('--pc', type=bool, default=True, help='type of dataset.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Load Dataset: {}".format(args.dataset))
if args.task == 'ppl':
    if args.dataset == '20ng':
        train_loader, vocab_size, voc, adj = get_train_loader_txt_ppl(args.dataset_dir, adj_file='dataset/20ng_spadj.pkl', batch_size=args.batch_size)
    elif args.dataset == 'rcv1':
        train_loader, vocab_size, voc, adj = get_train_loader_txt_ppl_rcv1(args.dataset_dir, adj_file='./dataset/rcv1_spadj.pkl', batch_size=args.batch_size)
    else:
        train_loader, vocab_size, voc, adj = get_train_loader_txt_ppl_r8(args.dataset_dir, adj_file='./dataset/r8_spadj.pkl', batch_size=args.batch_size)
else:
    train_loader, voc, adj = get_data_loader('20ng', args.dataset_dir, args.adj_path,  'train', args.batch_size)
    test_loader = get_data_loader('20ng', args.dataset_dir, args.adj_path, 'test', args.batch_size, shuffle=False, drop_last=False)

args.vocab_size = len(voc)
args.adj = adj

print("Build Model with ProGBN")
trainer = GBN_trainer(args,  voc_path=voc)
print("=======================  Start Training  =======================")
trainer.train(train_loader, train_loader)
