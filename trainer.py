import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from progbn import *
from utils.evaluation import *
from utils.save import *

class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)
        self.model = ProGBN(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def log_max(self, x):
        return torch.log(torch.max(x, self.model.real_min.to(x.device)))

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1.))
        return - likelihood / x.shape[1]

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.cuda() * Gam_shape * Wei_shape_res + self.log_max(Wei_shape_res)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma.cuda() + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / Wei_scale.shape[1]

    def forward_backward(self, x):
        phi_theta, theta, k_rec, l = self.model(x[0])
        coef = 10

        rec_x = [0] * self.layer_num
        loss = [0] * self.layer_num
        likelihood = [0] * self.layer_num
        KL = [0] * self.layer_num
        graph_lh = [0] * self.layer_num
        ones_tensor = torch.tensor(1.0, dtype=torch.float32).cuda()

        phi_alpha = [0] * self.layer_num
        for t in range(self.layer_num):
            if t == 0:
                self.model.rho[t] = self.model.topic_embedding[t]()
            else:
                self.model.rho[t] = self.model.rho_graph_encoder[t - 1](self.model.rho[t - 1].detach(), self.model.edge_index[t - 1])

            phi_alpha[t] = torch.softmax(torch.mm(self.model.rho[t],  torch.transpose(self.model.topic_embedding[t + 1](), 0, 1)), dim=0)
            rec_x[t] = torch.mm(phi_alpha[t], theta[t].view(-1, theta[t].size(-1)))

        for t in range(self.layer_num):
            if t == self.layer_num - 1:
                KL[t] = self.KL_GamWei(ones_tensor, ones_tensor, k_rec[t].permute(1, 0), l[t].permute(1, 0))
            else:
                KL[t] = self.KL_GamWei(phi_theta[t + 1], ones_tensor, k_rec[t].permute(1, 0), l[t].permute(1, 0))

            likelihood[t] = self.compute_loss(x[t].permute(1, 0), rec_x[t])
            re_adj = self.model.rho_decoder(self.model.rho[t])

            if t == 0:
                graph_lh[0] = torch.tensor(0.).cuda()
            else:
                graph_lh[t] = F.binary_cross_entropy(re_adj.view(-1), self.model.adj[t - 1].view(-1), weight=self.model.pos_weight[t - 1])

            if self.args.pc:
                loss[t] = coef * (1 - 0.2 * t) * likelihood[t] + KL[t] + 0.005 * graph_lh[t] # 1000 * torch.relu(likelihood[0] - 200)  + KL[0]

        return loss, likelihood, KL, graph_lh


    def train(self, train_data_loader, test_data_loader):

        for epoch in range(self.epochs):

            self.model.cuda()

            for i, (train_data, _) in enumerate(train_data_loader):
                    loss, likelihood, KL, graph_lh = self.forward_backward([torch.tensor(train_data[0], dtype=torch.float).cuda(),  torch.tensor(train_data[1], dtype=torch.float).cuda(), torch.tensor(train_data[0], dtype=torch.float).cuda(),
                                                        torch.tensor(train_data[2],dtype=torch.float).cuda(), torch.tensor(train_data[3], dtype=torch.float).cuda(),
                                                        torch.tensor(train_data[4], dtype=torch.float).cuda()])

                    total_loss = torch.tensor(0.).cuda()
                    for t in range(self.layer_num - 1, -1, -1):  # from layer layer_num-1-step to 0
                        total_loss += loss[t]
                    total_loss.backward()

                    for para in self.model.parameters():
                        flag = torch.isnan(para).any()
                        if flag:
                            continue

                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if epoch % self.args.save_freq == 0:
                for t in range(self.layer_num):
                     print('epoch {}|{}, layer: {}, loss: {}, likelihood: {}, graph lh: {}, KL: {}'.format(epoch, self.epochs,t, loss[t].item(),  likelihood[t].item(), graph_lh[t].item(), KL[t].item()))
                self.vis_txt(self.save_path)

                self.model.eval()
                save_checkpoint({'state_dict': self.model.state_dict(),
                                 'epoch': epoch,
                                 'rho': self.model.rho},
                                self.save_path, True)
                self.test(test_data_loader, epoch, task=self.args.task)

    def _ppl(self, x, X1):
        # x: K1 * N
        # V * N
        X2 = X1 / (X1.sum(0) + real_min)
        ppl = x * torch.log(X2.T + real_min) / -x.sum()
        return ppl.sum().exp()

    def test(self, data_loader, epoch, task='ppl'):
        if task == 'ppl':
            best_ppl = np.inf
            best_epoch = 0
            ppl = self.test_ppl(data_loader)
            if ppl < best_ppl:
                best_ppl = ppl
                best_epoch = epoch + 1
            print('Epoch {}|{}, test_ikelihood: {:.6f}'.format(epoch + 1, self.epochs, ppl))
            print('Best ppl: {:.6f} at epoch {}.'.format(best_ppl, best_epoch))
        else:  # task == 'clustering'
            best_purity = 0.
            best_nmi = 0.
            purity, nmi = self.test_clustering(data_loader)
            if purity > best_purity:
                best_purity = purity
                best_purity_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_path, 'ckpt_best_purity.pth')
                )
            if nmi > best_nmi:
                best_nmi = nmi
                best_nmi_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_path, 'ckpt_best_nmi.pth')
                )
            print('Epoch {}|{}, Purity: {:.6f} NMI: {:.6f}'.format(epoch + 1, self.epochs, purity, nmi))
            print("Best clustering purity: {:.6f} at epoch {}".format(best_purity, best_purity_epoch))
            print("Best clustering nmi: {:.6f} at epoch {}".format(best_nmi, best_nmi_epoch))

    def test_ppl(self, data_loader):
        num_data = len(data_loader)
        ppl_total = 0.
        for i, (x, y) in enumerate(data_loader):
            x = torch.tensor(x[0], dtype=torch.float).cuda()
            y = torch.tensor(y, dtype=torch.float).cuda()

            with torch.no_grad():
                phi_theta, _, _, _ = self.model(x)
                ppl = self._ppl(y, phi_theta[0])
                ppl_total += ppl.item() / num_data

        return ppl_total

    def test_clustering(self, data_loader):
        test_feats = []
        test_labels = []
        for i, (x, y) in enumerate(data_loader):
            for n in range(self.layer_num):
                x[n] = x[n].float().cuda()

            with torch.no_grad():
                _, theta, _, _ = self.model(x[0])
                theta = torch.cat(list(theta_i for theta_i in theta), dim=0)
                # theta = theta.detach().cpu().numpy()
                test_feats.append(standardization(theta.T))
                test_labels.append(y.numpy())

        test_feats = np.concatenate(test_feats, axis=0)
        test_labels = np.concatenate(test_labels)
        purity, nmi = text_clustering(test_feats, test_labels, num_clusters=20)

        return purity, nmi

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.GBN_models.load_state_dict(checkpoint['state_dict'])

    ################  visualization  ###################

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath='phi_output_1', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = phi_layer
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!!')
            assert self.voc != None

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

    def vis_txt(self, path):
        phi = []
        for t in range(self.layer_num):
            w_t = torch.mm(self.model.rho[t], torch.transpose(self.model.topic_embedding[t + 1](), 0, 1))
            phi_t = torch.softmax(w_t, dim=0).cpu().detach().numpy()
            phi.append(phi_t)
        self.vision_phi(phi, outpath=path + '_phi_output')