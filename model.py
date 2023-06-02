import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from utils import *
import numpy as np
import os
# from ../PGBN_tool import PGBN_sampler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GBN_model(nn.Module):
    def __init__(self, args):
        super(GBN_model, self).__init__()

        self.args = args
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(1.0).float()

        self.wei_shape = torch.tensor(1e-1).float()

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.topic_size = args.topic_size
        self.topic_size = [self.vocab_size] + self.topic_size
        self.layer_num = len(self.topic_size) - 1
        self.embed_size = args.embed_size

        self.adj = []
        for layer in range(self.layer_num - 1):
            self.adj.append(torch.from_numpy(args.adj[layer].todense()).cuda())
        self.pos_weight = []
        self.edge_index = []
        for layer in range(self.layer_num - 1):
            pos_weight = (self.adj[layer].shape[0]**2 - self.adj[layer].sum()) / self.adj[layer].sum()
            weight_mask = self.adj[layer].view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0)).cuda()
            weight_tensor[weight_mask] = pos_weight
            self.pos_weight.append(weight_tensor)
            self.edge_index.append(self.adj[layer].nonzero().t())

        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in range(self.layer_num)])

        h_encoder = [DeepConv1D(self.hidden_size, 1, self.vocab_size)]
        for i in range(self.layer_num - 1):
            h_encoder.append(ResConv1D(self.hidden_size, 1, self.hidden_size))
        self.h_encoder = nn.ModuleList(h_encoder)

        shape_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in
                         range(self.layer_num - 1)]
        shape_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.shape_encoder = nn.ModuleList(shape_encoder)

        gam_shape_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in
                         range(self.layer_num - 1)]
        gam_shape_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.gam_shape_encoder = nn.ModuleList(gam_shape_encoder)

        scale_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in range(self.layer_num - 1)]
        scale_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.scale_encoder = nn.ModuleList(scale_encoder)

        topic_embedding = [Get_topic_embedding(self.topic_size[i],self.embed_size) for i in range(self.layer_num + 1)]
        self.topic_embedding = nn.ModuleList(topic_embedding)

        decoder = [Conv1DSoftmaxEtm(self.topic_size[i], self.topic_size[i + 1], self.embed_size) for i in range(self.layer_num)]
        self.decoder = nn.ModuleList(decoder)

        for t in range(self.layer_num - 1, -1, 1):
            self.decoder[t - 1].alphas = self.decoder[t].rho

        self.rho = [0] * self.layer_num
        self.rho_graph_encoder = nn.ModuleList()
        for layer in range(self.layer_num - 1):
            self.rho_graph_encoder.append(GCNConv(in_channels=self.embed_size, out_channels=self.embed_size).cuda())

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min.cuda()))

    def reparameterize(self, Wei_shape_res, Wei_scale, Sample_num = 1):
        # sample one
        eps = torch.cuda.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        theta = torch.max(theta, self.real_min.cuda())
        return torch.mean(theta, dim=0, keepdim=False)

    def reparameterize2(self, Wei_shape_res, Wei_scale, Sample_num=10):
        # sample one
        eps = torch.cuda.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-log_max(1 - eps), torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        theta = torch.max(theta, self.real_min.cuda())
        return torch.mean(theta, dim=0, keepdim=False)

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1))
        return - likelihood / x.shape[1]

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.cuda() * Gam_shape * Wei_shape_res + self.log_max(Wei_shape_res)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma.cuda() + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / Wei_scale.shape[1]

    def _ppl(self, x, X1):
        # x: K1 * N
        # V * N
        X2 = X1 / (X1.sum(0) + real_min)
        ppl = x * torch.log(X2.T + real_min) / -x.sum()
        # ppl = tf.reduce_sum(x * tf.math.log(X2 + real_min)) / tf.reduce_sum(x)
        return ppl.sum().exp()

    def test_ppl(self, x, y):
        x_0 = self.infer(x)

        # _, theta_y, _, _ = self.forward_heart(y)
        ppl = self._ppl(y, x_0)
        # ret_dict.update({"ppl": ppl})
        return ppl


    def inner_product(self, x, dropout=0.):
        # default dropout = 0
        # x = F.dropout(x, dropout, training=self.training)
        x_t = x.permute(1, 0)
        x = x @ x_t
        # out = x.reshape(-1)
        return x

    def rho_decoder(self, x):
        re_adj = self.inner_product(x)
        re_adj = F.sigmoid(re_adj)

        return re_adj

    def forward(self, x):
        coef = 10

        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        gam_shape = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num

        rec_x = [0] * self.layer_num

        loss = [0] * self.layer_num
        likelihood =  [0] * self.layer_num
        KL = [0] * self.layer_num
        graph_lh = [0] * self.layer_num
        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x[0])))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t-1])))

            hidden_list[t] = hidden

        phi_alpha = [0] * self.layer_num
        phi = [0] * self.layer_num

        for t in range(self.layer_num):
            if t == 0:
                phi[t] = torch.softmax(torch.mm(self.topic_embedding[t](), torch.transpose(self.topic_embedding[t + 1](), 0, 1)), dim=0)
                #phi_alpha[t] = phi[t]
                self.rho[t] = self.topic_embedding[0]()
                phi_alpha[t] = torch.softmax(torch.mm(self.rho[0], torch.transpose(self.topic_embedding[t + 1](), 0, 1)), dim=0)  # phi[t]    #torch.mm(self.topic_embedding[0](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
            else:
                phi[t] = torch.softmax(torch.mm(self.topic_embedding[t]().detach(), torch.transpose(self.topic_embedding[t + 1](), 0, 1)), dim=0)
                #phi_alpha[t] = torch.mm(phi_alpha[t-1].detach(), phi[t])
                self.rho[t] = self.rho_graph_encoder[t - 1](self.rho[t - 1].detach(), self.edge_index[t - 1])

                phi_alpha[t] = torch.softmax(torch.mm(self.rho[t],  torch.transpose(self.topic_embedding[t + 1](), 0, 1)), dim=0)  # torch.mm(phi_alpha[t-1].detach(), phi[t])     #torch.mm(self.topic_embedding[0](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))


                # PHI = torch.mm(self.topic_embedding[0](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
                # PHI = torch.softmax(phi_alpha, dim=0)
                # rec_x[t] = torch.mm(phi_alpha, theta[t].view(-1, theta[t].size(-1)))

        for t in range(self.layer_num-1, -1, -1):
            if t == self.layer_num - 1:
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_list[t])),
                                       self.real_min.cuda())      # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())

                gam_shape[t] = torch.max(torch.nn.functional.softplus(self.gam_shape_encoder[t](hidden_list[t])),
                                       self.real_min.cuda())  # k_rec = 1/k
                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_list[t])), self.real_min.cuda())
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))
                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))

                rec_x[t] = torch.mm(phi_alpha[t], theta[t].view(-1, theta[t].size(-1)))
                phi_theta[t] = torch.mm(phi[t], theta[t].view(-1, theta[t].size(-1)))

                # else:
                #     phi = torch.softmax(torch.mm(self.topic_embedding[t](), torch.transpose(self.topic_embedding[t+1](), 0, 1)), dim=0)
                #     phi_alpha = phi
                #     # PHI =  torch.mm(self.topic_embedding[0](), torch.transpose(self.topic_embedding[t+1](), 0, 1))
                #     # PHI = torch.softmax(PHI, dim=0)
                #     # rec_x[t] = torch.mm(PHI, theta[t].view(-1, theta[t].size(-1)))
                # # phi = torch.softmax(phi, dim=0)
                # phi_theta[t] = torch.mm(phi, theta[t].view(-1, theta[t].size(-1)).detach())

                # if t == 0:
                #     phi_theta[t] = self.decoder[t](theta[t], t)
                # else:
                #     phi_theta[t] = self.decoder[t](theta[t].detach(), t)
            else:
                hidden_phitheta = torch.cat((hidden_list[t], phi_theta[t+1].permute(1, 0)), 1)

                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)),
                                       self.real_min.cuda())  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.cuda())
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))

                rec_x[t] = torch.mm(phi_alpha[t], theta[t].view(-1, theta[t].size(-1)))
                phi_theta[t] = torch.mm(phi[t], theta[t].view(-1, theta[t].size(-1)))

                # if t > 0:
                #     PHI = torch.mm(self.topic_embedding[0](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
                #     PHI = torch.softmax(PHI, dim=0)
                #     rec_x[t] = torch.mm(PHI, theta[t].view(-1, theta[t].size(-1)))
                #     phi = torch.mm(self.topic_embedding[t]().detach(), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
                # else:
                #     phi = torch.mm(self.topic_embedding[t](),  torch.transpose(self.topic_embedding[t + 1](), 0, 1))
                #
                # phi = torch.softmax(phi, dim=0)

                # if t > 0:
                #     phi = torch.softmax(torch.mm(self.topic_embedding[t]().detach(),
                #                    torch.transpose(self.topic_embedding[t + 1](), 0, 1)), dim=0)
                #     phi_alpha = torch.mm(phi_alpha, phi)
                #
                #     # PHI = torch.mm(self.topic_embedding[0](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
                #     # PHI = torch.softmax(phi_alpha, dim=0)
                #     rec_x[t] = torch.mm(phi_alpha, theta[t].view(-1, theta[t].size(-1)))
                #
                # else:
                #     phi = torch.softmax(torch.mm(self.topic_embedding[t](), torch.transpose(self.topic_embedding[t+1](), 0, 1)), dim=0)
                #     phi_alpha = phi
                #     # PHI =  torch.mm(self.topic_embedding[0](), torch.transpose(self.topic_embedding[t+1](), 0, 1))
                #     # PHI = torch.softmax(PHI, dim=0)
                #     # rec_x[t] = torch.mm(PHI, theta[t].view(-1, theta[t].size(-1)))
                # # phi = torch.softmax(phi, dim=0)

                # if t == 0:
                #      phi_theta[t] = torch.mm(phi[t], theta[t].view(-1, theta[t].size(-1)))
                # else:
                #      phi_theta[t] = torch.mm(phi[t], theta[t].view(-1, theta[t].size(-1)))
                # if t == 0:
                #     phi_theta[t] = self.decoder[t](theta[t], t)
                # else:
                #     phi_theta[t] = self.decoder[t](theta[t].detach(), t)


        for t in range(self.layer_num):
            if t == 0:
                likelihood[0] = self.compute_loss(x[0].permute(1, 0), phi_theta[t])
                if self.layer_num !=1:
                     KL[0] = self.KL_GamWei(phi_theta[1] , torch.tensor(1.0, dtype=torch.float32).cuda(),
                                              k_rec[0].permute(1, 0), l[0].permute(1, 0))
                else:
                     KL[t] = self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).cuda(),
                                           torch.tensor(1.0, dtype=torch.float32).cuda(),
                                           k_rec[t].permute(1, 0), l[t].permute(1, 0))
                # KL[t] = self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).cuda(),
                #                        torch.tensor(1.0, dtype=torch.float32).cuda(),
                #                        k_rec[t].permute(1, 0), l[t].permute(1, 0))
                graph_lh[0] = torch.tensor(0.).cuda()
                if self.args.pc:
                     loss[0] = coef * (1 - 0.2 * t) * likelihood[t] + KL[t] #1000 * torch.relu(likelihood[0] - 200)  + KL[0]
                else:
                     loss[0] = 1000 * torch.relu(likelihood[0] - 200)  + KL[0] #(1 - 0.2 * t) * likelihood[t] + KL[t] #  1000 * torch.relu(likelihood[0] - 180)  + KL[0]

            elif t == self.layer_num-1:
                KL[t] = self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).cuda(), torch.tensor(1.0, dtype=torch.float32).cuda(),
                                             k_rec[t].permute(1, 0), l[t].permute(1, 0))
                likelihood[t] = self.compute_loss(x[t].permute(1, 0), rec_x[t])  # * (1 - t * 0.2)

                re_adj = self.rho_decoder(self.rho[t])
                graph_lh[t] = F.binary_cross_entropy(re_adj.view(-1), self.adj[t - 1].view(-1), weight=self.pos_weight[t - 1])

                if self.args.pc:
                    loss[t] = coef * (1 - 0.2 * t) *likelihood[t] + KL[t] + 0.005 * graph_lh[t] # 1000 * torch.relu(likelihood[t] - 220) + KL[t] (1 - 0.2 * t) *
                else:
                    loss[t] = 1000 * torch.relu(likelihood[t] - 220) + KL[t] * (1 - 0.2 * t)  # 1000 * torch.relu(likelihood[0] - 180)  + KL[0]

                # loss += self.compute_loss(x.permute(1, 0),  rec_x[t].permute(1, 0))
            else:
                KL[t] = self.KL_GamWei(phi_theta[t + 1], torch.tensor(1.0, dtype=torch.float32).cuda(),
                                         k_rec[t].permute(1, 0), l[t].permute(1, 0))
                likelihood[t] = self.compute_loss(x[t].permute(1, 0), rec_x[t])  # * (1 - t * 0.2)0.5 * (1 - t * 0.1) *

                re_adj = self.rho_decoder(self.rho[t])
                graph_lh[t] = F.binary_cross_entropy(re_adj.view(-1), self.adj[t - 1].view(-1), weight=self.pos_weight[t - 1])

                if self.args.pc:
                    loss[t] = coef * (1 - 0.2 * t) * likelihood[t] + KL[t] + 0.005 * graph_lh[t] #1000 * torch.relu(likelihood[t] - 220) + KL[t](1 - 0.2 * t) *
                else:
                    loss[t] = 1000 * torch.relu(likelihood[t] - 220) + KL[t] * (1 - 0.2 * t) # 1000 * torch.relu(likelihood[0] - 180)  + KL[0]

        return loss, likelihood, KL, graph_lh

    def infer(self, x):

        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num
        loss = 0
        likelihood = 0
        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t-1])))

            hidden_list[t] = hidden

        for t in range(self.layer_num-1, -1, -1):
            if t == self.layer_num - 1:
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_list[t])),
                                       self.real_min.cuda())      # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())
                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_list[t])), self.real_min.cuda())
                l[t] = l_tmp[t]
                theta[t] = l_tmp[t].permute(1, 0)
                # phi_theta[t] = self.decoder[t](theta[t], t)

                if t == 0:
                    phi = torch.mm(self.topic_embedding[t](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
                else:
                    phi = torch.mm(self.topic_embedding[t]().detach(),
                                   torch.transpose(self.topic_embedding[t + 1](), 0, 1))

                phi = torch.softmax(phi, dim=0)
                phi_theta[t] = torch.mm(phi, theta[t].view(-1, theta[t].size(-1)))

            else:
                hidden_phitheta = torch.cat((hidden_list[t], phi_theta[t+1].permute(1, 0)), 1)
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)),
                                       self.real_min.cuda())  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())
                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.cuda())
                l[t] = l_tmp[t]
                theta[t] = l_tmp[t].permute(1, 0)
                # phi_theta[t] = self.decoder[t](theta[t], t)

                if t == 0:
                    phi = torch.mm(self.topic_embedding[t](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
                else:
                    phi = torch.mm(self.topic_embedding[t]().detach(),
                                   torch.transpose(self.topic_embedding[t + 1](), 0, 1))

                phi = torch.softmax(phi, dim=0)
                phi_theta[t] = torch.mm(phi, theta[t].view(-1, theta[t].size(-1)))

        return phi_theta[0]

