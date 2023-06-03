import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.modules import *

class ProGBN(nn.Module):
    def __init__(self, args):
        super(ProGBN, self).__init__()

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
            pos_weight = (self.adj[layer].shape[0] ** 2 - self.adj[layer].sum()) / self.adj[layer].sum()
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

        scale_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in
                         range(self.layer_num - 1)]
        scale_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.scale_encoder = nn.ModuleList(scale_encoder)

        topic_embedding = [Get_topic_embedding(self.topic_size[i], self.embed_size) for i in range(self.layer_num + 1)]
        self.topic_embedding = nn.ModuleList(topic_embedding)

        decoder = [Conv1DSoftmaxEtm(self.topic_size[i], self.topic_size[i + 1], self.embed_size) for i in
                   range(self.layer_num)]
        self.decoder = nn.ModuleList(decoder)

        for t in range(self.layer_num - 1, -1, 1):
            self.decoder[t - 1].alphas = self.decoder[t].rho

        self.rho = [0] * self.layer_num
        self.rho_graph_encoder = nn.ModuleList()
        for layer in range(self.layer_num - 1):
            self.rho_graph_encoder.append(GCNConv(in_channels=self.embed_size, out_channels=self.embed_size).cuda())

    def reparameterize(self, Wei_shape_res, Wei_scale, Sample_num=1):
        # sample one
        eps = torch.cuda.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-log_max(1 - eps), torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        theta = torch.max(theta, self.real_min.cuda())
        return torch.mean(theta, dim=0, keepdim=False)

    def inner_product(self, x, dropout=0.):
        # default dropout = 0
        # x = F.dropout(x, dropout, training=self.training)
        x_t = x.permute(1, 0)
        x = x @ x_t
        return x

    def rho_decoder(self, x):
        re_adj = self.inner_product(x)
        re_adj = F.sigmoid(re_adj)
        return re_adj

    def forward(self, x):

        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t - 1])))
            hidden_list[t] = hidden

        for t in range(self.layer_num - 1, -1, -1):
            if t == self.layer_num - 1:
                hidden_phitheta = hidden_list[t]
            else:
                hidden_phitheta = torch.cat((hidden_list[t], phi_theta[t + 1].permute(1, 0)), 1)
            k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)), self.real_min.cuda())  # k_rec = 1/k
            k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())
            l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.cuda())
            l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))
            theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))

            if t == 0:
                phi = torch.mm(self.topic_embedding[t](), torch.transpose(self.topic_embedding[t + 1](), 0, 1))
            else:
                phi = torch.mm(self.topic_embedding[t]().detach(), torch.transpose(self.topic_embedding[t + 1](), 0, 1))

            phi = torch.softmax(phi, dim=0)
            phi_theta[t] = torch.mm(phi, theta[t].view(-1, theta[t].size(-1)))

        return phi_theta, theta, k_rec, l



