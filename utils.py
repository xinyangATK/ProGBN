import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F

real_min = torch.tensor(1e-30)

def log_max(x):
    return torch.log(torch.max(x, real_min.cuda()))

def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):
    eulergamma = torch.tensor(0.5772, dtype=torch.float32)

    part1 = eulergamma.cuda() * (1 - 1 / Wei_shape) + log_max(
        Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)

    part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma.cuda() / Wei_shape)

    part3 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))

    KL = part1 + part2 + part3
    return KL


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf).cuda()
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf).cuda())
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class DeepConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(DeepConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf).cuda())

            w2 = torch.empty(nf, nf).cuda()
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf).cuda())

        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            rx = x
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = x + rx
        else:
            raise NotImplementedError
        return x


class ResConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(ResConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf).cuda())

            w2 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf).cuda())
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            rx = x
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = rx + x
        else:
            raise NotImplementedError
        return x


class Conv1DSoftmax(nn.Module):
    def __init__(self, voc_size, topic_size):
        super(Conv1DSoftmax, self).__init__()

        w = torch.empty(voc_size, topic_size).cuda()
        nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)

    def forward(self, x):
        w = torch.softmax(self.w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

class Get_topic_embedding(nn.Module):
    def __init__(self, topic_size, emb_size):
        super(Get_topic_embedding, self).__init__()
        self.topic_size = topic_size
        self.emb_size = emb_size

        w1 = torch.empty(self.topic_size, self.emb_size).cuda()
        nn.init.normal_(w1, std=0.02)
        self.rho = Parameter(w1)

    def forward(self):
        return self.rho

class Conv1DSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size, last_layer=None):
        super(Conv1DSoftmaxEtm, self).__init__()
        self.voc_size = voc_size
        self.topic_size = topic_size
        self.emb_size = emb_size

        if last_layer is None:
            w1 = torch.empty(self.voc_size, self.emb_size).cuda()
            nn.init.normal_(w1, std=0.02)
            self.rho = Parameter(w1)
        else:
            w1 = torch.empty(self.voc_size, self.emb_size).cuda()
            nn.init.normal_(w1, std=0.02)
            # self.rho = last_layer.alphas
            self.rho = Parameter(w1)

        w2 = torch.empty(self.topic_size, self.emb_size).cuda()
        nn.init.normal_(w2, std=0.02)
        self.alphas = Parameter(w2)

    def forward(self, x, t):
        if t == 4:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        elif t==0:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        else:
            w = torch.mm(self.rho, torch.transpose(self.alphas.detach(), 0, 1))

        w = torch.softmax(w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

import itertools
import math

class GaussSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size):
        super(GaussSoftmaxEtm, self).__init__()
        self.vocab_size = voc_size
        self.topic_size = topic_size
        self.embed_dim = emb_size
        self.sigma_min = 0.1
        self.sigma_max = 10.0
        self.C = 2.0

        self.w = 0

        # Model
        self.mu = nn.Embedding(self.vocab_size, self.embed_dim)
        self.log_sigma = nn.Embedding(self.vocab_size, self.embed_dim)

        self.mu_c = nn.Embedding(self.topic_size, self.embed_dim)
        self.log_sigma_c = nn.Embedding(self.topic_size, self.embed_dim)

    def el_energy(self, mu_i, mu_j, sigma_i, sigma_j):
        """
        :param mu_i: mu of word i: [batch, embed]
        :param mu_j: mu of word j: [batch, embed]
        :param sigma_i: sigma of word i: [batch, embed]
        :param sigma_j: sigma of word j: [batch, embed]
        :return: the energy function between the two batchs of  data: [batch]
        """

        # assert mu_i.size()[0] == mu_j.size()[0]

        det_fac = torch.sum(torch.log(sigma_i + sigma_j), 1)
        diff_mu = torch.sum((mu_i - mu_j) ** 2 / (sigma_j + sigma_i), 1)
        return -0.5 * (det_fac + diff_mu + self.embed_dim * math.log(2 * math.pi))

    def forward(self, x, t):
        for p in itertools.chain(self.log_sigma.parameters(),
                                 self.log_sigma_c.parameters()):
            p.data.clamp_(math.log(self.sigma_min), math.log(self.sigma_max))

        for p in itertools.chain(self.mu.parameters(),
                                 self.mu_c.parameters()):
            p.data.clamp_(-math.sqrt(self.C), math.sqrt(self.C))

        w = torch.zeros((self.vocab_size, self.topic_size)).cuda()
        for i in range(self.topic_size):
            log_el_energy = self.el_energy(self.mu.weight, self.mu_c.weight[i, :], torch.exp(self.log_sigma.weight), torch.exp(self.log_sigma_c.weight[i, :]))
            w[:, i] = torch.softmax(log_el_energy, dim=0)

        self.w = w
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x



def variable_para(shape, device='cuda'):
    w = torch.empty(shape, device=device)
    nn.init.normal_(w, std=0.02)
    return torch.tensor(w, requires_grad=True)


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


