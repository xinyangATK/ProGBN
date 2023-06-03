import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
real_min = torch.tensor(1e-30)

def log_max(x):
    return torch.log(torch.max(x, real_min.to(x.device)))

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