import torch
from torch import nn
import torch.nn.functional as F
from helper.utils import *

class time_gat(nn.Module):
    def __init__(self, hidden_size, concat=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb_size = hidden_size
        self.dropout = 0.5
        self.alpha = 0.2
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(self.hidden_size, self.emb_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, A):
        h = torch.mm(inp, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1,N).view(N*N,-1), h.repeat(N,1)], dim=1).view(N,-1,2*self.emb_size)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -1e12 * torch.ones_like(e)
        A = A + trans_to_cuda(torch.eye(A.shape[-1]))
        attention = torch.where(A>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        h_prime = time_weighted(h_prime)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


def time_weighted(h):
    cnt = h.shape[0]
    t = trans_to_cuda(torch.Tensor([1 / 2 ** (cnt - 1 - i) for i in range(cnt)]))
    t /= t.sum()
    h_t = torch.sum(h * t.reshape(-1, 1), dim=0)
    return h_t

def similarity_weighted(h):
    h_c = h[-1]
    sim = trans_to_cuda(torch.Tensor([(torch.matmul(h_i.T, h_c)) for h_i in h]))
    sim /= sim.sum()
    h_sim = torch.sum(h * sim.reshape(-1, 1), dim=0)
    return h_sim


