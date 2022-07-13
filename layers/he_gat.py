import torch
from torch import nn
import torch.nn.functional as F
from helper.utils import trans_to_cuda

class he_gat(nn.Module):
    def __init__(self, hidden_size, concat=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb_size = hidden_size
        self.dropout = 0.5
        self.alpha = 0.2
        self.concat = concat

        self.W_i = nn.Parameter(torch.zeros(size=(self.hidden_size, self.emb_size)))
        self.W_c = nn.Parameter(torch.zeros(size=(self.hidden_size, self.emb_size)))
        nn.init.xavier_uniform_(self.W_i.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_c.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def attention_cmp(self, Wh, A_):
        Wh1 = torch.matmul(Wh, self.a[:self.emb_size, :])
        Wh2 = torch.matmul(Wh, self.a[self.emb_size:, :])
        A_ = A_ + trans_to_cuda(torch.eye(A_.shape[-1]))
        e = (Wh1 + Wh2.transpose(1,2)) * A_
        e = self.leakyrelu(e)
        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(A_ > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return h_prime

    def forward(self, inp, A, l):
        Wh_i = torch.matmul(inp, self.W_i)
        Wh = torch.matmul(inp, self.W_c)
        for index,[i,j]  in enumerate(l):
            Wh[index][i:j] = Wh_i[index][i:j]
        h_prime = self.attention_cmp(Wh, A)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
