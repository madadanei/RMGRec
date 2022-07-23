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

        self.a_i = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)))
        self.a_c = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)))
        nn.init.xavier_uniform_(self.a_i.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_c.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, inp_none, A, l):
        Wh1_i = torch.matmul(inp_none, self.a_i[:self.emb_size, :])  # [681, 19, 1]
        Wh2_i = torch.matmul(inp, self.a_i[self.emb_size:, :])
        Wh1 = torch.matmul(inp, self.a_c[:self.emb_size, :])
        Wh2 = torch.matmul(inp, self.a_c[self.emb_size:, :])
        for index,[i,j]  in enumerate(l):
            Wh1[index][i:j] = Wh1_i[index][i:j]
            Wh2[index][i:j] = Wh2_i[index][i:j]
        A_ = A
        # A_ = A + trans_to_cuda(torch.eye(A.shape[-1]))
        e = (Wh1 + Wh2.transpose(1,2)) * A_
        e = self.leakyrelu(e)
        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(A_ > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
