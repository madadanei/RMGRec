from torch import nn
import torch
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, meta_size, params_size):
        super().__init__()
        self.linear = nn.Linear(meta_size, params_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        return out

class meta_ggsnn(nn.Module):
    def __init__(self, hidden_size, meta_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.meta_size = meta_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.w_generator = MLP(self.meta_size, self.gate_size*(self.input_size+self.hidden_size))
        self.b_generator = MLP(self.meta_size, self.gate_size*2)

    def init_model(self, mk):
        w, b = self.w_generator(mk), self.b_generator(mk)
        w_ih= w[:self.gate_size*self.input_size].reshape(self.gate_size, self.input_size)
        w_hh = w[self.gate_size*self.input_size:].reshape(self.gate_size, self.hidden_size)
        b_ih, b_hh = b[:self.gate_size], b[self.gate_size:]
        w_ih, w_hh, b_ih, b_hh = nn.Parameter(w_ih), nn.Parameter(w_hh), nn.Parameter(b_ih), nn.Parameter(b_hh)
        return w_ih, w_hh, b_ih, b_hh

    def ggnn_cell(self, hidden, A, mk):   
        w_ih, w_hh, b_ih, b_hh = self.init_model(mk)
        input_in = torch.matmul(A[:, :A.shape[0]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, A.shape[0]: 2 * A.shape[0]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 1)  # [2, 6, 200]

        gi = F.linear(inputs, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh) 

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + updategate * (hidden - newgate) 
        return hy

    def forward(self, hidden, A_seq, mk):
        # hidden : [6, 100]
        # A: [6-1, 6, 100]
        # mk: [6-1, 100]
        for i in range(len(A_seq)):
            hidden = self.ggnn_cell(hidden, A_seq[i], mk[i])
        return hidden # [6, 100]
