from torch import nn
from helper.utils import *

class mk_learner(nn.Module):
    def __init__(self, hidden_size, meta_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.meta_size = int(meta_size/2)
        self.linear = nn.Linear(self.hidden_size, self.meta_size)

    def forward(self, hidden, A_, out_):
        out = torch.true_divide(out_, torch.sum(out_,1).view(-1,1))
        hidden = self.linear(torch.matmul(A_, hidden))
        mk_node = torch.sum(hidden * out.view(out_.shape[0], -1, 1).float(), dim=1)
        return mk_node