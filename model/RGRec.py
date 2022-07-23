from layers.he_gat import *
from layers.time_gat import *

class RGRec(nn.Module):
    def __init__(self, opt, n_node):
        super(RGRec, self).__init__()
        self.n_node = n_node
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.relation = nn.Embedding(2, self.hidden_size)
        self.he_gat_freq = he_gat(self.hidden_size)
        self.he_gat_t = he_gat(self.hidden_size)
        self.he_gat_loc = he_gat(self.hidden_size)
        self.time_gat = time_gat(self.hidden_size)

        self.Wa = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.Wb = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))

        self.WS = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.WT = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.WF = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.STF = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.gate_size = self.hidden_size
        self.w_h1 = nn.Parameter(torch.zeros(self.gate_size, self.hidden_size))
        self.w_h2 = nn.Parameter(torch.zeros(self.gate_size, self.hidden_size))
        self.b_h1 = nn.Parameter(torch.zeros(self.gate_size))
        self.b_h2 = nn.Parameter(torch.zeros(self.gate_size))

        self.predictor = nn.Linear(self.hidden_size, self.n_node, bias=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step,  gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        for bias in self.parameters():
            bias.data.uniform_(-stdv, stdv)

    def forward(self, lens, a, A_freq, A_time, A_loc, items, mask, cross_s):
        ######  1. inter-session
        cross_in_items, A_freq, A_time, A_loc = cross_process(cross_s, items, A_freq, A_time, A_loc)
        hidden_p = self.embedding(cross_in_items)
        intra = self.relation(trans_to_cuda(torch.LongTensor([0]).squeeze()))
        cross = self.relation(trans_to_cuda(torch.LongTensor([1]).squeeze()))
        nums = trans_to_cpu(torch.count_nonzero(items, dim=1)).tolist()
        inl = [[1, nums[i]+1] if items[i][0]==0 else [0,nums[i]] for i in range(items.shape[0])]
        hidden = torch.matmul(torch.add(hidden_p, cross), self.Wb)
        hidden_i = torch.matmul(torch.add(hidden_p, intra), self.Wb)
        for index,[i,j] in enumerate(inl):
            hidden[index][i:j] = hidden_i[index][i:j]
        hidden_none = torch.matmul(self.embedding(items),self.Wa)

        hidden_p_freq = self.he_gat_freq(hidden, hidden_none, A_freq, inl).unsqueeze(1)
        hidden_p_t = self.he_gat_t(hidden, hidden_none, A_time, inl).unsqueeze(1)
        hidden_p_loc = self.he_gat_loc(hidden, hidden_none, A_loc, inl).unsqueeze(1)
        hidden_p_stf = (self.WS(hidden_p_loc) + self.WT(hidden_p_t) + self.WF(hidden_p_freq)).squeeze(1)
        hidden_p = self.STF(hidden_p_stf)

        get = lambda i: hidden_p[i][a[i]]
        s = torch.stack([get(i) for i in torch.arange(len(a)).long()])
        avg_mask = torch.true_divide(mask, torch.sum(mask,1).view(-1,1))
        s = torch.sum(s * avg_mask.view(mask.shape[0], -1, 1).float(), dim=1)

        ######  2. cross-session
        s_t, s_sim = [], []
        for idx, l in enumerate(lens):
            i, j = sum(lens[:idx]), sum(lens[:idx+1])
            hidden_s = s[i:j]
            A_s = trans_to_cuda(torch.ones(l, l))
            s_t.append(self.time_gat(hidden_s, A_s))
            s_sim.append(similarity_weighted(hidden_s))
        s_t  = torch.stack(s_t, 0)
        s_sim = torch.stack(s_sim, 0)

        h_t =  torch.matmul(s_t, self.w_h1)
        h_sim = torch.matmul(s_sim, self.w_h2)
        gate = torch.sigmoid(h_t + h_sim + self.b_h1)
        s_out = gate * s_t + (1 - gate) * s_sim

        out = self.predictor(s_out)
        l = F.log_softmax(out, dim=-1)
        return out, l
