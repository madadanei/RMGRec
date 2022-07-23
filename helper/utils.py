import numpy as np
from math import sin, cos, asin, sqrt
from collections import Counter
import transbigdata as tbd
import datetime
import torch
import random
import os

def load_model():
    pass

def save_model(model, name):
    road = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = road + '/log/' + name + '.pth'
    torch.save(model, path)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def transition(sessions,time_seq, loc_seq,item):
    trans = []
    for s,t,l in zip(sessions,time_seq,loc_seq):
        for i in range(len(s)-1):
            if s[i+1]==0 :
                break
            if s[i] in item or s[i+1] in item:
                tr = (s[i],s[i+1])
                A_t = 1/(time_cmp(t[i], t[i+1])) if time_cmp(t[i], t[i+1]) != 0 else 1
                A_l = 1/(distance_cmp(l[i], l[i+1])) if distance_cmp(l[i], l[i+1])!=0 else 1
                if s[i] in item and s[i+1] in item:
                    p = 1
                elif s[i] in item and s[i+1] not in item:
                    p = 2
                elif s[i] not in item and s[i+1] in item:
                    p = 3
                trans.append([tr, A_t, A_l, p])
    return trans

def cross_correlation(lens, inputs, items, time_seq, loc_seq):
    ll = []
    for i in range(len(lens)):
        ll.append([sum(lens[:i]), sum(lens[:i + 1])])
    aug = []
    num = (items == 0).sum(dim=1).tolist()
    for idx in ll:
        aug.append([])
        for i in range(idx[0] + 1, idx[1]):
            inp, t, l = inputs[idx[0]:i], time_seq[idx[0]:i], loc_seq[idx[0]:i]
            trans = transition(inp, t, l, items[i])
            if len(trans) > num[i]:
                trans = random.sample(trans, num[i])
            aug.append(trans)
    return aug

def cross_process(aug, items, A_freq, A_time, A_loc):
    num = (items == 0).sum(dim=1).tolist()
    for i in range(len(aug)):
        if len(aug[i]) != 0:
            for index, j in enumerate(aug[i]):
                if j[3] == 1:
                    item1 = trans_to_cpu(items[i]).tolist().index(j[0][0])
                    item2 = trans_to_cpu(items[i]).tolist().index(j[0][1])
                    A_freq[i][item1][item2] += 1
                elif j[3] == 2:
                    item1 = trans_to_cpu(items[i]).tolist().index(j[0][0])
                    item2 = -(num[i]-1)
                    num[i] -= 1
                    items[i][item2] = j[0][1]
                    A_time[i][item1][item2] = j[1]
                    A_loc[i][item1][item2] = j[2]
                    A_freq[i][item1][item2] = 1
                else:
                    item1 = -(num[i]-1)
                    num[i] -= 1
                    item2 = trans_to_cpu(items[i]).tolist().index(j[0][1])
                    items[i][item1] = j[0][0]
                    A_time[i][item1][item2] = j[1]
                    A_loc[i][item1][item2] = j[2]
                    A_freq[i][item1][item2] = 1
    A_freq = max_min(A_freq)
    A_time = max_min(A_time)
    A_loc = max_min(A_loc)
    return items, A_freq, A_time, A_loc



class Data():
    def __init__(self, data, shuffle=False):
        p, t, l, y = data_trans(data)
        inputs, mask, len_max = data_mask(p, [0])
        time_seq, _, _ = data_mask(t, [0])
        loc_seq, _, _ = data_mask(l, [0])
        self.inputs = np.asarray(inputs, dtype=object)
        self.time_seq = np.asarray(time_seq, dtype=object)
        self.loc_seq = np.asarray(loc_seq, dtype=object)
        self.mask = np.asarray(mask, dtype=object)
        self.targets = np.asarray(y)
        self.length = len(inputs)  # 有多少个任务
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.time_seq = self.time_seq[shuffled_arg]
            self.loc_seq = self.loc_seq[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_batch(self, i):  # 获取第i个batch
        inputs, time_seq, loc_seq, mask, targets = self.inputs[i], self.time_seq[i], self.loc_seq[i], self.mask[i], self.targets[i]
        items, alias_inputs = [], []
        A, A_freq, A_time, A_loc = [], [], [], []
        lens = [len(i) for i in inputs]  # [2, 8] 如果batch_size = 2
        inputs = [s for ss in inputs for s in ss] # 不区分用户
        time_seq = [s for ss in time_seq for s in ss]
        loc_seq = [s for ss in loc_seq for s in ss]
        mask = [s for ss in mask for s in ss]
        max_n_node = np.max([(len(np.unique(s))) for s in inputs])
        for s, t, loc in zip(inputs, time_seq, loc_seq):
            node = np.unique(s)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            s_A_freq, s_A_time, s_A_loc  = adj_generate_(s, max_n_node, t, loc)
            A_freq.append(s_A_freq)
            A_time.append(s_A_time)
            A_loc.append(s_A_loc)
            alias_inputs.append([np.where(np.unique(s) == i)[0][0] for i in s])
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A_freq = trans_to_cuda(torch.Tensor(np.array(A_freq)).float())  # [50, 7, 7]
        A_time = trans_to_cuda(torch.Tensor(np.array(A_time)).float())  # [50, 7, 7]
        A_loc = trans_to_cuda(torch.Tensor(np.array(A_loc)).float())  # [50, 7, 7]
        mask = trans_to_cuda(torch.Tensor(list(mask)).long())
        targets = trans_to_cuda(torch.Tensor(targets).long())
        cross_s = cross_correlation(lens, inputs, items, time_seq, loc_seq)
        return inputs, time_seq, loc_seq, lens, alias_inputs, A_freq, A_time, A_loc, items, mask, targets, cross_s

def adj_generate(s, max_node):
    node = np.unique(s)
    A = np.zeros((max_node, max_node))
    for i in np.arange(len(s) - 1):   # 是在算出度
        if s[i + 1] == 0:
            break
        u = np.where(node == s[i])[0][0]
        v = np.where(node == s[i + 1])[0][0]
        A[u][v] = 1
    sum_in = np.sum(A, 0)
    sum_in[np.where(sum_in == 0)] = 1
    A_in = np.divide(A, sum_in)
    sum_out = np.sum(A, 1)
    sum_out[np.where(sum_out == 0)] = 1
    A_out = np.divide(A.transpose(), sum_out)
    A = np.concatenate([A_in, A_out]).transpose()
    return A

def adj_generate_(s, max_node, t, loc):
    node = np.unique(s)
    A_freq = np.zeros((max_node, max_node))
    A_time = np.zeros((max_node, max_node))
    A_loc = np.zeros((max_node, max_node))
    for i in np.arange(len(s) - 1):
        if s[i + 1] == 0:
            break
        u = np.where(node == s[i])[0][0]
        v = np.where(node == s[i + 1])[0][0]
        A_freq[u][v] += 1
        A_time[u][v] = 1/(time_cmp(t[i], t[i+1])) if time_cmp(t[i], t[i+1]) != 0 else 1
        A_loc[u][v] = 1/(distance_cmp(loc[i], loc[i+1])) if distance_cmp(loc[i], loc[i+1])!=0 else 1
    return A_freq, A_time, A_loc

def distance_cmp(l1, l2):
    (lon1, lat1), (lon2, lat2) = l1, l2
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    distance= 2 * asin(sqrt(a)) * 6371 * 1000
    distance=round(distance/1000,3)
    return distance

def time_cmp(t1, t2):
    delta_t = (t2 - t1).item().total_seconds()
    return delta_t

def max_min(data):
    return (data - data.min()) / (data.max() - data.min())


def data_mask(data, mask_value):
    data_tmp = [s for ss in data for s in ss]
    lens = [len(s) for s in data_tmp]
    len_max = max(lens)   # 14
    ss_p = []
    mask = []
    for ss in data:
        lens_ss = [len(s) for s in ss]
        ss_ = []
        if type(ss[0][0]) in [int, tuple]:
            ss_ = ss
        else:
            for s in ss:    # 因为要shuffle操作，所以loc_seq和time_seq都要处理
                s_ = [np.datetime64(p) for p in  s]
                ss_.append(s_)
        s_p = [p + mask_value * (len_max - l) for p, l in zip(ss_, lens_ss)]
        s_mask = [[1] * l + [0] * (len_max - l) for l in lens_ss]
        ss_p.append(s_p)
        mask.append(s_mask)
    return ss_p, mask, len_max


def graph_mask(s, node):
    s_A = []
    for i in range(2, len(s) + 1):
        s_t = list(s[:i] + [0] * (len(s) - i))
        s_A.append(adj_generate(s_t, node))
    s_A = trans_to_cuda(torch.Tensor(np.array(s_A)).float())
    return s_A



# 将sessions拆分成poi列表,时间列表,位置列表,目标列表
def data_trans(data):
    sessions_xp = []
    sessions_xt = []
    sessions_xl = []
    sessions_y = []
    for i in data:
        sessions_xp.append(i[0])
        sessions_xt.append(i[1])
        sessions_xl.append(i[2])
        sessions_y.append(i[3][0])
    return sessions_xp, sessions_xt, sessions_xl, sessions_y


def all_node_count(data):
    n, _, _, n1 = data_trans(data)
    n = [i for ii in n for i in ii]
    n = [i for ii in n for i in ii]
    n.extend(n1)
    return len(set(n))

def all_session_count(data):
    n, _, _, _ = data_trans(data)
    return sum([len(i) for i in n])

def data_split(data, ratio):
    split_len = int(ratio*len(data))
    data_train = data[:split_len]
    data_test = data[split_len:]
    return data_train, data_test

def time_idx(t, delta):
    if t == 0:
        return 0
    if type(t) == np.datetime64:
        t = t.astype(datetime.datetime)
    h = int(t.hour / delta)
    t_ = (h + 24 / delta) if t.weekday() in [5, 6] else h
    return t_ + 1

def time_split(transition, delta, p_node):
    time_cluster = {}
    for i in transition:
        a, b = i[0], i[1]
        t = time_idx(a[1], delta)
        if t not in time_cluster:
            time_cluster[t] = [(a[0], b[0])]
        else:
            time_cluster[t].append((a[0], b[0]))
    time_cluster[0] = [(0, 0)]
    time_cluster_count = {}
    for i in time_cluster:
        time_cluster_count[i] = Counter(time_cluster[i])
    return adj_trans(time_cluster_count, p_node)


def region_idx(loc, params):
    if loc == 0:
        return (0, 0)
    lon, lat = loc
    (lonStart, latStart, deltaLon, deltaLat) = list(params.values())[:4]
    loncol = int(((lon - (lonStart - deltaLon / 2)) / deltaLon)) + 1
    latcol = int(((lat - (latStart - deltaLat / 2)) / deltaLat)) + 1
    return (loncol, latcol)

def bounds_get(transition, delta):
    bounds_lon = [i[2][0] for ii in transition for i in ii]
    bounds_lat = [i[2][1] for ii in transition for i in ii]
    lon1, lat1, lon2, lat2 = min(bounds_lon), min(bounds_lat), max(bounds_lon), max(bounds_lat)
    bounds = [lon1, lat1, lon2, lat2]
    params = tbd.area_to_params(bounds, accuracy=delta)
    return params

def region_split(transition, delta, p_node):
    region_cluster = {}
    params = bounds_get(transition, delta)
    for i in transition:
        a, b = i[0], i[1]
        loc = region_idx(a[2], params)
        if loc not in region_cluster:
            region_cluster[loc] = [(a[0], b[0])]
        else:
            region_cluster[loc].append((a[0], b[0]))
    region_cluster_count = {}
    for i in region_cluster:
        region_cluster_count[i] = Counter(region_cluster[i])
    return adj_trans(region_cluster_count, p_node)



def adj_trans(cluster, p_node):
    cluster_A = {}
    cluster_out = {}
    for i in cluster:
        A = np.zeros((p_node, p_node))
        out = np.zeros(p_node)
        for j in cluster[i]:
            A[j[0]][j[1]] = 1
            out[j[0]] = cluster[i][j]
        cluster_A[i] = A
        cluster_out[i] = out
    return cluster_A, cluster_out

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable