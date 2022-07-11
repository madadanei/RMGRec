import argparse
import os.path
import pickle
from datetime import datetime, timedelta
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NYC')
parser.add_argument('--sample', action='store_true')
parser.add_argument('--trace_len_min', type=int, default=10)
parser.add_argument('--poi_visit_min', type=int, default=10)
parser.add_argument('--delta_time_min', type=int, default=5)
parser.add_argument('--session_len_min', type=int, default=3)
parser.add_argument('--session_category_min', type=int, default=2)
parser.add_argument('--session_num_min', type=int, default=3)
parser.add_argument('--session_num_max', type=int, default=10)
opt = parser.parse_args()

if opt.dataset == 'NYC':
    dataset = 'NYC.txt'
elif opt.dataset == 'TKY':
    dataset = 'TKY.txt'
else:
    dataset = 'Gowalla.txt'

data = {}   # 读入全部数据  (u: poi, time, loc)
poi = {}    # 读入poi集合  (poi: num)

'''
1.read
'''
with open(dataset, 'r', encoding='ISO-8859-1') as f:
    lines_seen = set()
    for line in f:
        if opt.sample and len(lines_seen) == 100000:
            print('sample loading')
            break
        if line not in lines_seen:
            if dataset == 'Gowalla.txt':
                lines_seen.add(line)
                u, UTC, lat, lon, p = line.strip().split('\t')
                time = datetime.strptime(UTC, "%Y-%m-%dT%H:%M:%SZ")
                loc = (float(lon), float(lat))

            else:
                lines_seen.add(line)
                u, p, _, cat, lat, lon, delta_t, utc = line.strip().split('\t')
                utc = ''.join(utc.strip().split(' +0000'))
                time = datetime.strptime(utc, '%a %b %d %X %Y') + timedelta(minutes=int(delta_t))
                loc = (float(lon), float(lat))

            if u not in data:
                data[u] = [[p, time, loc]]
            else:
                data[u].append([p, time, loc])

            if p not in poi:
                poi[p] = 1

            else:
                poi[p] += 1

            if dataset == 'Gowalla.txt' and len(poi) == 400000:
                break

'''
2.select
'''
user_set = [u for u in data if len(data[u]) >= opt.trace_len_min]
poi_set = [p for p in poi if poi[p] >= opt.poi_visit_min]
data_filter = {}

for u in user_set:
    u_check_ins = sorted([p for p in data[u]], key = lambda p:p[1])  # time order
    sessions = {}

    for i, record in enumerate(u_check_ins):

        p, t, loc = record
        date = t.date()
        s = len(sessions)
        if p not in poi_set:
            continue
        if i == 0 or s == 0:
            sessions[s] = [record]
        else:
            if last_date != date:
                sessions[s] = [record]
            elif (t - last_t).total_seconds() >= opt.delta_time_min * 60:
                sessions[s - 1].append(record)
            else:
                pass
        last_t = t
        last_date = date

    sessions_filter = []
    for s in sessions:
        p_list = [i[0] for i in sessions[s]][:-1]
        if len(sessions[s]) >= opt.session_len_min and len(np.unique(p_list)) >= opt.session_category_min:
            sessions_filter.append(sessions[s])
    if len(sessions_filter) >= opt.session_num_min:
        data_filter[u] = sessions_filter

'''
3.generate tasks
'''
session_min = opt.session_num_min
sessions = []
session_num = 0


for u in data_filter:
    session_max = len(data_filter[u]) if len(data_filter[u]) < opt.session_num_max else opt.session_num_max
    for i in range(session_min, session_max+1):
        for j in range(session_max-i+1):
            x = data_filter[u][j:j+i-1]
            x.append(data_filter[u][j+i-1][:-1])
            session_num += len(x)
            y = data_filter[u][j+i-1][-1]
            sessions.append([x, y, u])

'''
4.reindex
'''
u_idx = {}
for u in data_filter:
    if u not in u_idx:
        u_idx[u] = len(u_idx) + 1

p_idx = {}
sessions_target = []
sequence_target = []
sequence_user_target = []

for s in sessions:
    p = []
    t = []
    l = []
    for s_i in s[0]:
        p.append([j[0] for j in s_i])
        t.append([j[1] for j in s_i])
        l.append([j[2] for j in s_i])
    for i in range(len(p)):
        for j in range(len(p[i])):
            if p[i][j] not in p_idx:
                p_idx[p[i][j]] = len(p_idx) + 1
            p[i][j] = p_idx[p[i][j]]
    if s[1][0] not in p_idx:
        p_idx[s[1][0]] = len(p_idx) + 1
    y = [p_idx[s[1][0]], s[1][1], s[1][2]]  # 要引入变量记录s 不然会报错

    p_ = [i for j in p for i in j]

    s_n = p[-1]
    s_ago = [i for j in p[:-1] for i in j]

    sessions_target.append([p, t, l, y])
    sequence_target.append([p_, y[0]])
    sequence_user_target.append([s_n, s_ago, u_idx[s[2]], y[0]])




'''
5.generate transitions
'''
st_list = list(data_filter.values())
for i in range(len(st_list)):
    for j in range(len(st_list[i])):
        for k in range(len(st_list[i][j])):
            p = st_list[i][j][k][0]
            if p in p_idx:
                st_list[i][j][k][0] = p_idx[p]

st_ = [i for ii in st_list for i in ii]

transition = []
p_idx_list = list(p_idx.values())
for i in range(len(st_)):
    for j in range(len(st_[i]) - 1):
        if st_[i][j][0] in p_idx_list and st_[i][j + 1][0] in p_idx_list:
            transition.append([st_[i][j], st_[i][j + 1]])

print(f"user num: {len(u_idx)}, poi num: {len(p_idx)}, check-in num: {len(lines_seen)}, "
      f"session num:{session_num}, task num:{len(sessions)}, transition num: {len(transition)}")


'''
7.store
'''
if not os.path.exists(opt.dataset):
    os.makedirs(opt.dataset)
session_path = opt.dataset + '/session.pickle'
transition_path = opt.dataset + '/transition.pickle'
sequence_path = opt.dataset + '/sequence.pickle'
sequence_user_path = opt.dataset + '/sequence_user.pickle'

pickle.dump(sessions_target, open(session_path, 'wb'))  # RMGRec 1
pickle.dump(transition, open(transition_path, 'wb'))    # RMGRec 2
pickle.dump(sequence_target, open(sequence_path, 'wb')) # SRGNN
pickle.dump(sequence_user_target, open(sequence_user_path, 'wb'))  # ASGNN