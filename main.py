import argparse
from model.RMGRec import *
from model.RGRec import *
from model.RMGRec_S import *
from model.RMGRec_T import *
from model.RMGRec_I import *
from log.logger import *
import pickle
from helper.train import train_test
from helper.metric import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='TKY')
parser.add_argument('--model', default='RMGRec')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--meta_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--train_test_ratio', type=float, default=0.8)
parser.add_argument('--time_split', type=int, default=2)
parser.add_argument('--region_split', type=int, default=10000)
parser.add_argument('--l2', type=float, default=1e-3, help='l2 penalty')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--patience', type=int, default=5)
opt = parser.parse_args()
logger = build_logger(opt.model)
seed_torch(50)


sessions = pickle.load(open('data/' + opt.dataset + '/session.pickle', 'rb'))
transition = pickle.load(open('data/' + opt.dataset+ '/transition.pickle','rb'))
train_data, test_data = data_split(sessions, opt.train_test_ratio)
train_data = Data(train_data, shuffle=True)
test_data = Data(test_data, shuffle=False)

if opt.model != 'RGRec':
    params = bounds_get(transition, opt.region_split)
    p_node = all_node_count(sessions) + 1
    s_node = all_session_count(sessions)
    transition_num = len(transition)
    region_A, region_out = region_split(transition, opt.region_split, p_node)
    time_A, time_out = time_split(transition, opt.time_split, p_node)
    model = trans_to_cuda(globals()[opt.model](opt, p_node, params, time_A, region_A, time_out, region_out))
    print(f"poi num:{p_node-1}")
    print(f"session num:{s_node}")
    print(f"transition:{transition_num}")
    print(f"time_delta num:{len(time_A)}")
    print(f"region_delata num:{len(region_A)}")

else:
    p_node = all_node_count(sessions) + 1
    s_node = all_session_count(sessions)
    model = trans_to_cuda(globals()[opt.model](opt, p_node))
    print(f"poi num:{p_node-1}")
    print(f"session num:{s_node}")

logging_load(logger, opt.model, opt, p_node - 1)
slices = train_data.generate_batch(model.batch_size)
start = time.time()
best_result = [0, 0, 0] * 3
best_epoch = [0, 0, 0] * 3
bad_counter = 0
for epoch in range(opt.epoch):
    logger.info('-------------------------------------------------------')
    logger.info('epoch: '+ str(epoch))
    hit, mrr, ndcg = train_test(model, train_data, test_data,logger,opt)
    best_result, best_epoch, flag = best(best_result, best_epoch, hit, mrr, ndcg, epoch)
    logger.info('HR@5:\t%.4f\tMRR@5:\t%.4f\tnDCG@5:  %.4f  Epoch:\t%d,\t%d,\t%d' % (best_result[0], best_result[1], best_result[2], best_epoch[0], best_epoch[1], best_epoch[2]))
    logger.info('HR@10:\t%.4f\tMRR@10:\t%.4f\tnDCG@10: %.4f  Epoch:\t%d,\t%d,\t%d' % (best_result[3], best_result[4], best_result[5],best_epoch[3], best_epoch[4], best_epoch[5]))
    logger.info('HR@15:\t%.4f\tMRR@15:\t%.4f\tnDCG@15: %.4f  Epoch:\t%d,\t%d,\t%d' % (best_result[6], best_result[7], best_result[8],best_epoch[6], best_epoch[7], best_epoch[8]))
    bad_counter += 1 - flag
    if bad_counter >= opt.patience:
        break
logger.info('-------------------------------------------------------')
end = time.time()
logger.info("Run time: %f s" % (end - start))

