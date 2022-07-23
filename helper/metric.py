import numpy as np

def evaluate(scores, targets, hit, mrr, ndcg):
    for score, target in zip(scores, targets):
        score_index = np.argsort(score)[-1::-1].tolist()
        rate = score_index.index(target) + 1
        if rate <= 5:
            hit[0] += 1
            mrr[0] += 1. / rate
            ndcg[0] += 1./ np.log2(rate+1)
        if rate <= 10:
            hit[1] += 1
            mrr[1] += 1. / rate
            ndcg[1] += 1./ np.log2(rate+1)
        if rate <= 15:
            hit[2] += 1
            mrr[2] += 1. / rate
            ndcg[2] += 1./ np.log2(rate+1)
    return hit, mrr, ndcg

def best(best_result, best_epoch, hit, mrr, ndcg, epoch):
    flag = 0
    if hit[0] >= best_result[0]:
        best_result[0] = hit[0]
        best_epoch[0] = epoch
        flag = 1
    if mrr[0] >= best_result[1]:
        best_result[1] = mrr[0]
        best_epoch[1] = epoch
        flag = 1
    if ndcg[0] >= best_result[2]:
        best_result[2] = ndcg[0]
        best_epoch[2] = epoch
        flag = 1
    if hit[1] >= best_result[3]:
        best_result[3] = hit[1]
        best_epoch[3] = epoch
        flag = 1
    if mrr[1] >= best_result[4]:
        best_result[4] = mrr[1]
        best_epoch[4] = epoch
        flag = 1
    if ndcg[1] >= best_result[5]:
        best_result[5] = ndcg[1]
        best_epoch[5] = epoch
        flag = 1
    if hit[2] >= best_result[6]:
        best_result[6] = hit[2]
        best_epoch[6] = epoch
        flag = 1
    if mrr[2] >= best_result[7]:
        best_result[7] = mrr[2]
        best_epoch[7] = epoch
        flag = 1
    if ndcg[2] >= best_result[8]:
        best_result[8] = ndcg[2]
        best_epoch[8] = epoch
        flag = 1
    return best_result, best_epoch, flag
