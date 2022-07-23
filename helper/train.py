from helper.utils import *
from helper.metric import *
from torch import nn

def forward(model, i, data, opt):
    inputs, time_seq, loc_seq, lens, a, A_freq, A_time, A_loc, items, mask, targets, cross_s = data.get_batch(i)
    if opt.model == 'RGRec':
        out, l = model.forward(lens, a, A_freq, A_time, A_loc, items, mask, cross_s)
    else:
        out, l = model.forward(lens, a, A_freq, A_time, A_loc, items, mask, cross_s, inputs, time_seq, loc_seq)
    return out, l, targets

def train_test(model, train_data, test_data, logger,opt):
    logger.info('start training:')
    model.train()
    train_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        _, l, targets = forward(model, i, train_data, opt)
        li = torch.gather(l, dim=1, index=targets[:, None])
        loss = -li.sum() / model.batch_size
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        model.optimizer.step()
        train_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            logger.info('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    logger.info('\tLoss:\t%.3f' % train_loss)
    model.scheduler.step()

    logger.info('start predicting: ')
    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    hit, mrr, ndcg = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    with torch.no_grad():
        for idx, i in enumerate(slices):
            out, _, targets = forward(model, i, test_data, opt)
            scores = trans_to_cpu(out).numpy()
            targets = trans_to_cpu(targets).numpy()
            hit, mrr,ndcg = evaluate(scores, targets, hit, mrr, ndcg)
    hit /= test_data.length
    mrr /= test_data.length
    ndcg /= test_data.length
    return hit, mrr, ndcg