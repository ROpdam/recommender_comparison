import time
import pandas as pd
import numpy as np
import tensorflow as tf
from Data_prep import standard_padding, get_x_y_sequences
K = tf.keras.backend

# Papers used:
# 1. Devooght, Robin, and Hugues Bersini. "Collaborative filtering with recurrent neural networks." arXiv preprint arXiv:1608.07400 (2016).

def get_metrics(ranked_df, steps=5, max_rank=20, stats=True, ndcg=True):
    """
    Computes hitcount@, recall@ and precision@ for the given ranked_df on each step until the max_rank
    :param ranked_df: pandas df where each row contains, user, a list: pred_items_ranked, a list: true items
    :param steps: after rank@1 the steps to divide max_rank with and calculate the metrics with
    :param max_rank: maximum rank to compute the metrics on
    :param stats: print duration
    :return: pandas df, where each row represents a rank@ value, the columns represent: hitcount@, recall@, precision@
    """
    s = time.time()
    ranks_at = [1] + [i for i in range(steps, max_rank + steps, steps)]
    hitcounts = []
    recs_at = []
    precs_at = []
    ndcgs_at = []
    metrics = pd.DataFrame(columns=['rank_at', 'hitcounts', 'recall', 'precision'])
    for rank in ranks_at:
        hitcount = 0
        ndcg_at = 0
        for i, row in ranked_df.iterrows():
            hitcount += len(set(row['true_id']) & set(row['pred_items_ranked'][:rank]))
            if ndcg:
                ndcg_at += getNDCG(row['pred_items_ranked'][:rank], row['true_id'])

        prec_at = hitcount / rank / len(ranked_df)
        rec_at = hitcount / len(ranked_df.iloc[0]['true_id']) / len(ranked_df)

        hitcounts.append(hitcount)
        recs_at.append(rec_at)
        precs_at.append(prec_at)
        if ndcg:
            ndcgs_at.append(ndcg_at/len(ranked_df))

    metrics['rank_at'] = ranks_at
    metrics['hitcounts'] = hitcounts
    metrics['recall'] = recs_at
    metrics['precision'] = precs_at
    if ndcg:
        metrics['ndcg'] = ndcgs_at
    if stats:
        print('Obtaining metrics time:', round(time.time() - s, 2))

    return metrics

import math
import time
def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0