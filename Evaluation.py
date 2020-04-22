import time
import pandas as pd
import numpy as np
import tensorflow as tf


def get_predictions(model, train_set, test_set, rank_at, temp=0.9):
    predictions_df = pd.DataFrame(columns=['user', 'pred_items_ranked', 'true_id'])
    for u in test_set.user_id.unique():
        test_user_seq = np.array(train_set[train_set['user_id'] == u]['item_id'])
        true_items = list(test_set[test_set['user_id'] == u]['item_id'])
        generated_predictions = []

        # Predict
        for item in range(rank_at):  # could be any number of recommended items you want to predict
            predictions = model(test_user_seq.reshape(-1, 1).T)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temp
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            test_user_seq = np.append(test_user_seq, predicted_id).reshape(-1, 1).transpose()

            #         half_test_seq = tf.expand_dims([predicted_id], 0)
            generated_predictions.append(predicted_id)

        predictions_df = predictions_df.append(
            {'user': u, 'pred_items_ranked': generated_predictions, 'true_id': true_items}, ignore_index=True)

    return predictions_df


def rank_predictions(model, test_set, rank_at):
    s = time.time()
    users = test_set.user_id.unique()
    test_user_items = test_set.groupby('user_id')['item_id'].apply(list)
    ranked_df = pd.DataFrame(columns=['pred_items_ranked', 'true_id'], index=users)

    pred_items_ranked = []
    true_items_list = []

    for u in users:
        true_items = []
        for true_item in test_user_items.loc[u]:
            true_items.append(true_item)

        predictions = np.dot(model['p'][u], model['q'].T)
        ids = np.argpartition(predictions, -rank_at)[-rank_at:]
        best_ids = np.argsort(predictions[ids])[::-1]
        best = ids[best_ids]

        pred_items_ranked.append(best)
        true_items_list.append(true_items)

    ranked_df['pred_items_ranked'] = pred_items_ranked
    ranked_df['true_id'] = true_items_list

    print('Ranking time:', round(time.time() - s, 2))

    return ranked_df


def get_metrics(ranked_df, steps, max_rank):
    s = time.time()
    ranks_at = [1] + [i for i in range(steps, max_rank + steps, steps)]
    hitcounts = recs_at = precs_at = []
    metrics = pd.DataFrame(columns=['rank_at', 'hitcounts', 'recall', 'precision'])
    for rank in ranks_at:
        hitcount = 0
        for i, row in ranked_df.iterrows():
            hitcount += len(set(row['true_id']) & set(row['pred_items_ranked'][:rank]))

        prec_at = hitcount / rank / len(ranked_df)
        rec_at = hitcount / len(ranked_df.iloc[0]['true_id']) / len(ranked_df)

        hitcounts.append(hitcount)
        recs_at.append(rec_at)
        precs_at.append(prec_at)

    metrics['rank_at'] = ranks_at
    metrics['hitcounts'] = hitcounts
    metrics['recall'] = recs_at
    metrics['precision'] = precs_at
    print('Obtaining metrics time:', round(time.time() - s, 2))
    return metrics