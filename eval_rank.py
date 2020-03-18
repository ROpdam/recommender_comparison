import pandas as pd
import numpy as np

def eval_rank(args=[]):
    result = args[0]
    users = args[1]
    items = args[2]
    test_ones = args[3]
    rank_at = args[4]

    pred = pd.DataFrame(index=users, columns=['score', 'ranked_items'])
    real = pd.DataFrame(index=users, columns=['score', 'ranked_items'])

    for u in users:
        user_item_pred_score = []
        user_item_true_value = []
        for i in items:
            user_item_pred_score.append((np.dot(result['p'][u], result['q'][i]), i))  # tuple list
            user_item_true_value.append((test_ones[u, i], i))

        user_item_pred_score.sort(reverse=True)
        user_item_true_value.sort(reverse=True)

        pred_item_ids, pred_item_scores = zip(*user_item_pred_score[:rank_at])
        real_item_ids, real_item_scores = zip(*user_item_true_value[:rank_at])

        pred.loc[u]['ranked_items'] = pred_item_ids
        pred.loc[u]['score'] = pred_item_scores

        real.loc[u]['ranked_items'] = real_item_ids
        real.loc[u]['score'] = real_item_scores

    result_dict = {'pred_rank': pred, 'true_rank': real}
    return result_dict

