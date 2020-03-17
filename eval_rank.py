import pandas as pd
import numpy as np

def eval_rank(args=[]):
    result = args[0]
    users = args[1]
    items = args[2]
    test_ones = args[3]

    pred = pd.DataFrame(index=users, columns=['ranked_items', 'score'])
    real = pd.DataFrame(index=users, columns=['ranked_items', 'score'])

    for u in users:
        user_item_pred_score = []
        user_item_true_value = []
        for i in items:
            user_item_pred_score.append((i, np.dot(result['p'][u], result['q'][i])))  # tuple list
            user_item_true_value.append((i, test_ones[u, i]))

        user_item_pred_score.sort(reverse=True)
        user_item_true_value.sort(reverse=True)

        pred_item_ids, pred_item_scores = zip(*user_item_pred_score)
        real_item_ids, real_item_scores = zip(*user_item_true_value)

        pred.loc[u]['ranked_items'] = pred_item_ids
        pred.loc[u]['score'] = pred_item_scores

        real.loc[u]['ranked_items'] = real_item_ids
        real.loc[u]['score'] = real_item_scores

    result_dict = {'pred_rank': pred, 'true_rank': real}
    return result_dict

