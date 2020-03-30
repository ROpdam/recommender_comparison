import pandas as pd
import numpy as np

def eval_rank(args=[]):
    model = args[0]
    users = args[1]
    items = args[2]
    test_ones = args[3]
    rank_at = args[4]

    result_df = pd.DataFrame(index=users, columns=['scores_ranked', 'pred_items_ranked', 'true_id'])

    for u in users:
        user_item_pred_score = []
        true_items = []
        for true_item in np.nonzero(test_ones[u].toarray()[0] > 0):
            true_items.append(true_item)
            
        for i in items:
            prediction = model['b'][i] + np.dot(model['p'][u], model['q'][i].T)
            user_item_pred_score.append((prediction, i))  # tuple list

        user_item_pred_score.sort(reverse=True) #np.argpartition
        pred_item_scores, pred_item_ids = zip(*user_item_pred_score[:rank_at])

        result_df.loc[u]['scores_ranked'] = pred_item_scores
        result_df.loc[u]['pred_items_ranked'] = pred_item_ids
        result_df.loc[u]['true_id'] = true_items

    return result_df

