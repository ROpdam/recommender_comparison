import pandas as pd
import numpy as np

def eval_rank_bpr(args=[]):
    model = args[0]
    users = args[1]
    items = args[2]
    test_user_items = args[3]
    rank_at = args[4]

    result_df = pd.DataFrame(index=users, columns=['pred_items_ranked', 'true_id'])

    for u in users:
        user_item_pred_score = []
        true_items = []
        for true_item in test_user_items.loc[u]:
            true_items.append(true_item)
            
        predictions = np.dot(model['p'][u], model['q'].T)
        ids = np.argpartition(predictions, -rank_at)[-rank_at:]
        best_ids = np.argsort(predictions[ids])[::-1]
        best = ids[best_ids]

        result_df.loc[u]['pred_items_ranked'] = best
        result_df.loc[u]['true_id'] = true_items

    return result_df

