import time
import pandas as pd
import numpy as np
import tensorflow as tf
from Data_prep import standard_padding
K = tf.keras.backend

# Papers used:
# 1. Devooght, Robin, and Hugues Bersini. "Collaborative filtering with recurrent neural networks." arXiv preprint arXiv:1608.07400 (2016).

def get_predictions(model, data, left_out, batch_size, max_seq_len, pad_value, rank_at):
    """
    Uses a Keras model with batch size set to number of test users to predict the rest of the sequences from the data per user.
    Finally creates predictions_df where each row represents user, a list pred_items_ranked and a list containing true_ids
    from the left_out df
    :param model: Keras RNN model with batch size set to batch_size (batch_size==number of test users)
    :param data: Test or Validation set (pandas)
    :param left_out: left out items (pandas)
    :param batch_size: batch_size==number of test users
    :param max_seq_len: Sequences will be padded to this value as this is also done while training
    :param pad_value: pad_value==total_items (as done while training)
    :param rank_at: maximum number of predictions to make
    :return: pandas df where each row represents a user, the columns represent: pred_items_ranked at rank_at,
             true_id extracted from test_set (as input for Evaluation.get_metrics
    """
    user_sequences_series = data.groupby('user_id')['item_id'].apply(list)
    left_out_items = left_out.groupby('user_id')['item_id'].apply(list)

    list_sequences = []
    true_ids = []
    for i, user_sequence in enumerate(user_sequences_series):
        list_sequences.append(user_sequence)
        true_ids.append(left_out_items.iloc[i])

    user_sequences = standard_padding(list_sequences, max_seq_len, pad_value=pad_value, eval=True, stats=False)
    final_preds = np.zeros((user_sequences.shape[0], rank_at))

    for i in range(rank_at):
        predictions = model.predict(user_sequences, batch_size=batch_size)
        for u_index, prediction in enumerate(predictions):
            padding_values = np.where(user_sequences[u_index] == pad_value)[0]
            pred_item_id = np.argmax(prediction)
            final_preds[u_index][i] = pred_item_id

            if padding_values.shape[0] > 0:
                first_pad_value = np.min(padding_values)
                user_sequences[u_index][first_pad_value] = pred_item_id
            else:
                new_user_sequence = np.append(user_sequences[u_index], pred_item_id)[1:]
                user_sequences[u_index] = new_user_sequence

    predictions_df = pd.DataFrame(list(zip(user_sequences_series.index, final_preds, true_ids)),
                                  columns=['user', 'pred_items_ranked', 'true_id'])

    return predictions_df


def rank_predictions(model, test_set, rank_at, stats=True):
    """
    The provided MF model is used to obtain values for all items per user from the test set
    and rank at rank_at per user, finally the true items are put together in the ranked_df result
    :param model: dict containing the p (user) and q (item) factors from a matrix factorisation model
    :param test_set: pandas df containing: user_id, last item_id(s) per user, sorted on datetime per user
    :param rank_at: maximum of top ranked items per user
    :param stats: print duration
    :return: pandas df, where each row represents a user, the columns represent: pred_items_ranked at rank_at,
             true_id extracted from test_set
    """
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
    if stats:
        print('Ranking time:', round(time.time() - s, 2))

    return ranked_df


def get_metrics(ranked_df, steps, max_rank, stats=True):
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
    if stats:
        print('Obtaining metrics time:', round(time.time() - s, 2))

    return metrics

def recall_metric(total_items):
    """

    :param labels:
    :param logits:
    :return:
    """
    def recall(labels, logits):
        labels = K.one_hot(tf.dtypes.cast(labels, tf.int32), total_items)
        labels = K.ones_like(labels)
        true_positives = K.sum(K.round(K.clip(labels * logits, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(labels, 0, 1)))
        r = true_positives / (possible_positives + K.epsilon())
        return r
    return recall


def create_diversity_bias(train_set, total_items, delta):
    """
    Pre-calculates the diversity bias needed in
    :param train_set:
    :param total_items:
    :param delta:
    :return:
    """
    item_id_bins = np.zeros((1,total_items), np.float32)
    item_counts = train_set.groupby('item_id')['user_id'].count().sort_values(ascending=False)
    bins = np.logspace(np.log10(item_counts.max()), np.log10(1), 11)
    item_counts.index, np.digitize([item_counts],bins)

    for item_id, count  in zip(item_counts.index, list(item_counts)):
        item_bin = np.digitize([count],bins)
        item_id_bins[0,item_id] = item_bin

    diversity_biases = tf.Variable(np.exp(item_id_bins[0] * -delta))
    return diversity_biases


def diversity_bias_loss(db, total_items):
    """
    Calculates Categorical Cross Entropy Loss divided by the diversity bias as defined in Paper 1
    :param db: precalculated diversity bias per item_id
    :return: categorical cross entropy loss function adjusted by the diversity bias
    """
    def loss(labels, logits):
        labels = tf.dtypes.cast(labels, tf.int32)
        oh_labels = K.one_hot(labels, total_items)
        standard_loss = tf.keras.losses.categorical_crossentropy(oh_labels, logits, from_logits=True)
        label_weights = tf.gather(db, labels, axis=0)
        db_loss = tf.math.multiply(standard_loss, label_weights)
        return db_loss
    return loss


def cce_loss(total_items):
    """
    Calculates Categorical Crossentropy Loss over the one hot encoded labels with the logits
    :param total_items: Maximum item_id for one hot encoding
    :return: categorical cross entropy loss function
    """
    def loss(labels, logits):
        oh_labels = K.one_hot(tf.dtypes.cast(labels, tf.int32), total_items)
        return tf.keras.losses.categorical_crossentropy(oh_labels, logits, from_logits=True)
    return loss


############################################# NOT NEEDED ANYMORE ######################################################
# def get_predictions(model, test_set, test_left_out_items, rank_at):
#     """
#     Uses a Keras model with batch size set to 1 to predict the rest of the sequences from the train_set per user.
#     Finally creates predictions_df where each row represents user, a list pred_items_ranked and a list containing true_ids
#     from the test_set
#     :param model: Keras RNN model with batch size set to 1
#     :param test_set: pandas df containing: user_id, last item_id(s) per user, without their last (chron) item
#     :param test_left_out_items: pandas df of the last (chron) item of every user in test_test
#     :param rank_at: maximum of top ranked items per user
#     :param temp: temperature, 1 means no deviation from model prediction
#     :return: pandas df where each row represents a user, the columns represent: pred_items_ranked at rank_at,
#              true_id extracted from test_set
#     """
#     user_sequences = test_set.groupby('user_id')['item_id'].apply(list)
#     user_true_items = test_left_out_items.groupby('user_id')['item_id'].apply(list)
#     users = user_true_items.index
#     all_predictions = []
#     all_true_items = []
#     from progressbar import progressbar
#     pBar = progressbar.ProgressBar()
#
#     for user in users:#pBar(users):
#         predictions = []
#         user_seq = user_sequences[user]
#         for i in range(rank_at):
#             pred_item_id = model.predict_classes(np.array([user_seq,]), batch_size=1)[0]
#             user_seq.append(pred_item_id)
#             predictions.append(pred_item_id)
#
#         all_true_items.append(user_true_items[user])
#         all_predictions.append(predictions)
#
#     predictions_df = pd.DataFrame(list(zip(user_sequences.index, all_predictions, all_true_items)),
#                               columns=['user', 'pred_items_ranked', 'true_id'])
#
#     return predictions_df
