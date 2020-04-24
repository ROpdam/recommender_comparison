import numpy as np
import tensorflow as tf

def leave_users_out(full_data, leave_out, seed=1234):
    """
    Leaves leave_out number of users out of the full dataset
    :param full_data: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param leave_out: number of users to leave out
    :param seed: set seed for picking the same users
    :return: full_data df without leave_out users and the left out subset of leave_out users
    """
    np.random.seed(seed)
    full_data['index'] = full_data.index
    user_index_df = full_data.groupby('user_id')['index'].apply(list) #TODO: Double check change from 'user' to 'user_id'
    users = np.random.choice(list(user_index_df.index), leave_out, replace=False)
    users_indices = []

    for user in users:
        users_indices.extend(user_index_df.loc[user])

    sub_set = full_data.loc[users_indices]
    remaining = full_data.drop(users_indices)

    return remaining.drop(columns=['index']), sub_set.drop(columns=['index'])


def leave_last_x_out(full_data, n_users, leave_out=1, already_picked=[], seed=1234):
    """
    Leaves last leave_out items out for all n_users in full_data
    :param full_data: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param n_users: number of users to leave the last leave_out items out of the full_data
    :param leave_out: number of last item_ids per user
    :param already_picked: If method is used again for the same full_data, provide list of users to not have overlap
    :param seed: set seed for picking the same users
    :return: full_data df without leave_out items per users and the left out subset of leave_out item_id(s)
             per user_id(s)
    """
    np.random.seed(seed)

    full_data['index'] = full_data.index
    user_items_ind = full_data.groupby('user_id')['index'].apply(list)
    users = full_data.user_id.unique()
    leave_out_indices = []
    users_picked = []

    for i in range(len(full_data.user_id.unique())):
        random_user = np.random.choice(users)
        item_indices = user_items_ind[random_user]  # random user's items indices
        if random_user in users_picked or len(item_indices) <= leave_out or random_user in already_picked:
            random_user = np.random.choice(users)
            item_indices = user_items_ind[random_user]  # random user's items indices
        else:
            users_picked.append(random_user)
            leave_out_indices.extend(item_indices[-leave_out:])

        if len(users_picked) == n_users:
            break

    if len(users_picked) < n_users:
        error = 'Cannot pick ' + str(n_users) + ' users with more than ' + str(leave_out) + ' items'
        solution = '\nTry a smaller test and/or validation percentage of the data'
        raise ValueError(error + solution)

    leave_out_set = full_data.loc[leave_out_indices]  # the last items of n_users users with n_item > leave_out
    full_data_leave_one_out = full_data.drop(leave_out_indices)  # drops last items for n_users users

    return full_data_leave_one_out.drop(columns=['index']), leave_out_set.drop(columns=['index'])


def train_val_test_split(df, batch_size, val_perc, test_perc, n_items_val, n_items_test, stats=True):
    """
    Create specific train, validation and test set for recommender systems
    :param df: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param batch_size: used in the LSTM
    :param val_perc: percentage of users to use for validation set
    :param test_perc: percentage of users to use for test set
    :param n_items_val: number of items to leave out of df and put in the validation set per user
    :param n_items_test: number of items to leave out of df and put in the test set per user
    :param stats: print new datasets stats
    :return: total users and total items from df, train set, validation set and test set
    """
    df['item_id'] = df.item.astype('category').cat.codes
    df['user_id'] = df.user.astype('category').cat.codes

    total_users = len(df.user_id.unique())  # Need all users for BPR
    total_items = len(df.item_id.unique())  # Need all items for CFRNN

    users_to_remove = len(df.user_id.unique()) % batch_size  # Batch size compatible for CFRNN
    df_new, deleted_users = leave_users_out(df, users_to_remove)

    test_users = int(test_perc * total_users / batch_size + 1) * batch_size  # Number of users to be used for testing
    test_last_items = n_items_test  # Items to be removed from test users in train set and used in test set

    val_users = int(val_perc * total_users / batch_size + 1) * batch_size
    val_last_items = n_items_val

    train_set, test_set = leave_last_x_out(df_new, test_users, test_last_items)
    test_users_list = test_set.user_id.unique()
    train_set, val_set = leave_last_x_out(train_set, val_users, val_last_items, already_picked=test_users_list)

    if stats:
        print('Total number of items:', total_items)
        print('Total users:', total_users)
        print('Number of train users:', len(train_set.user_id.unique()))
        print('Number of test users:', test_users)
        print('Number of validation users:', val_users, '\n')
        print('Users deleted:', len(deleted_users.user_id.unique()))

    return total_users, total_items, train_set, val_set, test_set


def get_x_y_sequences(dataset, shift=1, ordered=True, stats=True):
    """

    :param dataset: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param shift: by how much should the target (y) sequence be shifted
    :param ordered: should the sequences be ordered by length
    :param stats: print number of sequences, avg length, std_dev, median
    :return: list user_sequences_x, list shifted user_sequences_y and pandas df user_order if order is True,
             else float median
    """
    user_sequences_x = []
    user_sequences_y = []
    lengths = []
    
    if ordered:
        users = list(dataset.groupby('user_id')['item_id'].count().sort_values().index) #ordered, shortest first
    else:
        users = dataset.user_id.unique()

    for u in users:
        user_item_seq = np.array(dataset[dataset['user_id'] == u]['item_id'])
        user_sequences_x.append(user_item_seq[:-shift])
        user_sequences_y.append(user_item_seq[shift:])
        lengths.append(len(user_item_seq))

    median = np.median(lengths)

    if stats:
        print('Number of sequences x:', len(user_sequences_x),
              '\nAvg sequence length x:', np.average(lengths),
              '\nStd_dev sequence length x:', np.round(np.std(lengths), 2),
              '\nMedian of sequence length x:', median)
    
    if ordered: 
        return user_sequences_x, user_sequences_y, users
    else:
        return user_sequences_x, user_sequences_y, median


def min_padding(sequences, batch_size, min_len, max_len):
    """
    Given a list of sequences sorted on length, this function creates batches where each batch is padded (post)
    until the length of the longest sequence in the batch. sequences < min_len will be excluded and > max_len
    will be truncated (pre), NOTE: Batch_Generator needed for use in training of Keras model (see Helpers.py)
    :param sequences: list of sequences ordered by sequence length per user
    :param batch_size: number of sequences per batch
    :param min_len: minimum sequence length for it to be put in a batch
    :param max_len: maximum sequence length to be truncated
    :return: list of padded_sequences as numpy arrays
    """
    padded_sequences = []
    batch = []
    max_batch_seq_len = 0
    for i, seq in enumerate(sequences):
        if len(seq) > min_len:
            batch.append(seq)
            if max_batch_seq_len > max_len:
                max_batch_seq_len = max_len

            elif max_batch_seq_len < len(seq):
                max_batch_seq_len = len(seq)

            if (i + 1) % batch_size == 0:
                padded_sequences.append(
                    tf.keras.preprocessing.sequence.pad_sequences(batch, maxlen=int(max_batch_seq_len), padding='post',
                                                                  truncating='pre'))
                max_batch_seq_len = 0
                batch = []

    return padded_sequences


def standard_padding(sequences, max_length, stats=True):
    """
    Pads (post) sequences up until max_length with zeros
    :param sequences: list of sequences per user
    :param max_length: maximum length to pad sequences to
    :param stats: print number of sequences, acg sequence length, st_dev of sequence length
    :return: tensorflow dataset consisting of the padded sequences
    """
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=int(max_length), padding='post', truncating='pre')
    if stats:
        print('number of sequences:', padded_sequences.shape[0], 
              '\navg sequence length:', np.average([i.shape[0] for i in padded_sequences]),
              '\nstd_dev sequence length:', np.std([i.shape[0] for i in padded_sequences]))
        
    return tf.data.Dataset.from_tensor_slices(padded_sequences)
