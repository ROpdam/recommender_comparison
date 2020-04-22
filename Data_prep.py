import numpy as np
import tensorflow as tf

def leave_users_out(full_data, leave_out, seed=1234):
    np.random.seed(seed)
    full_data['index'] = full_data.index
    user_index_df = full_data.groupby('user')['index'].apply(list)
    users = np.random.choice(list(user_index_df.index), leave_out, replace=False)
    users_indices = []

    for user in users:
        users_indices.extend(user_index_df.loc[user])

    sub_set = full_data.loc[users_indices]
    remaining = full_data.drop(users_indices)

    return remaining.drop(columns=['index']), sub_set.drop(columns=['index'])


def leave_last_x_out(full_data, n_users, leave_out=1, already_picked=[], seed=1234):
    """
    Input: data must contain user_id
    Output: full_data = without all last (time order) entries in leave one out set
            leave_one_out_set = data with one user and one item from full_data
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
    Input: df with user and item id, batch size for CFRNN data, val and test perc of users
           number of last items to leave out for val and test set
    Output: full_data = total users and items of the original df,
            Train, validation and test sets
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


def get_x_y_sequences_ordered(dataset, n_unknowns_in_y=1, stats=True):
    user_sequences_x = []
    user_sequences_y = []
    lengths = []
    users = dataset.user_id.unique()
    shortest_u_seq_order = list(df.groupby('user_id')['item_id'].count().sort_values().index)

    for u in shortest_u_seq_order:
        user_item_seq = np.array(dataset[dataset['user_id'] == u]['item_id'])
        user_sequences_x.append(user_item_seq[:-n_unknowns_in_y])
        user_sequences_y.append(user_item_seq[n_unknowns_in_y:])
        lengths.append(len(user_item_seq))

    median = np.median(lengths)

    if stats:
        print('Number of sequences x:', len(user_sequences_x),
              '\nAvg sequence length x:', np.average(lengths),
              '\nStd_dev sequence length x:', np.round(np.std(lengths), 2),
              '\nMedian of sequence length x:', median)

    return user_sequences_x, user_sequences_y, shortest_u_seq_order


def min_padding(sequences, batch_size, max_len):
    padded_sequences = []
    batch = []
    max_batch_seq_len = 0
    for i, seq in enumerate(sequences):
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
