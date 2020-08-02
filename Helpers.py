import tensorflow as tf
import numpy as np
from time import time
import multiprocessing as mp        
import progressbar as progressbar
import csv
import os

class TimingCallback(tf.keras.callbacks.Callback):
    """
    Log training time callback to be used in tf.keras.model.fit
    """
    def __init__(self):
      self.logs=[]
    def on_epoch_begin(self,epoch, logs={}):
      self.starttime=time()
    def on_epoch_end(self, epoch, logs={}):
      self.logs.append(time()-self.starttime)

    
def create_user_sample(user_items, train_users, train_items, params, num_processes):
    users_splits = np.array_split(np.array(train_users),num_processes)
    args = []
    for user_split in users_splits:
        args.append((user_items, user_split, train_items, params['num_neg']))
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(create_user_sample_worker, args)

    user_inputs, item_inputs, labels = [], [], []
    for res_epochs in results: 
        user_inputs.extend(res_epochs['u'])
        item_inputs.extend(res_epochs['i'])
        labels.extend(res_epochs['l'])

    return user_inputs, item_inputs, labels


def create_user_sample_worker(user_items, train_users, train_items, num_neg):
    user_inputs, item_inputs, labels = [], [], []
    for user in train_users:
        # All positive items for this user
        u_items = user_items[user]
        
        # Per positive item, sample num_neg negative items
        for u_item in u_items:
            pos_item = np.random.choice(u_items)

            user_inputs.append(user)
            item_inputs.append(pos_item)
            labels.append(1)

            # Add negative item
            for i in range(num_neg):
                neg_item = np.random.choice(train_items)
                while neg_item in u_items:  # neg item j cannot be in the set of pos items of user u
                    neg_item = np.random.choice(train_items)

                user_inputs.append(user)
                item_inputs.append(neg_item)
                labels.append(0)

    return {'u':user_inputs, 'i':item_inputs, 'l':labels}


def create_ncf_samples(params, train_set, store_path, dataset_name):
    num_neg = params['num_neg']
    sample_name = f'{dataset_name}_samples_{num_neg}_neg'
    
    if not os.path.isdir(store_path + '/' + sample_name):
        os.makedirs(store_path + '/' + sample_name)
        print('dir created: ' + store_path + '/' + sample_name)
    
    user_items = train_set.groupby('user_id')['item_id'].apply(list)
    train_users = train_set.user_id.unique()
    train_items = train_set.item_id.unique()
    num_processes = mp.cpu_count()
    
    epochs = params['epochs']
    for epoch in range(epochs):
        print(f'Epoch: {epoch}/{epochs-1}')
        user_inputs, item_inputs, labels = create_user_sample(user_items, train_users,               train_items, params, num_processes)
        samples = [user_inputs, item_inputs, labels]
        
        file = open(f'{store_path}/{sample_name}/{sample_name}_{epoch}.csv', 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(samples)
    
    
    
######################################## NOT NEEDED ##########################################
# class Batch_Generator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, X, y, batch_size=1, shuffle=True):
#         'Initialization'
#         self.X = X
#         self.y = y
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.y)#int(np.floor(len(self.y)/self.batch_size))
#
#     def __getitem__(self, index):
#         return self.__data_generation(index)
#
#     def on_epoch_end(self):
#         'Shuffles indexes after each epoch'
#         self.indexes = np.arange(len(self.y))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, index):
#         return np.stack(self.X[index]), np.stack(self.y[index])