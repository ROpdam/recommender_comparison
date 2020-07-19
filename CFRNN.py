import numpy as np
import time
import math
import pandas as pd
import random
import os
import progressbar
import tensorflow as tf
import matplotlib.pyplot as plt
from Data_prep import standard_padding, get_x_y_sequences
from Helpers import TimingCallback
K = tf.keras.backend
# Papers used:
# 1. Devooght, Robin, and Hugues Bersini. "Collaborative filtering with recurrent neural networks." arXiv preprint arXiv:1608.07400 (2016).

class CFRNN:

    def __init__(self, total_users, total_items, params):
        self.total_users = total_users
        self.total_items = total_items
        self.params = params
        self.train_time = params['train_time']
        self.epochs = params['epochs']
        self.batch_size = params['BATCH_SIZE']
        self.learning_rate = params['learning_rate']
        self.delta = params['delta']
        self.max_seq_len = params['max_seq_len']
        self.embedding_dim = params['embedding_dim']
        self.rnn_units = params['rnn_units']
        self.ckpt_dir = params['ckpt_dir']
        self.pad_value = params['pad_value']
        self.test_users = params['test_users']
        self.val_users = params['val_users']
        self.model = []
        self.history = {}
        self.diversity_bias = []
     
    
    def build_model(self, ckpt_dir='', return_sequences=True, initializer='glorot_uniform', summary=True):
        """
        Building a sequential LSTM model in Keras
        :param return sequences: whether return sequences has to be True in the sequential RNN model
        :param ckpt_dir: Location for storing the checkpoints
        :param initializer: Which weight initializer to use
        :param summary: True => print model.summary()
        :param return_sequences: True when training, False when predicitng next item
        :return: model of type tf.keras.model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.total_items + 1, #+1 if masking value is total_items
                                      self.embedding_dim,
                                      batch_input_shape=[self.batch_size, None]),

            tf.keras.layers.Masking(mask_value=self.total_items),

            tf.keras.layers.LSTM(units=self.rnn_units,
                                 return_sequences=return_sequences,
                                 stateful=False,  # Reset cell states with each batch
                                 recurrent_initializer=initializer),

            tf.keras.layers.Dense(self.total_items)
        ])
        
        if len(ckpt_dir) > 0:
            model.load_weights(ckpt_dir).expect_partial()
        
        self.model = model
        
        if summary:
            print(model.summary())
    
    
    def train(self, train_set, val_set, callback_names=['checkpoint', 'early_stopping', 'timing'], initial_epoch=0, verbose=1, patience=15):
        """
        Train the LSTM model, ths function only specifies which callbacks are used before calling self.fit
        :param train_set: TF batch training set
        :param val_set: TF batch validation set
        :param callback_names: list of which callbacks to use, available: 'checkpoint', 'early_stopping', 'timing' (timing is defined in Helpers.py)
        :param initial_epoch: same as Keras model
        :param verbose: same as Keras model
        :param patience: same as Keras model
        """
        # Configure Callbacks
        all_callbacks = []
        if 'checkpoint' in callback_names:
            all_callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath = self.ckpt_dir,    
                                                         monitor = 'loss',    
                                                         mode = 'min',   
                                                         save_weights_only = True,
                                                         period = 10))
            
        if 'early_stopping' in callback_names:
            all_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                                           min_delta = 0.0001,
                                                           mode = 'min',
                                                           patience = patience))
            
        if 'timing' in callback_names:
            all_callbacks.append(TimingCallback())
        
        self.fit(train_set, val_set, all_callbacks, initial_epoch, verbose)
        
        
    def fit(self, train_set, val_set, all_callbacks, initial_epoch, verbose):
        """
        Wrapper of Keras.model.fit function, storing the history in self.history
        :param train_set: TF batch training set
        :param val_set: TF batch validation set
        :param all_callbacks: list of callbacks specified in self.train
        :param initial_epoch: same as Keras model
        :param verbose: same as Keras model
        """
        print('Fitting LSTM with parameters:')
        print(pd.DataFrame.from_dict(self.params, orient='index')[0])
        self.history = self.model.fit(x=train_set, 
                        validation_data=val_set, 
                        epochs=self.epochs,
                        callbacks=all_callbacks,
                        verbose=verbose,
                        initial_epoch=initial_epoch)
    

    def store_model(self, path, params, history, train_time, eval_metrics=[], store=True):
        """
        Storing the trained and/or tested LSTM results in a pandas dataframe
        :param path: Where to store / add this dataframe with the existing model
        :param params: Parameters used for the model
        :param history: Training History of the model
        :param train_time: Elapsed train time of the model
        :param eval_metrics: Prediction Metrics
        :param store: Whether to actually store the df or return the df
        :return: all_models dataframe containing rows of all CFRNN models stored in path
        """
        total_recall = 0
        if len(eval_metrics) > 0:
            total_recall = eval_metrics['recall'].sum()

        final_results = {**self.params, **history,
                         'train_time': train_time,
                         'test_recall':total_recall}

        if os.path.exists(path):
            all_models = pd.read_pickle(path)
            all_models = all_models.append(final_results, ignore_index=True)
        else:
            all_models = pd.DataFrame(columns=final_results.keys())
            all_models = all_models.append(final_results, ignore_index=True)

        if store:
            all_models.to_pickle(path)
            
        return all_models
    
    
#     def recall_metric(self):
#         """

#         :param labels:
#         :param logits:
#         :return:
#         """
#         def recall(labels, logits):
#             labels = K.one_hot(tf.dtypes.cast(labels, tf.int32), self.total_items)
#             labels = K.ones_like(labels)
#             true_positives = K.sum(K.round(K.clip(labels * logits, 0, 1)))
#             possible_positives = K.sum(K.round(K.clip(labels, 0, 1)))
#             r = true_positives / (possible_positives + K.epsilon())
#             return r
#         return recall


    def create_diversity_bias(self, train_set):
        """
        Pre-calculates the diversity bias needed in the diversity_bias_loss, stores db in self.diversity_bias
        :param train_set: the train_set as a pandas df: user_id, item_id, datetime
        """
        item_id_bins = np.zeros((1, self.total_items+1), np.float32)
        item_counts = train_set.groupby('item_id')['user_id'].count().sort_values(ascending=False)
        bins = np.logspace(np.log10(item_counts.max()), np.log10(1), 11)
        item_counts.index, np.digitize([item_counts],bins)

        for item_id, count  in zip(item_counts.index, list(item_counts)):
            item_bin = np.digitize([count],bins)
            item_id_bins[0,item_id] = item_bin

        diversity_bias = tf.Variable(np.exp(item_id_bins[0] * -self.delta))
        
        self.diversity_bias = diversity_bias


    def diversity_bias_loss(self):
        """
        (decorator) Calculates Categorical Cross Entropy Loss divided by the diversity bias (self.diversity_bias created in self.create_diversity_bias)as defined in Paper 1
        :return: categorical cross entropy loss function adjusted by the diversity bias
        """
        def loss(labels, logits):
            labels = tf.dtypes.cast(labels, tf.int32)
            oh_labels = K.one_hot(labels, self.total_items)
            standard_loss = tf.keras.losses.categorical_crossentropy(oh_labels, logits, from_logits=True)
            label_weights = tf.gather(self.diversity_bias, labels, axis=0)
            db_loss = tf.math.multiply(standard_loss, label_weights)
            return db_loss
        return loss


    def cce_loss(self):
        """
        Calculates Categorical Crossentropy Loss over the one hot encoded labels with the logits
        :return: categorical cross entropy loss function
        """
        def loss(labels, logits):
            oh_labels = K.one_hot(tf.dtypes.cast(labels, tf.int32), self.total_items)
            return tf.keras.losses.categorical_crossentropy(oh_labels, logits, from_logits=True)
        return loss

    
    def compile_model(self, diversity_bias=True, train_set=[]):
        """
        Compiles the model build with self.build_LSTM, creating the diversity_bias when True and train_set is provided using Keras.model.compile
        :param diversity_bias: whether to include the diversity bias in the loss or not
        :param train_set: the pandas df train_set needed for the pre-calculation of the diversity_bias in create_diversity_bias
        """
        if diversity_bias:
            if len(train_set) == 0:
                raise Exception('Cannot create Diversity Bias without a train set')
            print('Creating Diveristy Bias')
            self.create_diversity_bias(train_set)
            loss = self.diversity_bias_loss()
        else:
            loss = self.cce_loss()
        
        optimizer = tf.keras.optimizers.Adagrad(lr=self.learning_rate)
        metrics = [self.recall_metric()]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print('Compiled LSTM')
        
        
    def create_seq_batch_tf_dataset(self, df, shift=1, stats=True, drop_remainder=True):
        """
        :param df: pandas df where each row consists of user_id, item_id and datetime (chronologically)
        :param shift: how much to shift the x sequences
        :param max_seq_len: maximum sequence length
        :param drop_remainder: drop remainder as in tf.dataset.batch
        :return:
        """
        user_sequences_x, user_sequences_y, median = get_x_y_sequences(df, shift, stats=stats)
        sequences_data_x = standard_padding(user_sequences_x, self.max_seq_len, pad_value=self.pad_value, stats=stats)
        sequences_data_y = standard_padding(user_sequences_y, self.max_seq_len, pad_value=self.pad_value, stats=stats)

        dataset = tf.data.Dataset.zip((sequences_data_x, sequences_data_y))
        dataset = dataset.batch(self.batch_size, drop_remainder=drop_remainder)

        return dataset

    
    def data_split(self, df, val=False):
        """
        Split df according to self.val_users, self.test_user
        :param df: pandas df where each row consists of user_id, item_id and datetime (chronologically)
        """
        if val:
            n = self.val_users
        else: 
            n = self.test_users
            
        users_ids = np.random.choice(df['user_id'].unique(), n, replace=False)
        n_set = df[df['user_id'].isin(users_ids)]
        remaining_set = df.drop(n_set.index)
        return remaining_set, n_set

    
    def get_predictions(self, train_set, test_set, left_out_items, batch_size, rank_at, ckpt_dir='', summary=False, exclude_already_seen=True):
        """
        Uses the stored Keras LSTM model with batch size set to None to predict the rest of the sequences from the data per user.
        Finally creates predictions_df where each row represents user, a list pred_items_ranked and a list containing true_ids
        from the left_out df
        :param train_set: pandas df where each row consists of user_id, item_id and datetime (chronologically) used in training
        :param test_set: pandas df where each row consists of user_id, item_id and datetime (chronologically) to be used now (contains all but 1 item of the test users)
        :param left_out_items: pandas df where each row consists of user_id, item_id and datetime (chronologically) with the held-out items
        :param rank_at: maximum rank to compute the metrics on
        :param batch_size: batch_size==number of test users
        :param summary: True => print model.summary()
        :param exclude_already_seen: whether to exclude the items already seen in the train_set when predicting
        :return: pandas df where each row represents a user, the columns represent: pred_items_ranked at rank_at,
                 true_id extracted from test_set (as input for Evaluation.get_metrics
        """
        self.batch_size = None
        self.build_model(ckpt_dir=ckpt_dir, return_sequences=False, summary=summary)
            
        n_batches = int(len(left_out_items) / batch_size)
        data_sequences, _, _ = get_x_y_sequences(test_set, stats=False)
        data_seqs_padded = standard_padding(data_sequences, self.max_seq_len, self.pad_value, eval=True, stats=False)
        data_seqs_splits = np.array_split(data_seqs_padded, n_batches, axis=0)
        if exclude_already_seen:
            already_seen = pd.concat([train_set, test_set]).groupby('user_id')['item_id'].apply(list)
            
        # Get True items
        test_left_out_items = left_out_items.groupby('user_id')['item_id'].apply(list)
        
        # Extend final predictions with predictions made on batches
        preds = []
        for split in data_seqs_splits:
            preds.extend(self.model.predict(split))
        
        # Exclude alredy seen items
        final_preds = []
        
        for user, pred in zip(test_left_out_items.index, preds):
            if exclude_already_seen:
                pred[already_seen[user]] = -np.inf
            ids = np.argpartition(pred, -rank_at)[-rank_at:]
            final_preds.append(ids[np.argsort(pred[ids][::-1])])
            
        preds_df = pd.DataFrame(list(zip(test_left_out_items.index, final_preds, list(test_left_out_items))),
                                columns=['user', 'pred_items_ranked', 'true_id'])

        return preds_df


    # TODO: add store_path
    def plot_training(self):
        """
        Plot the training loss and validation loss
        """
        his = self.history.history
        
        plt.plot(his['loss'])
        plt.plot(his['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.title('Training Loss')
        plt.show()
        
#         plt.plot(his['recall'])
#         plt.plot(his['val_recall'])
#         plt.legend(['recall', 'val_recall'])
#         plt.title('Training Recall')
#         plt.show()