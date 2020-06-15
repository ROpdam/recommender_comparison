import numpy as np
import time
import math
import pandas as pd
import random
import os
import progressbar
import tensorflow as tf
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
        self.model_id = params['model_id']
        self.train_time = params['train_time']
        self.epochs = params['epochs']
        self.batch_size = params['BATCH_SIZE']
        self.learning_rate = params['learning_rate']
        self.delta = params['delta']
        self.max_seq_len = params['max_seq_len']
        self.embedding_dim = params['embedding_dim']
        self.rnn_units = params['rnn_units']
        self.ckpt_dir = ''.join([params['ckpt_dir'], '_', params['model_id']])
        self.pad_value = params['pad_value']
        self.test_users = params['test_users']
        self.val_users = params['val_users']
        self.model = []
        self.history = {}
        self.diversity_bias = []
     
    
    def build_model(self, ckpt_dir='', return_sequences=True, initializer='glorot_uniform', summary=True):
        """
        Building the LSTM model in Keras
        :param total_items: Number of items from the full df
        :param embedding_dim: Number of embedding dimensions (100)
        :param mask_value: Value used for Masking, NOTE: total_items is used for padding and masking so embedding +1
        :param rnn_units: Number of hidden units
        :param batch_size: batch_size
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
            model.load_weights(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
        
        self.model = model
        model._name = self.model_id
        
        if summary:
            print(model.summary())
    
    
    def train(self, train_set, val_set, callback_names=['checkpoint', 'early_stopping', 'store_hist', 'timing'], initial_epoch=0, verbose=1, append_hist=True):
        # Configure Callbacks
        all_callbacks = []
        if 'checkpoint' in callback_names:
            all_callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath = self.ckpt_dir,    
                                                         monitor = 'val_recall',    
                                                         mode = 'max',    
                                                         save_best_only = True,
                                                         save_weights_only = True))
            
        if 'early_stopping' in callback_names:
            all_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor = 'val_recall',
                                                           min_delta = 0.0001,
                                                           mode = 'max',
                                                           patience = 15))
            
        if 'store_hist' in callback_names:
                 all_callbacks.append(tf.keras.callbacks.CSVLogger(f'../CFRNN_storage/train_logs/log_{self.model_id }',
                                                              append=append_hist))
            
        if 'timing' in callback_names:
            all_callbacks.append(TimingCallback())
        
        self.fit(train_set, val_set, all_callbacks, initial_epoch, verbose)
        
        
    def fit(self, train_set, val_set, all_callbacks, initial_epoch, verbose):
        print('Fitting LSTM with parameters:')
        print(pd.DataFrame.from_dict(self.params, orient='index')[0])
        self.history = self.model.fit(x=train_set, 
                        validation_data=val_set, 
                        epochs=self.epochs,
                        callbacks=all_callbacks,
                        verbose=verbose,
                        initial_epoch=initial_epoch)
    
    # To be removed
    def update_results(model_dict, final_results):
        """
        If the model was already present in the all_models dataframe in store_LSTM_model, this function will change
        the number of epochs, add the train_time and append the training history of the model_dict model provided by
        store_LSTM_model
        :param model_dict: all stats from a model as a dict
        :param final_results: the new stats of the same model to update model_dict with
        :return: Updated final_results with model_dict
        """
        print('Adding to existing DataFrame')
        final_results['epochs'] = model_dict['epochs']
        final_results['train_time'] += model_dict['train_time']
        train_values = ['recall', 'val_recall', 'loss', 'val_loss']
        for train_value in train_values:
            final_results[train_value].extend(model_dict[train_value])

        return final_results


    def store_LSTM_model(self, path, params, history, train_time, eval_metrics=[], store=True):
        """
        Storing the trained and/or tested LSTM model in:
        1. An existing pandas dataframe if it already exists in path
        2. A new pandas dataframe
        :param path: Where to store / add this dataframe with the existing model
        :param params: Parameters used for the model
        :param history: Training History of the model
        :param train_time: Elapsed train time of the model
        :param eval_metrics: Prediction Metrics
        :param store: Whether to actually store the df or return the df
        :return: all_models dataframe which keeps track of:
        TODO: Fill in list of columns from all_models
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


    def get_predictions(self, test_set, left_out_items, batch_size, rank_at, ckpt_dir='', summary=False):
        """
        Uses a Keras LSTM model with batch size set to None to predict the rest of the sequences from the      data per user.
        Finally creates predictions_df where each row represents user, a list pred_items_ranked and a list containing true_ids
        from the left_out df
        :param test_set: Test or Validation set (pandas)
        :param left_out_items: left out items (pandas)
        :param batch_size: batch_size==number of test users
        :param pad_value: (mask_value) pad_value==total_items (as done while training)
        :param rank_at: maximum number of predictions to make
        :return: pandas df where each row represents a user, the columns represent: pred_items_ranked at rank_at,
                 true_id extracted from test_set (as input for Evaluation.get_metrics
        """
        self.batch_size = None
        self.build_LSTM_model(mask_value=self.pad_value, ckpt_dir=ckpt_dir, return_sequences=False, summary=summary)
            
        n_batches = int(len(left_out_items) / batch_size)
        data_sequences, _, _ = get_x_y_sequences(test_set, stats=False)
        data_seqs_padded = standard_padding(data_sequences, self.max_seq_len, pad_value, eval=True, stats=False)
        data_seqs_splits = np.array_split(data_seqs_padded, n_batches, axis=0)

        # Extend final predictions with predictions made on batches
        final_preds = []
        for split in data_seqs_splits:
            final_preds.extend(self.make_predictions(split, pad_value, rank_at))

        # Get True items
        test_left_out_items = left_out_items.groupby('user_id')['item_id'].apply(list)

        preds_df = pd.DataFrame(list(zip(test_left_out_items.index, final_preds, list(test_left_out_items))),
                                columns=['user', 'pred_items_ranked', 'true_id'])

        return preds_df


    def make_predictions(self, user_sequences, rank_at):
        """
        :param model:
        :param user_sequences:
        :param rank_at:
        :return:
        """
        final_preds = np.zeros((user_sequences.shape[0], rank_at), dtype='int32')
        for i in range(rank_at):
            predictions = self.model.predict(user_sequences)
            for u_index, prediction in enumerate(predictions):
                pred_item_id = np.argmax(prediction)
                final_preds[u_index][i] = pred_item_id

                padding_values = np.where(user_sequences[u_index] == self.pad_value)[0]
                if padding_values.shape[0] > 0:
                    first_pad_value = np.min(padding_values)
                    user_sequences[u_index][first_pad_value] = pred_item_id
                else:
                    new_user_sequence = np.append(user_sequences[u_index], pred_item_id)[1:]
                    user_sequences[u_index] = new_user_sequence
                    
        return final_preds

    
    def recall_metric(self):
        """

        :param labels:
        :param logits:
        :return:
        """
        def recall(labels, logits):
            labels = K.one_hot(tf.dtypes.cast(labels, tf.int32), self.total_items)
            labels = K.ones_like(labels)
            true_positives = K.sum(K.round(K.clip(labels * logits, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(labels, 0, 1)))
            r = true_positives / (possible_positives + K.epsilon())
            return r
        return recall


    def create_diversity_bias(self, train_set):
        """
        Pre-calculates the diversity bias needed in
        :param train_set:
        :param total_items:
        :param delta:
        :return:
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
        Calculates Categorical Cross Entropy Loss divided by the diversity bias as defined in Paper 1
        :param db: precalculated diversity bias per item_id
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
        :param total_items: Maximum item_id for one hot encoding
        :return: categorical cross entropy loss function
        """
        def loss(labels, logits):
            oh_labels = K.one_hot(tf.dtypes.cast(labels, tf.int32), self.total_items)
            return tf.keras.losses.categorical_crossentropy(oh_labels, logits, from_logits=True)
        return loss

    def compile_model(self, diversity_bias=True, train_set=[]):
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
        :param df:
        :param shift:
        :param max_seq_len:
        :param pad_value:
        :param batch_size:
        :param drop_remainder:
        :return:
        """
        user_sequences_x, user_sequences_y, median = get_x_y_sequences(df, shift, stats=stats)
        sequences_data_x = standard_padding(user_sequences_x, self.max_seq_len, pad_value=self.pad_value, stats=stats)
        sequences_data_y = standard_padding(user_sequences_y, self.max_seq_len, pad_value=self.pad_value, stats=stats)

        dataset = tf.data.Dataset.zip((sequences_data_x, sequences_data_y))
        dataset = dataset.batch(self.batch_size, drop_remainder=drop_remainder)

        return dataset

    
    def data_split(self, df, val=False):
        if val:
            n = self.val_users
        else: 
            n = self.test_users
            
        users_ids = np.random.choice(df['user_id'].unique(), n, replace=False)
        n_set = df[df['user_id'].isin(users_ids)]
        remaining_set = df.drop(n_set.index)
        return remaining_set, n_set