import numpy as np
import time
import math
import pandas as pd
import random
import os
from sklearn.metrics import roc_auc_score
import progressbar
import tensorflow as tf

class BPR():
    """
    Bayesian Personalised Ranking
    :param total_users: the total users before the train test split to be used in the user factor p
    :param total_items: the total items before the train test split to be used in the item factor q
    :param params: parameters to be used by the algorithm
    """
    def __init__(self, total_users, total_items, params):
        self.total_users = total_users
        self.total_items = total_items
        self.params = params
        self.nolf = params['nolf']
        self.n_iterations = params['n_iterations']
        self.sample_size = params['sample_size']
        self.seed = params['seed']
        self.alpha = params['alpha']
        self.reg_user = params['reg_user']
        self.reg_item = params['reg_item']
        self.alpha_decay = self.alpha / self.n_iterations

        self.model = {}
        self.model['val_auc'] = []
        self.user_items = pd.DataFrame()
        self.train_users = []
        self.train_items = []

        self.val_user_items = pd.DataFrame()
        self.val_users = []

    def fit(self, train_set, val_set=[], verbose=1):
        """
        Fit BPR to the train_set, if val_set is provided the AUC metrics will be computed and printed per iteration
        :param train_set: pandas df containing user_id and item_id
        :param val_set: validation pandas df containing user_id and item_id
        :param verbose: 1 means print creating samples, loss per iteration and validation AUC during training
                        (if val_set provided)
        :return: -None: stores values in self.model
        """
        # Init
        s = time.time()
        if self.seed > 0:
            np.random.seed(self.seed)
        p = np.random.normal(0, .1, (self.total_users, self.nolf))  # users
        q = np.random.normal(0, .1, (self.total_items, self.nolf))  # items

        ## Used for sampling
        self.user_items = train_set.groupby('user_id')['item_id'].apply(list)
        self.train_users = train_set.user_id.unique()
        self.train_items = train_set.item_id.unique()

        if len(val_set) > 0:
            ## Used for validating
            self.val_user_items = val_set.groupby('user_id')['item_id'].apply(list)
            self.val_users = val_set.user_id.unique()

        ## Track losses and alphas used
        loss_list = []
        alphas = []

        ## Create samples for all iterations
        if verbose == 1:
            print('Creating', str(self.n_iterations), 'samples of length', str(self.sample_size))
        all_uij_samples = self.sample()

        # Training Loop
        for iteration in range(self.n_iterations):
            it_loss = 0
            uij_samples = all_uij_samples[iteration]

            for uij_sample in uij_samples:
                u = uij_sample[0]
                i = uij_sample[1]
                j = uij_sample[2]

                ## Calculate the difference between positive and negative item
                diff = np.dot(p[u], (q[i] - q[j]).T)

                ## Obtain loss
                loss_value = - np.log(self.sigmoid(diff))
                regulariser = self.reg_user * np.dot(p[u], p[u]) + self.reg_item * np.dot(q[i], q[
                    i]) + self.reg_item / 10 * np.dot(q[j], q[j])
                it_loss += (loss_value + regulariser) / self.sample_size

                ## Derivative of the difference for update
                diff_deriv = self.sigmoid(- diff)

                ## Update the factors of the latent features, using their respective derivatives
                ## See http://ethen8181.github.io/machine-learning/recsys/4_bpr.html
                p[u] += self.alpha * (diff_deriv * (q[i] - q[j]) - self.reg_user * p[u])
                q[i] += self.alpha * (diff_deriv * p[u] - self.reg_item * q[i])
                q[j] += self.alpha * (diff_deriv * (-p[u]) - self.reg_item * q[j])

            ## Store iteration variables
            self.model['p'] = p
            self.model['q'] = q

            if len(val_set) > 0:  # TODO: safe best & early stopping
                val_auc = self.AUC()
                self.model['val_auc'].append(val_auc)
                if verbose == 1:
                    print('iteration:', iteration, ' loss:', round(it_loss, 6), ' val AUC:',
                          val_auc)  # , ' val prec@' + str(val_rank), ':', round(prec_at,5), ' val rec@' + str(val_rank), ':', round(rec_at,5), '  Hits:', hitcount)#'  alpha:', self.alpha)
            elif verbose == 1:
                print('iteration:', iteration, ' loss:', round(it_loss, 6))

            if iteration > 0:
                self.update_alpha(loss_list[-1], it_loss)

            alphas.append(self.alpha)
            loss_list.append(it_loss)

        # Store train values
        train_time = time.time() - s
        self.model['train_loss'] = loss_list
        self.model['learning_rate'] = alphas
        self.model['train_time'] = train_time

    def sample(self):
        """
        Creates n_iteration user (u), positive item (i), negative item (j), samples of sample_size from the training set
        :return: list of n_iteration samples of sample size
        """
        all_uij_samples = []
        for n in range(self.n_iterations):
            uij_samples = []
            for s in range(int(self.sample_size)):
                u = int(np.random.choice(self.train_users))
                u_items = self.user_items[u]
                i = random.choice(u_items)
                j = int(np.random.choice(self.train_items))
                while j in u_items:  # neg item j cannot be in the set of pos items of user u
                    j = int(np.random.choice(self.train_items))

                uij_samples.append([u, i, j])

            all_uij_samples.append(uij_samples)

        return all_uij_samples

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def update_alpha(self, last_loss, it_loss):
        """
        Adjust learning rate according to the bold driver principle
        :param last_loss: loss of previous iteration
        :param it_loss: current loss
        :return: -None: changes the learning rate alpha of the model
        """
        if (last_loss < it_loss):  # bold driver
            self.alpha = 0.5 * self.alpha
            return

        self.alpha = (1 - self.alpha_decay) * self.alpha

    def AUC(self):
        """
        Compute AUC of the validation set
        :return: AUC score
        """
        auc = 0.0
        n_users = len(self.val_users)

        for u in self.val_users:
            y_pred = np.dot(self.model['p'][u], self.model['q'].T)
            y_true = np.zeros(self.total_items)
            y_true[self.val_user_items[u]] = 1
            auc += roc_auc_score(y_true, y_pred)

        auc /= n_users
        return auc

    def store_results(self, log_path, res_name, file_name, stats=True):
        """
        Store the model as a row in a pandas df (pickle) named res_name, stores: train loss, val_auc, train_time,
        learning_rates, file_name, p and q factors
        :param log_path: where to store/find the results
        :param res_name: the name of the results pandas df, if name is not found, a new pandas df is created and
                         stored (pickle)
        :param file_name: the name of the dataset file
        :param stats: print whether new results are created or the current model is added to an existing pandas df
        :return: -None:
        """
        result_info = {'train_loss': self.model['train_loss'], 'val_auc': self.model['val_auc'],
                       'train_speed': self.model['train_time'], 'lr': self.model['learning_rate'], 'file': file_name}
        other_info = {'p': self.model['p'],
                      'q': self.model['q']}  # 'train_size':train_size, 'test_size':test_size, 'val_size':val_size}
        final_log = dict(result_info, **self.params, **other_info)

        if not os.path.exists(log_path + res_name):
            df_results = pd.DataFrame(columns=final_log.keys())
            df_results.to_pickle(log_path + res_name)
            if stats:
                print('new results created')

        else:
            df_results = pd.read_pickle(log_path + res_name)
            if stats:
                print('results added')

        df_results = df_results.append(final_log, ignore_index=True)
        df_results.to_pickle(log_path + res_name)


######################################## LSTM ###########################################

#Architecture
def build_LSTM_model(total_items, embedding_dim, mask_value, rnn_units, batch_size, return_sequences=True):
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
        tf.keras.layers.Embedding(total_items + 1, #+1 if masking value is total_items
                                  embedding_dim,
                                  batch_input_shape=[batch_size, None]),

        tf.keras.layers.Masking(mask_value=mask_value),

        tf.keras.layers.LSTM(units=rnn_units,
                             return_sequences=return_sequences,
                             stateful=False,  # Reset cell states with each batch
                             recurrent_initializer='glorot_uniform'),

        tf.keras.layers.Dense(total_items)
    ])
    return model

#Storage
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


def store_LSTM_model(path, params, history, train_time, eval_metrics=[], store=True):
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

    final_results = {**params, **history,
                     'train_time': train_time,
                     'epochs': len(history['loss']),
                     'test_recall':total_recall}

    if os.path.exists(path):
        all_models = pd.read_pickle(path)
        if final_results['model_id'] in set(all_models['model_id']):
            model_index = 0
            if len(all_models) > 1:
                model_index = all_models[all_models['model_id'] == final_results['model_id']].index[0]
            new_final_results = update_results(all_models.iloc[model_index].to_dict(), final_results)
            all_models = all_models.drop(model_index).append(new_final_results, ignore_index=True)
        else:
            all_models = all_models.append(final_results, ignore_index=True)
    else:
        all_models = pd.DataFrame(columns=final_results.keys())
        all_models = all_models.append(final_results, ignore_index=True)

    if store:
        all_models.to_pickle(path)
    return all_models

######################################## NeuMF ###########################################

def build_GMF_model(total_items, total_users, nolf, regs=[0, 0], seed=1234):
    user_input = tf.keras.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = tf.keras.Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = tf.keras.layers.Embedding(input_dim=total_users,
                                               output_dim=nolf,
                                               name='user_latent_factors',
                                               embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                         stddev=0.01,
                                                                                                         seed=seed),
                                               embeddings_regularizer=tf.keras.regularizers.l2(regs[0]),
                                               input_length=1)

    item_embedding = tf.keras.layers.Embedding(input_dim=total_items,
                                               output_dim=nolf,
                                               name='item_latent_factors',
                                               embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                         stddev=0.01,
                                                                                                         seed=seed),
                                               embeddings_regularizer=tf.keras.regularizers.l2(regs[1]),
                                               input_length=1)

    user_latent_f = tf.keras.layers.Flatten()(user_embedding(user_input))
    item_latent_f = tf.keras.layers.Flatten()(item_embedding(item_input))

    predict_vector = tf.keras.layers.Multiply()([user_latent_f, item_latent_f])

    final_predictions = tf.keras.layers.Dense(units=1,
                                              activation='sigmoid',
                                              kernel_initializer='lecun_uniform',
                                              name='prediction')(predict_vector)

    model = tf.keras.Model(inputs=[user_input, item_input],
                           outputs=[final_predictions])
    model._name="GMF"

    return model


def build_MLP_model(total_items, total_users, layers=[20,10], reg_layers=[0,0], seed=1234):
    # Total Layers
    num_layers=len(layers)
    
    # Inputs
    user_input = tf.keras.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = tf.keras.Input(shape=(1,), dtype='int32', name='item_input')

    # First Layer
    MLP_Embedding_User = tf.keras.layers.Embedding(input_dim=total_users, 
                                   output_dim=int(layers[0]/2), 
                                   name='user_latent_factors',
                                   embeddings_initializer= tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed), 
                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0]), 
                                   input_length=1)
    
    MLP_Embedding_Item = tf.keras.layers.Embedding(input_dim=total_items, 
                                   output_dim=int(layers[0]/2), 
                                   name='item_latent_factors',
                                   embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed), 
                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0]), 
                                   input_length=1)   
    
    user_latent_f = tf.keras.layers.Flatten()(MLP_Embedding_User(user_input))
    item_latent_f = tf.keras.layers.Flatten()(MLP_Embedding_Item(item_input))

    predict_vector = tf.keras.layers.Concatenate(axis=-1)([user_latent_f, item_latent_f])
    
    # Layers
    for layer_id in range(1, num_layers):
        layer = tf.keras.layers.Dense(layers[layer_id], 
                      kernel_regularizer=tf.keras.regularizers.l2(reg_layers[layer_id]),  
                      bias_regularizer=tf.keras.regularizers.l2(reg_layers[layer_id]), 
                      activation='relu', 
                      name=f'layer{layer_id}')
        predict_vector = layer(predict_vector)
        
    # Final prediction layer
    prediction = tf.keras.layers.Dense(1, 
                       activation='sigmoid', 
                       kernel_initializer='lecun_uniform', 
                       bias_initializer ='lecun_uniform',   
                       name='prediction')(predict_vector)
    
    model = tf.keras.Model(inputs=[user_input, item_input], 
                  outputs=[prediction])
    
    return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers, alpha=0.5):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_latent_factors').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_latent_factors').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_latent_factors').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_latent_factors').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for layer_id in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer(f'layer{layer_id}').get_weights()
        model.get_layer(f'layer{layer_id}').set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    
    model.get_layer('prediction').set_weights([alpha*new_weights, (1-alpha)*new_b])    
    
    return model


def build_NeuMF_model(total_users, total_items, mf_nolf=10, reg_mf=[0,0], layers=[10], reg_layers=[0]):
    num_layer = len(layers)
    
    user_input = tf.keras.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = tf.keras.Input(shape=(1,), dtype='int32', name='item_input')
    
    MF_Embedding_User = tf.keras.layers.Embedding(input_dim=total_users,
                                  output_dim=mf_nolf,
                                  name='mf_embedding_user',
                                  embeddings_initializer='normal',
                                  embeddings_regularizer=tf.keras.regularizers.l2(reg_mf[0]),
                                  input_length=1)
    MF_Embedding_Item = tf.keras.layers.Embedding(input_dim=total_items,
                                  output_dim=mf_nolf,
                                  name='mf_embedding_item',
                                  embeddings_initializer='normal',
                                  embeddings_regularizer=tf.keras.regularizers.l2(reg_mf[1]),
                                  input_length=1)
    
    MLP_Embedding_User = tf.keras.layers.Embedding(input_dim=total_users,
                                                   output_dim=int(layers[0]/2),
                                                   name='mlp_embedding_user',
                                                   embeddings_initializer='normal',
                                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0]),
                                                   input_length=1)
    MLP_Embedding_Item = tf.keras.layers.Embedding(input_dim=total_items,
                                                   output_dim=int(layers[0]/2),
                                                   name='mlp_embedding_item',
                                                   embeddings_initializer='normal',
                                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0]),
                                                   input_length=1)
    
    mf_user_latent = tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))
    mf_vector = tf.keras.layers.Multiply()([mf_user_latent, mf_item_latent])
    
    mlp_user_latent = tf.keras.layers.Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = tf.keras.layers.Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = tf.keras.layers.Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])
    
    for layer_id in range(1, num_layer):
        layer = tf.keras.layers.Dense(units=layers[layer_id], 
                                      kernel_regularizer=tf.keras.regularizers.l2(reg_layers[layer_id]), 
                                      activation='relu', 
                                      name=f'layer{layer_id}')
        mlp_vector = layer(mlp_vector)
    
    predict_vector = tf.keras.layers.Concatenate(axis=-1)([mf_vector, mlp_vector])
    prediction = tf.keras.layers.Dense(units=1, 
                                       activation='sigmoid', 
                                       kernel_initializer='lecun_uniform', 
                                       name = "prediction")(predict_vector)
    
    model = tf.keras.Model(inputs=[user_input, item_input], 
                  outputs=[prediction])

    model._name = 'NeuMF'
    
    return model


def neumf_train_loop(model, samples, params, callbacks, history={'loss':[]}):
    print(f'\nFitting {model.name} with parameters:')
    print('Parameters:', pd.DataFrame.from_dict(params, orient='index'))
    
    all_user_inputs, all_item_inputs, all_labels = samples
    for epoch in range(params['epochs']):
        print(f'Epoch: {epoch}')

        user_inputs = all_user_inputs[epoch]
        item_inputs = all_item_inputs[epoch]
        labels = all_labels[epoch]

        hist = model.fit([np.array(user_inputs), np.array(item_inputs)], 
                  np.array(labels), 
                  batch_size=params['batch_size'], 
                  verbose=1, 
                  epochs=1, 
                  shuffle=True,
                  callbacks=callbacks)

        history['loss'].append(round(hist.history['loss'][0],5))
    
    return model, history


def create_NeuMF_samples(data, epochs, sample_size, n_user_neg_samples=1):
    all_user_inputs, all_item_inputs, all_labels = [], [], []
    user_items = data.groupby('user_id')['item_id'].apply(list)
    train_users = data.user_id.unique()
    train_items = data.item_id.unique()

    pbar = progressbar.ProgressBar()
    for n in pbar(range(epochs)):
        user_inputs, item_inputs, labels = [], [], []
        for s in range(int(sample_size)):
            # Add positive item
            u = np.random.choice(train_users)
            u_items = user_items[u]
            i = np.random.choice(u_items)

            user_inputs.append(u)
            item_inputs.append(i)
            labels.append(1)

            # Add negative item
            for i in range(n_user_neg_samples):
                j = np.random.choice(train_items)
                while j in u_items:  # neg item j cannot be in the set of pos items of user u
                    j = np.random.choice(train_items)

                user_inputs.append(u)
                item_inputs.append(j)
                labels.append(0)

        all_user_inputs.append(user_inputs)
        all_item_inputs.append(item_inputs)
        all_labels.append(labels)

    return [all_user_inputs, all_item_inputs, all_labels]