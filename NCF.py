import numpy as np
import time
import math
import pandas as pd
import os
import progressbar
import tensorflow as tf

class NCF:
    """
    Using the NCF we build Generalized Matrix Factorisation (GMF), Multiplayer Perceptron Matrix Factorisation (MLP) and combine the two     in Neural Matrix Factorisation (NeuMF)
    - paper: http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p173.pdf
    - blog: https://medium.com/@victorkohler/collaborative-filtering-using-deep-neural-networks-in-tensorflow-96e5d41a39a1
    - code: https://github.com/Leavingseason/NeuralCF/blob/master
    """
    def __init__(self, total_users, total_items, GMF_params={}, MLP_params={}, NeuMF_params={}):
        self.total_users = total_users
        self.total_items = total_items
        self.GMF_params = GMF_params
        self.MLP_params = MLP_params
        self.NeuMF_params = NeuMF_params
        self.GMF = ''
        self.MLP = ''
        self.NeuMF = ''
        self.history = {'GMF':{'loss':[]}, 'MLP':{'loss':[]}, 'NeuMF':{'loss':[]}}
    
    def build_GMF_model(self, seed=1234):
        try:
            nolf = self.GMF_params['nolf']
            regs = self.GMF_params['regs']
        except:
            raise Exception('GMF_params empty')
        
        user_input = tf.keras.Input(shape=(1,), dtype='int32', name='user_input')
        item_input = tf.keras.Input(shape=(1,), dtype='int32', name='item_input')

        user_embedding = tf.keras.layers.Embedding(input_dim=self.total_users,
                                                   output_dim=nolf,
                                                   name='user_latent_factors',
                                                   embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                             stddev=0.01,
                                                                                                             seed=seed),
                                                   embeddings_regularizer=tf.keras.regularizers.l2(regs[0]),
                                                   input_length=1)

        item_embedding = tf.keras.layers.Embedding(input_dim=self.total_items,
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
        self.GMF = model
        
        self.compile_model(model._name)


    def build_MLP_model(self, optimizer='Adam', seed=1234):
        try:
            layers = self.MLP_params['layers']
            reg_layers = self.MLP_params['reg_layers']
        except:
            raise Exception('No layers or reg_layers provided in MLP_params')
            
        # Total Layers
        num_layers=len(layers)

        # Inputs
        user_input = tf.keras.Input(shape=(1,), dtype='int32', name='user_input')
        item_input = tf.keras.Input(shape=(1,), dtype='int32', name='item_input')

        # First Layer
        MLP_Embedding_User = tf.keras.layers.Embedding(input_dim=self.total_users, 
                                       output_dim=int(layers[0]/2), 
                                       name='user_latent_factors',
                                       embeddings_initializer= tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed), 
                                       embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0]), 
                                       input_length=1)

        MLP_Embedding_Item = tf.keras.layers.Embedding(input_dim=self.total_items, 
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
        model._name = 'MLP'
        self.MLP = model
        
        self.compile_model(model._name)

    #TODO: Implement keras normal initializers to pick seed yourself
    def build_NeuMF_model(self):
        try:
            mf_nolf = self.NeuMF_params['nolf']
            reg_mf = self.NeuMF_params['reg_mf']
            layers = self.NeuMF_params['layers']
            reg_layers = self.NeuMF_params['reg_layers']
        except:
            raise Exception('Missing one of the following in NeuMF_params:nolf, reg_mf, layers or reg_layers')
            
        num_layer = len(layers)

        user_input = tf.keras.Input(shape=(1,), dtype='int32', name='user_input')
        item_input = tf.keras.Input(shape=(1,), dtype='int32', name='item_input')

        MF_Embedding_User = tf.keras.layers.Embedding(input_dim=self.total_users,
                                      output_dim=mf_nolf,
                                      name='mf_embedding_user',
                                      embeddings_initializer='normal',
                                      embeddings_regularizer=tf.keras.regularizers.l2(reg_mf[0]),
                                      input_length=1)
        MF_Embedding_Item = tf.keras.layers.Embedding(input_dim=self.total_items,
                                      output_dim=mf_nolf,
                                      name='mf_embedding_item',
                                      embeddings_initializer='normal',
                                      embeddings_regularizer=tf.keras.regularizers.l2(reg_mf[1]),
                                      input_length=1)

        MLP_Embedding_User = tf.keras.layers.Embedding(input_dim=self.total_users,
                                                       output_dim=int(layers[0]/2),
                                                       name='mlp_embedding_user',
                                                       embeddings_initializer='normal',
                                                       embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0]),
                                                       input_length=1)
        MLP_Embedding_Item = tf.keras.layers.Embedding(input_dim=self.total_items,
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

        self.NeuMF = model
        self.compile_model(model._name)
        

    def compile_model(self, model_name):
        model, params = self.get_model(model_name)
        if params['optimizer'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])
        elif params['optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr=params['learning_rate'])
        elif params['optimizer'] == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(lr=params['learning_rate'])
        elif params['optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(lr=params['learning_rate'])
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
            
        
    def train_model(self, name='', samples=[], verbose=1, store_path=''):
        model, params = self.get_model(name)
        
        ckpts_prefix = os.path.join(params['ckpt_dir'], "ckpt")
        ckpts_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpts_prefix,    
                                                         monitor='loss',    
                                                         mode='min',    
                                                         save_best_only=True,
                                                         save_weights_only=True)
        
        if len(samples) == 0:
            raise Exception('No samples available, create samples first using: create_samples')
        
        self.fit(model, params, samples, [ckpts_callback], verbose)
        
        if len(store_path) > 0:
            model.save_weights(store_path)
        

    def fit(self, model, params, samples, callbacks, verbose):
        print(f'\nFitting {model._name} with parameters:')
        print(pd.DataFrame.from_dict(params, orient='index'))
        all_user_inputs, all_item_inputs, all_labels = samples
        for epoch in range(params['epochs']):
            print(f'Epoch: {epoch}')

            user_inputs = all_user_inputs[epoch]
            item_inputs = all_item_inputs[epoch]
            labels = all_labels[epoch]

            hist = model.fit([np.array(user_inputs), np.array(item_inputs)], 
                      np.array(labels), 
                      batch_size=params['batch_size'], 
                      verbose=verbose, 
                      epochs=1, 
                      shuffle=True,
                      callbacks=callbacks)

            self.history[model._name]['loss'].append(round(hist.history['loss'][0],5))
    
    
    #TODO: Multiprocessing?
    def create_samples(self, data, name=''):
        print(f'Creating Samples for {name}')
        _, params = self.get_model(name)
        all_user_inputs, all_item_inputs, all_labels = [], [], []
        user_items = data.groupby('user_id')['item_id'].apply(list)
        train_users = data.user_id.unique()
        train_items = data.item_id.unique()

        pbar = progressbar.ProgressBar()
        for n in pbar(range(params['epochs'])):
            user_inputs, item_inputs, labels = [], [], []
            for s in range(int(params['sample_size'])):
                # Add positive item
                u = np.random.choice(train_users)
                u_items = user_items[u]
                i = np.random.choice(u_items)

                user_inputs.append(u)
                item_inputs.append(i)
                labels.append(1)

                # Add negative item
                for i in range(params['num_neg']):
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
           
        
    def get_model(self, name):
        if self.GMF._name == name:
            return self.GMF, self.GMF_params
        elif self.MLP._name == name:
            return self.MLP, self.MLP_params
        elif self.NeuMF._name == name:
            return self.NeuMF, self.NeuMF_params
        else:
            raise Exception(f'{name} is an unkown model or not built yet')

            
    def use_pretrain_model(self, GMF_weights_path='', MLP_weights_path='', alpha=0.5):
        if len(GMF_weights_path) > 0:
            self.build_GMF_model()
            self.GMF.load_weights(GMF_weights_path).expect_partial()
        if len(MLP_weights_path) > 0:
            self.build_MLP_model()
            self.MLP.load_weights(MLP_weights_path).expect_partial()
        
        # MF embeddings
        gmf_user_embeddings = self.GMF.get_layer('user_latent_factors').get_weights()
        gmf_item_embeddings = self.GMF.get_layer('item_latent_factors').get_weights()
        self.NeuMF.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
        self.NeuMF.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

        # MLP embeddings
        mlp_user_embeddings = self.MLP.get_layer('user_latent_factors').get_weights()
        mlp_item_embeddings = self.MLP.get_layer('item_latent_factors').get_weights()
        self.NeuMF.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
        self.NeuMF.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

        # MLP layers
        for layer_id in range(1, len(self.MLP_params['layers'])):
            mlp_layer_weights = self.MLP.get_layer(f'layer{layer_id}').get_weights()
            self.NeuMF.get_layer(f'layer{layer_id}').set_weights(mlp_layer_weights)

        # Prediction weights
        gmf_prediction = self.GMF.get_layer('prediction').get_weights()
        mlp_prediction = self.MLP.get_layer('prediction').get_weights()
        new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
        new_b = gmf_prediction[1] + mlp_prediction[1]

        self.NeuMF.get_layer('prediction').set_weights([alpha*new_weights, (1-alpha)*new_b])    
        
        
    def get_predictions(self, name, train_set, test_set, rank_at=20):
        model, _ = self.get_model(name)
        test_user_items = test_set.groupby('user_id')['item_id'].apply(list)
        pbar = progressbar.ProgressBar()
        preds_ranked = []
        true_items = []
        for u in pbar(test_user_items.index):
            true_item = test_user_items[u]

            user_array = np.full(self.total_items, u, dtype='int32')
            preds = np.hstack(model.predict([user_array, np.arange(self.total_items)], batch_size=self.total_items, verbose=0))
            ids = np.argpartition(preds, -rank_at)[-rank_at:]
            best_ids = np.argsort(preds[ids])[::-1]
            best = np.arange(self.total_items)[ids[best_ids]]

            preds_ranked.append(best)
            true_items.append(true_item)

        ranked_df = pd.DataFrame(list(zip(test_user_items.index, preds_ranked, true_items)),
                                 columns=['users', 'pred_items_ranked', 'true_id'])
        return ranked_df

    #To be removed
    def sample_prediction(self, name, train_set, test_set, sample_len=100, rank_at=20):
        model, _ = self.get_model(name)
        user_items = train_set.groupby('user_id')['item_id'].apply(list)
        test_user_items = test_set.groupby('user_id')['item_id'].apply(list)
        train_items = train_set.item_id.unique()

        preds_ranked = []
        true_items = []
        pbar = progressbar.ProgressBar()
        for u in pbar(test_user_items.index):
            true_item = test_user_items[u]
            pos_items = user_items[u]
            neg_items = set(train_items) - set(pos_items)
            neg_sample = np.random.choice(list(neg_items), sample_len-1)
            total_sample = np.append(neg_sample, true_item)
            user_array = np.full(len(total_sample), u, dtype='int32')

            preds = np.hstack(model.predict([user_array, np.array(total_sample)], batch_size=sample_len, verbose=0))
            ids = np.argpartition(preds, -rank_at)[-rank_at:]
            best_ids = np.argsort(preds[ids])[::-1]
            best = total_sample[ids[best_ids]]

            preds_ranked.append(best)
            true_items.append(true_item)

        ranked_df = pd.DataFrame(list(zip(test_user_items.index, preds_ranked, true_items)),
                                 columns=['users', 'pred_items_ranked', 'true_id'])

        return ranked_df