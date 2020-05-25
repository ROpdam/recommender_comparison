import numpy as np
import time
import math
import pandas as pd
import random
import os
from sklearn.metrics import roc_auc_score
import progressbar

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
        self.sample_size = int(params['sample_size'])
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

    def train_model(self, train_set, val_set=[], verbose=1):
        """
        Fit BPR to the train_set, if val_set is provided the AUC metrics will be computed and printed per iteration
        :param train_set: pandas df containing user_id and item_id
        :param val_set: validation pandas df containing user_id and item_id
        :param verbose: 1 means print creating samples, loss per iteration and validation AUC during training
                        (if val_set provided)
        :return: -None: stores values in self.model
        """
        # Seed
        if self.seed > 0:
            np.random.seed(self.seed)

        ## Used for sampling
        self.user_items = train_set.groupby('user_id')['item_id'].apply(list)
        self.train_users = train_set.user_id.unique()
        self.train_items = train_set.item_id.unique()

        if len(val_set) > 0:
            ## Used for validating
            self.val_user_items = val_set.groupby('user_id')['item_id'].apply(list)
            self.val_users = val_set.user_id.unique()

        

        ## Create samples for all iterations
        if verbose == 1:
            print(f'Creating {self.n_iterations} samples of length {self.sample_size}')
        all_uij_samples = self.sample()
        
        self.fit(all_uij_samples, val_set, verbose)
        
        
    def fit(self, all_uij_samples, val_set, verbose):
        """
        """
        # Init user, item matrices and time
        p = np.random.normal(0, .1, (self.total_users, self.nolf))  # users
        q = np.random.normal(0, .1, (self.total_items, self.nolf))  # items
        s = time.time()
        ## Track losses and alphas used
        loss_list = []
        alphas = []
        
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
                    print('iteration:', iteration, ' loss:', round(it_loss, 6), ' val AUC:',val_auc)  
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
        pbar = progressbar.ProgressBar()
        for n in pbar(range(self.n_iterations)):
            uij_samples = []
            for s in range(self.sample_size):
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


    def get_predictions(self, test_set, rank_at=20, stats=True):
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
        test_user_items = test_set.groupby('user_id')['item_id'].apply(list)
        test_users = test_user_items.index

        pred_items_ranked = []
        true_items_list = []

        for u in test_users:
            true_items = []
            for true_item in test_user_items.loc[u]:
                true_items.append(true_item)

            predictions = np.dot(self.model['p'][u], self.model['q'].T)
            ids = np.argpartition(predictions, -rank_at)[-rank_at:]
            best_ids = np.argsort(predictions[ids])[::-1]
            best = ids[best_ids]

            pred_items_ranked.append(best)
            true_items_list.append(true_items)

        ranked_df = pd.DataFrame(columns=['pred_items_ranked', 'true_id'], index=test_users)
        ranked_df['pred_items_ranked'] = pred_items_ranked
        ranked_df['true_id'] = true_items_list
        
        if stats:
            print('Ranking time:', round(time.time() - s, 2))

        return ranked_df
    
    
    def sample_prediction(self, train_set, test_set, sample_len=100, rank_at=20):
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
            preds = np.dot(self.model['p'][u], self.model['q'][total_sample].T)
            ids = np.argpartition(preds, -rank_at)[-rank_at:]
            best_ids = np.argsort(preds[ids])[::-1]
            best = total_sample[ids[best_ids]]

            preds_ranked.append(best)
            true_items.append(true_item)

        ranked_df = pd.DataFrame(list(zip(test_user_items.index, preds_ranked, true_items)),
                                 columns=['users', 'pred_items_ranked', 'true_id'])

        return ranked_df
