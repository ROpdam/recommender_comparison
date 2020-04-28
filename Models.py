import numpy as np
import time
import math
import pandas as pd
import random
import os
from sklearn.metrics import roc_auc_score

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