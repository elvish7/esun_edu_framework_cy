"""
This program is using lightfm dataset package, and designed for ligthgm recommendation.
Author: Jia-Huei Ju (jhjoo@citi.sinica.edu.tw)
"""
import collections
import copy
from collections.abc import Iterable
import pandas as pd
import numpy as np
import logging
import scipy.sparse as sp
from utils import concat_feature, aggregate_fn, npmapping
from data_matrix import COOMatrix 
from data_matrix import NETList

class RecDataset:
    def __init__(self,
                 train_interaction_instance, 
                 eval_interaction_instance, 
                 test_interaction_instance=None):
        """
        Args:
            train_interaction_instance: the object class of training interactions.
            eval_interaction_instance: the object class of evaluation interactions.
            user_instance: the object class of suers.
            item_instance: the object class of items.
        """

        # [TODO] evaluation interaction split
        self.splits = ['train'] 
        self.splits += ['eval'] if eval_interaction_instance else []
        self.splits += ['test'] if test_interaction_instance else []

        # instanace of objects
        self.entity = {}
        self.entity['train'] = train_interaction_instance
        self.entity['eval'] = eval_interaction_instance
        self.entity['test'] = test_interaction_instance

        # mapping fundcion
        self.user_id_mapping = collections.OrderedDict()
        self.item_id_mapping = collections.OrderedDict()
        self.user_feature_mapping = collections.OrderedDict()
        self.item_feature_mapping = collections.OrderedDict()
        ## reversed mapping function
        self.user_idx_mapping = None
        self.item_idx_mapping = None

        # building sparse matrix by the interaction data (list of tuple)
        # (1) initialize the matrices dictionary
        # (2) build the interaction matrix
        # (3) build the user/item id matrix
        # (4) initialize the cold-start dictionary
        self.matrices = {}
        self.cold_start = {}

        for split in self.splits:
            self.matrices.update({split: {}})
            self.cold_start.update({split: {}})

    def _mapping_assertion(self):
        if len(self.user_id_mapping) != 0:
            logging.warning("user id mapping had been build, mapping won't be initialized.")
        if len(self.item_id_mapping) != 0:
            logging.warning("Item id mapping had been build, mapping won't be initialized.")

    def fit(self, 
            user_instance, 
            item_instance,
            user_features=True, 
            item_features=True, 
            user_cold_start=None, 
            item_cold_start=None):
        """fit the unique users and items, and their meta features to separated mapping functions.

        Args:
            user_instance: 'User()' entity
            item_instance: 'Item()' entity
        """
        self._mapping_assertion()

        # maybe calculate cold-start here.
        for user_id in user_instance.get_id_list():
            self.user_id_mapping.setdefault(user_id, len(self.user_id_mapping))
        for item_id in item_instance.get_id_list():
            self.item_id_mapping.setdefault(item_id, len(self.item_id_mapping))

        if user_features:
            for u_feature in user_instance.get_feature_list():
                self.user_feature_mapping.setdefault(u_feature, len(self.user_feature_mapping))
        if item_features:
            for i_feature in item_instance.get_feature_list():
                self.item_feature_mapping.setdefault(i_feature, len(self.item_feature_mapping))
    
    def fit_reversed(self):
        self.user_idx_mapping = {v:k for (k, v) in self.user_id_mapping.items()}
        self.item_idx_mapping = {v:k for (k, v) in self.item_id_mapping.items()}

    def build_feature_matrix(self,
                             split, 
                             user_instance=None, 
                             item_instance=None):
        """
        function building user/item's meta to the feature matrix.

        1) Building ID identity matrix of unique users and items.
        2) Building meta features matrix of users and items. (this can be used as pure-meta cf)
        3) combine ID+feature as the final user-item matrix.

        Args:
            user_instance: User() object
            item_instance: Item() object

        [TODO] make the get data more easily, follow the way of interaction building
        """
        # 1) building id matrix of user and items
        self.matrices[split]['user_ids'] = sp.identity(len(self.user_id_mapping))
        self.matrices[split]['item_ids'] = sp.identity(len(self.item_id_mapping))

        # 2) Building feature matrix 
        ## User metas
        feature_matrix = COOMatrix(
                shape=(len(self.user_id_mapping), len(self.user_feature_mapping)),
                dtype=np.float32
        )
        users = user_instance.get_data()
        if isinstance(users, dict):
            user_ids_indexed = npmapping(users['ids'], self.user_id_mapping)
            user_features_indexed = npmapping(users['metas'], self.user_feature_mapping)
            n_ids, n_metas = user_features_indexed.shape
            feature_matrix.append_all(
                    all_i=np.repeat(user_ids_indexed, n_metas),
                    all_j=user_features_indexed.reshape(-1),
                    all_v=np.ones(n_ids * n_metas)
            )
        self.matrices[split]['user_features'] = feature_matrix.tocoo()

        ## Item metas
        items = item_instance.get_data()
        feature_matrix = COOMatrix(
                shape=(len(self.item_id_mapping), len(self.item_feature_mapping)),
                dtype=np.float32
        )
        if isinstance(items, dict):
            item_ids_indexed = npmapping(items['ids'], self.item_id_mapping)
            item_features_indexed = npmapping(items['metas'], self.item_feature_mapping)
            n_ids, n_metas = item_features_indexed.shape
            feature_matrix.append_all(
                    all_i=np.repeat(item_ids_indexed, n_metas),
                    all_j=item_features_indexed.reshape(-1),
                    all_v=np.ones(n_ids * n_metas)
            )
        self.matrices[split]['item_features'] = feature_matrix.tocoo()
        
        # 3) combine ID with meta features 
        self.__combine_id_and_feature_matrix(split=split)

    def __combine_id_and_feature_matrix(self, split):
        self.matrices[split]['user_ids_and_features'] = sp.hstack(
                [self.matrices[split]['user_ids'], self.matrices[split]['user_features']]
        )
        self.matrices[split]['item_ids_and_features'] = sp.hstack(
                [self.matrices[split]['item_ids'], self.matrices[split]['item_features']]
        )

    def build_auto_matrix(self, 
                          split, 
                          data_name=None,
                          dict_key=None,
                          include_mixed_regular=False,
                          heuristic_type=None,
                          split_eval=None,
                          return_matrix=False):
        """
        Args:
            data_name: (1) interaction/weights (2) regular (3) heuristic (types)
            dict_key:  (1) ratings/None (2) regular (3) ratings (types)
        """

        if data_name == 'interactions' or data_name == 'weights':
            data = self.entity[split].get_data()
        if data_name == 'regular':
            data = self.entity[split].get_regular_plan_data(
                    include_mixed_regular=include_mixed_regular
            )
        if data_name == 'heuristic':
            data = self.entity[split].get_heuristic_data(
                    heuristic_by=heuristic_type,
                    expand_users=npmapping(self.cold_start[split_eval]['users'],
                                           self.user_id_mapping,
                                           reverse=True),
            )

        # Build empty matrix
        data_matrix = COOMatrix(
                shape=(len(self.user_id_mapping), len(self.item_id_mapping)),
                dtype=np.float32
        )
        # Append elements by coordinate
        data_matrix.append_all(
                all_i=npmapping(data['users'], self.user_id_mapping),
                all_j=npmapping(data['items'], self.item_id_mapping),
                all_v=data[dict_key] if data_name != 'weights' else np.repeat(1, len(data['users']))
        )
        if return_matrix:
            return data_matrix.tocoo()
        else:
            print(f"Building matrices from '{data_name}' data by key '{dict_key}'")
            self.matrices[split][data_name] = data_matrix.tocoo()


    def get_intersection_mask(self,
                              base='eval', 
                              reference='train',
                              verbose=False, 
                              return_diff=False):
        """
        Function for add the diff interaction matrxi from split to train

        Args:
            base: (str or sparse matrix), the base interaction to-be compared.
            reference: (str or sparse matrix), the compared interaction..

        Retruns:
            same_matrix: the subset of base sparse 2d array, which overlapped with reference.
                - (Regular to Regular): Regular products and easiest -->ignore
                - (Regular to Solo): Degraded products --> (exclude or include ?)
                - (solo to Regular): Upgraded products --> Remind user to sign the contract.
                - (solo to Solo): Continuous subscribing --> Remind user to subscribe then.
            diff_matrix: the subset of base sparse 2d array, which differ from reference.
        """
        def casting_to_arr(x):
            if isinstance(x, str):
                x_prime = self.matrices[x]['interactions'].toarray()
            if isinstance(x, sp.coo_matrix):
                x_prime = x.toarray()
            if isinstance(x, np.ndarray):
                x_prime = x
            return x_prime

        base, reference = casting_to_arr(base), casting_to_arr(reference)
        intersection = np.multiply(base!=0, reference!=0)
        same_matrix = np.multiply(base, intersection)
        diff_matrix = np.multiply(base, ~intersection)

        if verbose:
            print(f"DiffMatrix({{\
                    \n\tbase: {np.count_nonzero(base)}\
                    \n\treference: {np.count_nonzero(reference)}\
                    \n\tintersection: {intersection.sum()}\
                    \n\tsame: {np.count_nonzero(same_matrix)}\
                    \n\tdifference: {np.count_nonzero(diff_matrix)}\
                    \n}})")

        return intersection, (same_matrix, diff_matrix)

    def build_interaction_network(self, 
                                  split, 
                                  dict_key='ratings', 
                                  output_file='networks.txt'):
        
        data = self.entity[split].get_data()

        # Build empty networks
        data_network = NETList(
                shape=(len(self.user_id_mapping), len(self.item_id_mapping)),
                dtype=np.float32,
        )

        # Append elements by coordinate
        data_network.append_all(
                all_i=npmapping(data['users'], self.user_id_mapping),
                all_j=npmapping(data['items'], self.item_id_mapping),
                all_v=data['ratings']
        )

        data_network.totxt(output_file)

        return output_file

    def calculate_cold_start(self):
        """
        Function for calculate the cold-start items and users based on the training interactions,
        [TODO] Mapping the index back to the user/item id
        """
        # calculate non-interacted users/items (from training interactions)
        non_interacted_users_train = np.where(
                self.matrices['train']['interactions'].toarray().sum(axis=1) == 0
        )[0]
        non_interacted_items_train = np.where(
                self.matrices['train']['interactions'].toarray().sum(axis=0) == 0,
        )[0]
        self.cold_start['train'] = {
                'users': non_interacted_users_train,
                'items': non_interacted_items_train
        }

        # calculate cold-start users/items of "eval/test..." according to training interactions
        for key, matrices in self.matrices.items():
            if key != 'train':
                non_interacted_users = np.where(
                        matrices['interactions'].toarray().sum(axis=1) == 0
                )[0]
                non_interacted_items = np.where(
                        matrices['interactions'].toarray().sum(axis=0) == 0
                )[0]

                self.cold_start[key] = {
                        'users': np.intersect1d(non_interacted_users, non_interacted_users_train),
                        'items': np.intersect1d(non_interacted_items, non_interacted_items_train)
                }


        print(f"**** cold-start users and item in this dataset **** \
                \n - Number of non-interacted users: {len(self.cold_start['eval']['users'])} \
                \n - Number of non-interacted items: {len(self.cold_start['eval']['items'])}")




    def __repr__(self):
        return f"RecDataset({{\
                \n\tsplit: {self.splits},\
                \n\tnum_users: {len(self.user_id_mapping)} \
                \n\tnum_items: {len(self.item_id_mapping)}\
                \n\tnum_user_features: {len(self.user_feature_mapping)}\
                \n\tnum_item_features: {len(self.item_feature_mapping)}\
                \n}})"

class Interaction: 
    def __init__(self, 
                 user_id_column_name='cust_no', 
                 item_id_column_name='wm_prod_code', 
                 utility='sum'):
        self.df = None
        self.df_agg = None
        self.utility = utility
        self.user_id_column_name = user_id_column_name
        self.item_id_column_name = item_id_column_name
        self.user_id_list = []
        self.item_id_list = []

        # Fund-rec specific parameters [USERS]
        self.df_agg_regular = None

    def __repr__(self):
        NULL = None
        n_users = len(self.df[self.user_id_column_name].unique())
        n_items = len(self.df[self.item_id_column_name].unique())

        if self.df_agg is None:
            return f"Interaction({{\
                    \n\tstart date: {NULL}\
                    \n\tend date: {NULL}\
                    \n\tnum users: {n_users} \
                    \n\tnum items: {n_items}\
                    \n\tnum interaction: {len(self.df)} \
                    \n\taverage subscribing times: {len(self.df)/n_users}\
                    \n\taverage regular-subscribing items: {len(self.df[self.df.deduct_cnt > 0])/n_users}\
                    \n\taverage solo-subscribing items: {len(self.df[self.df.deduct_cnt == 0])/n_users}\
                    \n}})"
        else:
            return f"Interaction({{\
                    \n\tstart date: {NULL}\
                    \n\tend date: {NULL}\
                    \n\tnum users: {n_users} \
                    \n\tnum items: {n_items}\
                    \n\tnum interaction: {len(self.df)} \
                    \n\tnum interaction (aggregated): {len(self.df_agg)} \
                    \n\taverage subscribing times: {len(self.df)/n_users}\
                    \n\taverage regular-subscribing times: {len(self.df[self.df.deduct_cnt > 0])/n_users}\
                    \n\taverage solo-subscribing times: {len(self.df[self.df.deduct_cnt == 0])/n_users}\
                    \n\taverage subscribed items: {len(self.df_agg)/n_users}\
                    \n\taverage regular-subscribed ratio: {self.df_agg_regular.r_regular_per_user.sum()/n_users}\
                    \n\taverage solo-subscribed ratio: {self.df_agg_regular.r_solo_per_user.sum()/n_users}\
                    \n\taverage mixed-subscribed ratio: {self.df_agg_regular.r_mixed_per_user.sum()/n_users}\
                    \n}})"


    def truncate(self, cust_avail_list=None, prod_avail_list=None):
        """ For truncating the cold-start user/item.  """
        if isinstance(cust_avail_list, (np.ndarray, list)):
            self.df_agg = self.df_agg[self.df_agg[self.user_id_column_name].isin(cust_avail_list)]
        if isinstance(prod_avail_list, (np.ndarray, list)):
            self.df_agg = self.df_agg[self.df_agg[self.item_id_column_name].isin(prod_avail_list)]
        return self

    def _aggregate(self, criteria, regular=False):
        """Aggregation of the re-subscribe transactions.

        Args: 
            criteria: aggregation type of the repeated transcations, 
                default is None, which followed the initialized settings by 'self.rating'
            regular: aggregate with regular label as the "regular ratio", 
                which "1" indicdates the purely "regular", "0" indicate the purely "solo", others are the mixed.
        """
        if criteria in ['sum', 'mean', 'max', 'min', 'freq', 'count', 'binary']:
            # Simple rating aggregated function
            f = criteria 
            if criteria == 'binary':
                f = (lambda x: 1)
            elif criteria == 'freq':
                f = 'count'
            self.df_agg = self.df.groupby([self.user_id_column_name, self.item_id_column_name]).agg(
                    prod_rating=('txn_amt', f)
            ).reset_index()

        else:
            # Complex rating aggregated function
            print("Using the self-defined rating function....it'll take a while...")
            agg_fn = aggregate_fn(criteria)
            self.df_agg = self.df.groupby([self.user_id_column_name, self.item_id_column_name]).apply(
                    func=agg_fn
            ).reset_index()


    def _preprocess_time(self, unit='day'):
        for col in self.df.columns:
            if '_dt' in col:
                if unit == 'month':
                    self.df[col].apply(lambda x: "-".join(x.split('-')[0:2]))
                elif unit == 'year':
                    self.df[col].apply(lambda x: x.split('-')[0])

                self.df[col] = self.df[col].apply(pd.Timestamp)

    def preprocess(self, 
                   min_required_items=0, 
                   min_required_users=0,
                   used_txn_info='all',
                   min_time_unit='day',
                   utility=None,
                   do_regular=False):
        """
        The preprocessing pipeline function for the corresponding fund transatcions.
        It can be customized depend on different purpose. The preprocess detail is as follow,

        1) Valid transaction, filtering by 
            - min_required_items: the minimun of items that should be considered in interaction matrix.
            - max_required_items: the maximum of items that should be considered in interaction matrix.
        2) detail transaction information used (potentially useful)
            - txn_amt: transaction amount.
            - txn_dt: transaction date.
            - deduct_cnt: subsribing times.
        3) Utility function of rate, rate the user-item pair includes types as follow,
            - first: baseline, amount of first transaction.
            - binary: baseline, buy or not buy.
            - sum: summation of subscribing amount of this fund.
            - mean: average of subscirbing amount of this fund.
            - freq: frequency of subsribing of this fund.
            - max: maximun of subsribing amount of this fund.
            - min: minimum of subsribing amount of this fund.
        4) Variables updates: By adopting the preprocessing (agg includes), 
            user/item id lists are provided.
        5) Aggregate the heurisitc statistics of transactions.
        """
        # ***** 1  *****
        self.df.dropna(inplace=True) # [TODO] what transactions are removed? maybe we should keep it.
        self.df = self.df[self.df.groupby("cust_no")["cust_no"].transform(len) > \
                min_required_users] 
        self.df = self.df[self.df.groupby("wm_prod_code")["wm_prod_code"].transform(len) > \
                min_required_items] 

        # ***** 2  *****
        if used_txn_info != 'all':
            self.df = self.df[['cust_no', 'wm_prod_code'] + used_txn_info] 
        self._preprocess_time(min_time_unit)

        # ***** 3  *****
        self._aggregate(criteria=self.utility if utility is None else utility, regular=do_regular)
        if do_regular:
            self._calculate_regular_plan(regular_cnt_col_name='deduct_cnt')

        # ***** 4  *****
        self.user_ids = self.df_agg[self.user_id_column_name].unique() 
        self.item_ids = self.df_agg['wm_prod_code'].unique()

        # ***** 5  *****
        self.df_agg_heuristic = \
                self.df.groupby([self.user_id_column_name, self.item_id_column_name]).agg(
                        frequency=('txn_amt', len), 
                        volume=('txn_amt', sum)
                )

        return self

    def get_user_list(self, aggregated=True):
        assert self.user_ids is None, \
                print("The preprocessing step should be done in advnace.")
        return self.user_ids if aggregated else self.df.cust_no.unique()

    def get_item_list(self, aggregated=True):
        assert self.item_ids is None, \
                print("The preprocessing step should be done in advnace.")
        return self.item_ids if aggregated else self.df.wm_prod_code.unique()

    def get_data(self):
        interaction_users = self.df_agg[self.user_id_column_name].values
        interaction_items = self.df_agg[self.item_id_column_name].values
        interaction_ratings = self.df_agg['prod_rating'].values

        return {'users': interaction_users, 
                'items': interaction_items, 
                'ratings': interaction_ratings}

    def get_heuristic_data(self, 
                           heuristic_by='random',
                           expand_users=None,
                           expand_items=None):
        """
        function for generating the heuristic interaction, 
        Firstly, aggregate the proxy dataframe df_agg (differ from self.df_agg),
        then transform the aggregated amomt based on the heuristic approach.
        Steps:
            (1) Specify the random sampling probabilities of products
            (2) Select the heuristic recommendation type, include random, volume, frequency
                - random_warm: choose 10 random products among the subscribed products (warm).
                - volume: rank the rating by subscribing amount of product.
                - frequency: rank the rating by subsrcibing times of product.
                - [TODO] average-volume 
            (3) exapnd the users with cold-start, append them randomly 
            (4) [DEBUG] Align the user/item available list to interactions.
            (5) transforme the df into the numpy array (to-be-built into matrices)

        Args:
            heuristic-by: Type of heuristic approach for valida interaction, as follow:
            expand_users: Apply the heurisitic on the cold-start users.
            expand_items: Apply the heurisitic on the cold-start items.
        """

        # ***** 1  *****
        if 'weighted' in heuristic_by:
            # [CONCERN] only calculate on the products in known interaction (warm)
            p_items = self.df.groupby('wm_prod_code').agg(
                    frequency=('txn_amt', len)
            ).reindex(self.item_ids)['frequency'].values
        else:
            p_items = None

        # ***** 2  *****
        if 'random' in heuristic_by:
            users = np.repeat(self.user_ids, 10)
            items = np.array([])
            for i in range(len(self.user_ids)):
                items = np.append(items, np.random.choice(self.item_ids, 10, p=p_items))

            df_agg_heuristic = pd.DataFrame({
                self.user_id_column_name: users, 
                self.item_id_column_name: items, 
                heuristic_by: np.tile(list(range(10, 0, -1)), len(self.user_ids))
            })
        elif heuristic_by in 'volume frequency':
            df_agg_heuristic = self.df_agg_heuristic.groupby('cust_no')[heuristic_by].rank(
                    'dense', ascending=True
            ).reset_index()

        # ***** 3  *****
        if isinstance(expand_users, list):
            for u in expand_users:
                df_agg_heuristic = df_agg_heuristic.append(
                        pd.DataFrame(
                            [[u, i, 10-rank] for rank, i in \
                                    enumerate(np.random.choice(self.item_ids, 10, p=p_items))],
                            columns=[self.user_id_column_name, 
                                     self.item_id_column_name, 
                                     heuristic_by],
                            ignore_index=True
                        )
                )

        # ***** 4  *****
        df_agg_heuristic = df_agg_heuristic[df_agg_heuristic[self.user_id_column_name].isin(
            self.df_agg[self.user_id_column_name].unique()
        )]
        df_agg_heuristic = df_agg_heuristic[df_agg_heuristic[self.item_id_column_name].isin(
            self.df_agg[self.item_id_column_name].unique()
        )]

        # ***** 5  *****
        interaction_users = df_agg_heuristic[self.user_id_column_name].values
        interaction_items = df_agg_heuristic[self.item_id_column_name].values
        interaction_ratings = df_agg_heuristic[heuristic_by].values

        return {'users': interaction_users, 
                'items': interaction_items, 
                'ratings': interaction_ratings}

    def _calculate_regular_plan(self, regular_cnt_col_name='deduct_cnt'):
        # regular subscribed among all unqiue-items (per user)
        df_regular = self.df.groupby([self.user_id_column_name, self.item_id_column_name]).agg(
                r_regular=(regular_cnt_col_name, lambda x: sum(x > 0)/len(x) ),
                r_solo=(regular_cnt_col_name, lambda x: sum(x == 0)/len(x) ),
                counts=(regular_cnt_col_name, len),
        ).reset_index()

        # merge the rating
        self.df_agg = self.df_agg.merge(df_regular, on=[self.user_id_column_name, self.item_id_column_name])

        # regular subscribed among all unqiue-interaction (per user)
        self.df_agg_regular = self.df_agg.groupby([self.user_id_column_name]).agg(
                r_regular_per_user=('r_regular', lambda x: sum(x == 1)/len(x) ),
                r_solo_per_user=('r_regular', lambda x: sum(x == 0)/len(x) ),
                r_mixed_per_user=('r_regular', lambda x: sum( (x > 0) & ( x < 1) )/len(x) ),
        ).reset_index()
        # self.r_repeated = 0
        # self.r_repeated_agg = 0

    def get_regular_plan_data(self, include_mixed_regular=False):
        """
        Retrieve the subset of the dataframe by the "PURELY REGUALLY" subscribed items
        """
        if include_mixed_regular:
            df = self.df_agg[self.df_agg.r_regular != 1]
        else:
            df = self.df_agg[self.df_agg.r_regular == 1]

        regular_users = df[self.user_id_column_name].values
        regular_items = df[self.item_id_column_name].values
        regular_labels = np.repeat(1, len(df))

        return {'users': regular_users, 
                'items': regular_items, 
                'regular': regular_labels}



    def read_csv(self, path):
        self.df = pd.read_csv(path, index_col=0)

class User:
    def __init__(self, id_column_name='cust_no'):
        self.df = None
        self.id_column_name = id_column_name

    def preprocess(self, 
                   time_alignment,
                   cust_filter_list=None,
                   na_preps=None,
                   meta_preps={}):
        """
        The preprocessing pipeline function for the corresponding customer profiles.
        It can be customized depend on different purpose. The preprocess detail is as follow,

        1a) data filtering on metas (columns)
            - meta_preps.keys(): includes all used metas, 
        1b) data filtering on row (records)
            - null row removal: remove the row with null value.
            - time alignment: specify the 'time_aligned=True' for aggregating redundant users by the latest one.
            - custotmer alignment: specify 'cust_filter_list=[uid1, uid2, ..]' to align user id in dataframe.
        2) feature engineering, specify the {'<column name>', <function>}
            - age
            - cust_vintage
            - gender_code
            - income_range_code
            - risk_type_code
        3) feature rebuild: meta concat with value by colon.
        [TODO] Replace the column name call by the class variables
        """
        # ***** 1a *****
        self.df = self.df[['cust_no'] + list(meta_preps.keys()) ]

        # ***** 1b *****
        for col, cnt in self.df.count().to_dict().items():
            if cnt < len(self.df) * 0.5:
                self.df.drop(columns=col, inplace=True)
                print(f"Column {col} is removed, insufficient valida data: {cnt}")
            elif cnt < len(self.df):
                filled_value = na_preps[col](self.df[col])
                self.df.fillna({col: filled_value}, inplace=True)
                print(f"{len(self.df)-cnt} NA values in column {col} are replaced by {filled_value}")
        self.df = self.df.sort_values(time_alignment).groupby('cust_no').tail(1)
        if cust_filter_list:
            self.df = self.df[self.df['cust_no'].isin(cust_filter_list)]

        # ***** 2  *****
        for col, prep_function in meta_preps.items():
            self.df[col] = prep_function(self.df[col])

        # ***** 3  *****
        self.df = concat_feature(self.df, off_cols=['cust_no'])

        return self

    def __repr__(self):
        NULL = None
        return f"User({{\
                \n\tnum_users: {len(self.df.cust_no.unique())} \
                \n\tnum_item_metas: {len(self.df.drop(columns=self.id_column_name).columns)} \
                \n\tnum_item_features: {len(self.get_feature_list())} \
                \n}})"

    def get_data(self, return_type='dict'):
        """
        return the user ID list and the user features list according to the order of user ID list
        """
        assert len(self.df[self.id_column_name]) == len(self.df.drop(columns=self.id_column_name)), \
                'The length of user id and feauture sets inconsistent.'
        # print("The user data contains {} customers.".format(len(self.df)))

        if return_type == 'dict':
            user_ids = self.df[self.id_column_name].values
            user_features = self.df.drop(columns=self.id_column_name).values
            return {'ids': user_ids, 'metas': user_features}

    def get_id_list(self):
        return self.df['cust_no'].unique()

    def get_feature_list(self):
        feature_list = np.empty(0)
        for col in self.df.columns:
            if col != 'cust_no':
                feature_list = np.append(feature_list, self.df[col].unique()) 

        return feature_list if len(feature_list) != 0 else []

    def read_csv(self, path):
        self.df = pd.read_csv(path, index_col=0)


class Item:
    def __init__(self, id_column_name='wm_prod_code'):
        self.df = None
        self.id_column_name = id_column_name

    def __repr__(self):
        NULL = None
        return f"Item({{\
                \n\tnum_items: {len(self.df.wm_prod_code.unique())} \
                \n\tnum_item_metas: {len(self.df.drop(columns=self.id_column_name).columns)} \
                \n\tnum_item_features: {len(self.get_feature_list())} \
                \n}})"

    def preprocess(self, 
                   fund_filter_list=None,
                   na_preps={},
                   meta_preps={}):

        """
        The preprocessing pipeline function for the corresponding product data.
        It can be customized depend on different purpose. The preprocess detail is as follow,

        1a) data filtering on metas (columns)
            - meta_preps.keys(): includes all used metas, 
                [NOTE] use the 'identity' for the meta without preprocessing.
        1b) data filtering on row (records)
            - null row removal: remove the row with null value.
            - product alignment: specify 'product_filter_list=[pid1, pid2, ...]' to align item id in dataframe.
        2) feature engineering, specify the {'<column name>', <function>}
            - can_rcmd_ind
            - prod_ccy
            - prod_risk_code
            - prod_detail_type_code
            - mkt_rbot_ctg_ic
        3) feature rebuild: meta concat with value by colon.
        """
        # ***** 1a *****
        self.df = self.df[['wm_prod_code'] + list(meta_preps.keys())]

        # ***** 1b *****
        for col, cnt in self.df.count().to_dict().items():
            if cnt < len(self.df) * 0.5:
                self.df.drop(columns=col, inplace=True)
                print(f"Column '{col}' is removed, insufficient valida data: {cnt}")
        self.df.dropna(inplace=True) 
        if fund_filter_list:
            self.df = self.df[self.df['wm_prod_code'].isin(fund_filter_list)]

        # ***** 2  *****
        for col, prep_function in meta_preps.items():
            self.df[col] = prep_function(self.df[col])

        # ***** 3  *****
        self.df = concat_feature(self.df, off_cols=['wm_prod_code'])

        return self

    def get_data(self, return_type='dict', return_features=True):
        """ return the item ID list and the item features list according to the order of item ID list """

        assert len(self.df[self.id_column_name]) == len(self.df.drop(columns=self.id_column_name)), \
                'The length of item id and feauture sets inconsistent.'
        # print("The item data contains {} different funds/products.".format(len(self.df)))

        if return_type == 'dict':
            item_ids = self.df[self.id_column_name].values
            item_features = self.df.drop(columns=self.id_column_name).values
            return {'ids': item_ids, 'metas': item_features}

    def get_id_list(self):
        return self.df['wm_prod_code'].unique()

    def get_feature_list(self):
        feature_list = np.empty(0)
        for col in self.df.columns:
            if col != 'wm_prod_code':
                feature_list = np.append(feature_list, self.df[col].unique()) 

        return feature_list if len(feature_list) != 0 else []

    def read_csv(self, path):
        self.df = pd.read_csv(path, index_col=0)
