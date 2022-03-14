import collections
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils import npmapping

class Evaluation:
    
    def __init__(self,
                 dataset_instance,
                 evaluate_train,
                 evaluate_diff,
                 evaluate_regular,
                 heuristics,):

        self.dataset = dataset_instance

        # truth and predictions
        self.truth = collections.defaultdict(dict)
        self.pred = collections.defaultdict(dict)
        self.heuristics = {}

        # results records
        self.results_per_user = collections.defaultdict(dict)
        self.heuristics_per_user = {}
        for _type in heuristics:
            self.heuristics[_type] = collections.defaultdict(dict)
            self.heuristics_per_user[_type] = collections.defaultdict(dict)

        self.eval_diff = evaluate_diff
        self.eval_regular = evaluate_regular
        self.eval_train = evaluate_train
        self.intersection_mask = {}

        self.results_df = pd.DataFrame()

        # calculate the ground truth information
        self.build_truth_ranksets()

    def _load_data(self, interactions,  topn=None):
        """Build for loading interaction data 
        then sorting items interacited by rating (or predicted scores)

        [NOTE]: 3 type of input interactions:
            - Training interaction matrices
            - Prediction interaction numpy 2d array (with full ranking)
            - heuristic selected ranks of numpy 2d array.
        """
        if isinstance(interactions, sp.coo_matrix):
            interactions = interactions.toarray()

        rankdict = collections.defaultdict(list)
        # 1) sort the array row-wise (user-wise)
        valid_interactions = np.count_nonzero(interactions, axis=1)
        interactions_user_wise_sorted = np.fliplr(np.argsort(interactions, axis=1))[:, :topn]

        print(valid_interactions[:10])
        print(valid_interactions[-10:-1])

        for user_idx, item_idx_list in enumerate(interactions_user_wise_sorted):
            valid_item_idx_list = item_idx_list[:valid_interactions[user_idx]]
            if len(valid_item_idx_list) >= 1:
                rankdict[user_idx] = [item_idx for item_idx in valid_item_idx_list]

        return rankdict

    def build_truth_ranksets(self, split='eval'):
        """ The function call for process the dataset into the interaction data, includes:
        Args:
            train/eval/test: the interactions. 
                - exploit/explore: the "repeated"/"new" interactions
        """
        if self.eval_train:
            self.truth['train'] = self._load_data(self.dataset.matrices['train']['interactions'])

        self.truth[f'{split}'] = self._load_data(self.dataset.matrices[f'{split}']['interactions'])
        if self.eval_diff:
            intersection, (exploit_subset, explore_subset) = self.dataset.get_intersection_mask(
                    base=self.dataset.matrices[f'{split}']['interactions'],
                    reference=self.dataset.matrices['train']['interactions'],
                    verbose=True,
                    return_diff=True,
            )
            self.intersection_mask[f'{split}'] = intersection
            self.truth[f'{split}-exploit'] = self._load_data(exploit_subset)
            self.truth[f'{split}-explore'] = self._load_data(explore_subset)

            regular_subset = self.dataset.build_auto_matrix(
                    split='train',
                    data_name='regular',
                    dict_key='regular',
                    include_mixed_regular=False,
                    return_matrix=True
            )
            if self.eval_regular:
                intersection, (exploit_regular_subset, exploit_non_regular_subset) = self.dataset.get_intersection_mask(
                        base=exploit_subset,
                        reference=regular_subset,
                        verbose=True,
                        return_diff=True,
                )

                self.intersection_mask[f'{split}-exploit'] = intersection
                self.truth[f'{split}-exploit-R2*'] = self._load_data(exploit_regular_subset)
                self.truth[f'{split}-exploit-S2*'] = self._load_data(exploit_non_regular_subset)

    def build_pred_ranksets(self, prediction=None, split='eval'):
        self.pred[f'{split}'] = self._load_data(prediction)
        if self.eval_train:
            self.pred['train'] = self.pred[f'{split}']

        if self.eval_diff:
            prediction_exploit = prediction * self.intersection_mask[split]

            self.pred[f'{split}-explore'] = self._load_data(
                    prediction * ~self.intersection_mask[split]
            )
            self.pred[f'{split}-exploit'] = self._load_data(
                    prediction_exploit
            )
            if self.eval_regular:
                self.pred[f'{split}-exploit-R2*'] = self._load_data(
                        prediction_exploit * self.intersection_mask[f'{split}-exploit']
                )
                self.pred[f'{split}-exploit-S2*'] = self._load_data(
                        prediction_exploit * ~self.intersection_mask[f'{split}-exploit']
                )

    def build_heuristic_ranksets(self, _type, split='eval'):

        heuristic_matrix = self.dataset.build_auto_matrix(
                split='train',
                data_name='heuristic',
                dict_key='ratings',
                heuristic_type=_type,
                split_eval=split,
                return_matrix=True
        )
        heuristic_matrix = heuristic_matrix.toarray()
        self.heuristics[_type][f'{split}'] = self._load_data(heuristic_matrix)

        if self.eval_train:
            self.heuristics[_type]['train'] = self.heuristics[_type][f'{split}']

        if self.eval_diff:
            heuristic_exploit = heuristic_matrix * self.intersection_mask[f'{split}']

            self.heuristics[_type][f'{split}-explore'] = self._load_data(
                    heuristic_matrix * ~self.intersection_mask[split]
            )
            self.heuristics[_type][f'{split}-exploit'] = self._load_data(
                    heuristic_exploit
            )
            if self.eval_regular:
                self.heuristics[_type][f'{split}-exploit-R2*'] = self._load_data(
                        heuristic_exploit * self.intersection_mask[f'{split}-exploit']
                )
                self.heuristics[_type][f'{split}-exploit-S2*'] = self._load_data(
                        heuristic_exploit * ~self.intersection_mask[f'{split}-exploit']
                )

    def eval(self, 
             prediction, 
             metrics='p@5', 
             ranking=True,
             exclude_cold_start=False,
             max_positive_truth=None,
             do_heuristic=False):
        """
        Main evaluating function, which includes the different type of metrics,
        and their corresponding parameter settings.

        Args:
            ranking: make the predicted score to ranks (order).
            prediction: the (n_user, n_item) matrix with predicted ranks or rating scores.
            metrics: the evaluation metrics, only the "ungraded" judging now.
                - precision, average_precision, reciprocal_rank
            exclude_cold_start: setting False (default). But if using meta-feature, 
                the cold-start problem should not be ignored.
            max_positive_truth: the number of valid interactions for each user-item pair.
                set None by default, indicates all the truth items are included.
        """
        if ranking:
            prediction = prediction.argsort().argsort()

        self.build_pred_ranksets(prediction=prediction, split='eval')

        metrics = [metrics] if isinstance(metrics, str) else metrics

        for eval_set in self.truth:
            for metric in set(metrics):
                EVAL_FUNCTION(metric)(
                        result_dict=self.results_per_user[(metric, eval_set)], 
                        truth_dict=self.truth[eval_set], 
                        pred_dict=self.pred[eval_set],
                        n=max_positive_truth 
                )

        # do the heuristic approach with each groundtruth
        if do_heuristic:
            for heuristic_type in self.heuristics:
                self.build_heuristic_ranksets(_type=heuristic_type, split='eval')
                for eval_set in self.truth:
                    for metric in set(metrics):
                        EVAL_FUNCTION(metric)(
                                result_dict=\
                                        self.heuristics_per_user[heuristic_type][(metric, eval_set)], 
                                truth_dict=self.truth[eval_set], 
                                pred_dict=self.heuristics[heuristic_type][eval_set],
                                n=max_positive_truth 
                        )

    def get_average_eval(self, steps=False, do_heuristic=False):
        results_dict = collections.defaultdict(list)
        results = "==========================================\n"
        for eval_set in self.truth:
            n_users = len(self.truth[eval_set])
            n_items = sum([len(item_list) for item_list in self.truth[eval_set].values()])
            results += " [{:<20}]: {:<7} ({:.4f})\n".format(
                    eval_set, 
                    n_users,
                    n_items / n_users
            )
        results += "==========================================\n"
        for key in sorted(self.results_per_user, key=lambda x: x[0]):
            metric, eval_set = key
            average = np.mean(list(self.results_per_user[key].values()))
            results += "  {:<5} on {:<20}: {:.4f}\n".format(
                    metric, 
                    eval_set,
                    average
            )
            # prepare for final table
            results_dict[f'{metric}-{eval_set}'].append(average)

        # add the record of traing steps
        self.results_df = self.results_df.append(
                pd.DataFrame(data=results_dict, index=[str(steps)])
        )

        # do the heuristics
        if do_heuristic:
            for _type in self.heuristics_per_user:
                results_dict = collections.defaultdict(list)
                for key in sorted(self.heuristics_per_user[_type], key=lambda x: x[0]):
                    metric, eval_set = key
                    average = np.mean(list(self.heuristics_per_user[_type][key].values()))
                    results_dict[f'{metric}-{eval_set}'].append(average)

                # add the record of traing steps
                self.results_df = self.results_df.append(
                        pd.DataFrame(data=results_dict, index=[str(_type)])
                )

        return results + "================================="

    def save_recommedation_to_file(self, file_path='recommendation_detail.tsv'):
        self.dataset.fit_reversed() # get the reversed mapping fuinction
        f = open(file_path, 'w')
        for i, user_idx in enumerate(self.truth['eval']):
            user_id = self.dataset.user_idx_mapping[user_idx]
            # list the old product re-subsribed 
            truth_old_item_id_list = npmapping(
                    self.truth['eval-exploit'][user_idx], 
                    self.dataset.item_idx_mapping
            )
            pred_old_item_id_list = npmapping(
                    self.pred['eval-exploit'][user_idx], 
                    self.dataset.item_idx_mapping
            )
            for i, (t, p) in enumerate(zip(truth_old_item_id_list, pred_old_item_id_list)):
                f.write(user_id+'\tEXPLOIT\t'+str(i+1)+'\t'+str(t)+'\t'+str(p)+'\n')

            # list the new product re-subsribed 
            truth_new_item_id_list = npmapping(
                    self.truth['eval-explore'][user_idx], 
                    self.dataset.item_idx_mapping
            )
            pred_new_item_id_list = npmapping(
                    self.pred['eval-explore'][user_idx], 
                    self.dataset.item_idx_mapping
            )
            for i, (t, p) in enumerate(zip(truth_new_item_id_list, pred_new_item_id_list)):
                f.write(user_id+'\tEXPLORE\t'+str(i+1)+'\t'+str(t)+'\t'+str(p)+'\n')

            # easy-to-read separation
            if i > 0 and i % 100 == 0:
                f.write('\n')

    def save_results_to_file(self, file_path='evaluation.csv'):
        self.results_df.to_csv(file_path)

    def __repr__(self):
        """
        [TODO] List down the each evaluation set's statisitcs, e.g. number of users.
        """
        NULL=None
        return f"Evaluation({{\
                \n\tevaluating sets: {self.truth.keys()},\
                \n}})"

def EVAL_FUNCTION(metric):
    """
    Function of setting evaulation metric for rec sys judgements.

    Args:
        metric: the evaluation measurement, incldues (preicsion at k)

    Returns:
        f: the function call, includes the arguments:
            result_dict: the dictionary for storing metric evaluation scores. (per user)
            truth_dict: the truth interaction. (per user)
            pred_dict: the predicted interaction. (per user)
            n: the 'valud' relevant judgements.
                    (but for the efficiency, define in the 'load_data(topn=n)' would be better.
            k: cut off at the predicted rankiing list.
    """
    try:
        metric, cut_off = metric.split("@")
    except:
        cut_off = None

    # 1. Precision
    def precision(result_dict, 
                  truth_dict, 
                  pred_dict, 
                  n=None, 
                  k=int(cut_off)):

        for user_idx in truth_dict:
            # preicisionn
            truth_item_idx_list = [i for i in truth_dict[user_idx][:n]]
            pred_item_idx_list = [i for i in pred_dict[user_idx][:k]]
            result_dict[user_idx] = \
                len( set(truth_item_idx_list) & set(pred_item_idx_list) ) / k

    # 2. Average Precision
    def average_precision(result_dict, 
                          truth_dict, 
                          pred_dict, 
                          n=None, 
                          k=int(cut_off)):

        for user_idx in truth_dict:
            truth_item_idx_list = [i for i in truth_dict[user_idx][:n]]
            pred_item_idx_list = [i for i in pred_dict[user_idx][:k]]

            # average preicision
            ap = 0
            n_hits = 0
            for order, item_idx in enumerate(pred_item_idx_list, start=1):
                if item_idx in truth_item_idx_list:
                    n_hits += 1
                    ap += (n_hits / order)

            result_dict[user_idx] = ap / min(k, len(truth_item_idx_list))

    # 3. Reciprocal Rank
    def reciprocal_rank(result_dict, 
                        truth_dict, 
                        pred_dict, 
                        n=None, 
                        k=int(cut_off)):

        for user_idx in truth_dict:
            truth_item_idx_list = [i for i in truth_dict[user_idx][:n]]
            pred_item_idx_list = [i for i in pred_dict[user_idx][:k]]

            # average preicision
            rr = 0
            for order, item_idx in enumerate(pred_item_idx_list, start=1):
                if item_idx in truth_item_idx_list:
                    rr += (1 / order)

            result_dict[user_idx] = rr / min(k, len(truth_item_idx_list))

    # 4. Recall
    def recall(result_dict, 
                truth_dict, 
                pred_dict, 
                n=None, 
                k=int(cut_off)):

        for user_idx in truth_dict:
            # recall
            truth_item_idx_list = [i for i in truth_dict[user_idx][:n]]
            pred_item_idx_list = [i for i in pred_dict[user_idx][:k]]
            result_dict[user_idx] = \
                len( set(truth_item_idx_list) & set(pred_item_idx_list) ) / min(k, len(truth_item_idx_list))

    return {'precision': precision, 
            'P': precision,
            'average_precision': average_precision, 
            'AP': average_precision, 'mAP': average_precision,
            'reciprocal_rank': reciprocal_rank, 
            'RR': reciprocal_rank, 'mRR': reciprocal_rank,
            'recall': recall, 
            'R': recall}[metric]
    

