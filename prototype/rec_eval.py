import argparse
import collections
from evaluation import EVAL_FUNCTION
import numpy as np

class EvalDict:

    def __init__(self, 
                 eval_exploit=False,
                 eval_regular=False,
                 is_pred=False):

        self.is_pred = is_pred
        self.data_dict = {"EXPLORE": collections.defaultdict(list), 
                          "EXPLOIT": collections.defaultdict(list)}
        self.counter = collections.defaultdict(int)

        if eval_regular:
            self.data_dict['REGULAR'] = collections.defaultdict(list)
            self.data_dict['SOLO'] = collections.defaultdict(list)

    def sort(self, max_n_judge=None, subset='all', reverse=True):
        if subset == 'all':
            for s in self.data_dict.keys():
                self.sort(max_n_judge=max_n_judge, subset=s, reverse=reverse)
        else:
            for u in self.data_dict[subset]:
                self.data_dict[subset][u].sort(key=lambda x: x[1], reverse=reverse)
                self.data_dict[subset][u] = list(map(lambda x: x[0], self.data_dict[subset][u]))
        return 0

    def read_rec(self, path):
        f = open(path, 'r')
        if self.is_pred:
            for line in f:
                uid, pid, EEtype, RStype, score, rank = line.strip().split('\t')
                self.data_dict[EEtype][uid].append( (pid, rank) )
                self.counter[EEtype] += 1
                if RStype != 'NA':
                    self.data_dict[RStype][uid].append( (pid, rank) )
                    self.counter[RStype] += 1
        else:
            for line in f:
                uid, pid, EEtype, RStype, rating = line.strip().split('\t')
                self.data_dict[EEtype][uid].append( (pid, rating) )
                self.counter[EEtype] += 1
                if RStype != 'NA':
                    self.data_dict[RStype][uid].append( (pid, rating) )
                    self.counter[RStype] += 1
        f.close()

    def __repr__(self):
        NULL = None
        key_and_num = "\n\t  {:10} {:<9} {:<9}".format("SUBSET", "USERS", "EXAMPLES")
        for s in self.data_dict:
            key_and_num += "\n\t* {:<10} {:<9} {:<9}".format(
                    s, len(self.data_dict[s].keys()), self.counter[s]
            )

        return f"EvalDict({{\
                {key_and_num}\
                \n}})"

def main(args):

    # Load EvalDict
    truth = EvalDict(
            eval_exploit=args.evaluation_exploit,
            eval_regular=args.evaluation_regular
    )
    truth.read_rec(path=args.path_rec_truth)
    truth.sort()

    pred = EvalDict(
            eval_exploit=args.evaluation_exploit,
            eval_regular=args.evaluation_regular,
            is_pred=True
    )
    pred.read_rec(path=args.path_rec_prediction)
    pred.sort(reverse=False)
    
    # Eval collector
    result_per_user = collections.defaultdict(dict)

    for eval_set in truth.data_dict:
        # for metric in args.evaluation_metrics:
        for metric in args.evaluation_metrics:
            EVAL_FUNCTION(metric)(
                    result_dict=result_per_user[(metric, eval_set)],
                    truth_dict=truth.data_dict[eval_set],
                    pred_dict=pred.data_dict[eval_set],
                    n=None
            )

    # results (per user and mean)
    results_outprint = "=" * 30 + "\n"

    for metric_n_subset in result_per_user:
        metric, subset = metric_n_subset
        average = np.mean(list(result_per_user[metric_n_subset].values()))
        results_outprint += " {:<7} on {:<10} {:.4f}\n".format(
                metric, subset, average
        )

    print(truth)
    print(results_outprint + "=" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-truth', '--path_rec_truth', type=str, default="rec/truth.rec")
    parser.add_argument('-pred', '--path_rec_prediction', type=str, default="rec/pred.rec")
    parser.add_argument('-exploit', '--evaluation_exploit', action='store_false', default=True)
    parser.add_argument('-regular', '--evaluation_regular', action='store_false', default=True)
    parser.add_argument('-metric', '--evaluation_metrics', action='append')

    args = parser.parse_args()

    # check if the truth file is ready
    if ".rec" not in args.path_rec_truth:
        print("Please make sure the file includes the following format:\
                \nUSER_ID\tITEM_ID\tTYPE\tREGULAT\tRATING")
    # check if the pred file is ready
    if ".rec" not in args.path_rec_prediction:
        print("Please make sure the file includes the following format:\
                \nUSER_ID\tITEM_ID\tTYPE\tREGULAT\tSCORE\tRANK")
        exit(0)

    main(args)
