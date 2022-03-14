import argparse
import collections
import pandas as pd
import numpy as np
from lightfm.data import Dataset 
from lightfm import LightFM
from pysmore import SMORe
from datasets import Interaction, User, Item, RecDataset
from evaluation import Evaluation, EVAL_FUNCTION
from trainer import RecTrainer
from data_matrix import COOMatrix
from utils import npmapping
from utils import (
        identity,
        median_binarization, 
        binning_10,
        quantile_binning_25,
        prod_ccy_categorization,
        discard_categorization_100
)
from utils import (
        fill_mean,
        fill_mode,
        fill_median
)

def main(args):

    user_id_mapping = get_user(args.path_user)
    item_id_mapping= get_item(args.path_item)

    base = get_interaction_matrix(
            path=args.path_transaction_eval,
            user_mapping=user_id_mapping,
            item_mapping=item_id_mapping
    )
    ref, ref_sub = get_interaction_matrix(
            path=args.path_transaction,
            user_mapping=user_id_mapping,
            item_mapping=item_id_mapping,
            regular=True
    )

    # get overlapped subsets
    subsets = collections.defaultdict(np.ndarray)
    _ , subsets['exploit'], subsets['explore'] = get_intersection(
            left_mat=base,
            right_mat=ref,
            return_left=True
    )
    _ ,  subsets['regular'], subsets['solo'] = get_intersection(
            left_mat=subsets['exploit'],
            right_mat=ref_sub,
            return_left=True,
    )

    base_type = np.zeros(base.shape)
    # zero index indicates the un-subscribed
    base_type[subsets['explore'] != 0] = 1
    base_type[subsets['regular'] != 0] = -1
    base_type[subsets['solo'] != 0] = 2

    row_idx_list, col_idx_list = np.nonzero(base)
    user_ids = npmapping(row_idx_list, user_id_mapping, reverse=True)
    item_ids = npmapping(col_idx_list, item_id_mapping, reverse=True)

    with open(args.path_rec_output, 'w') as f:
        for (uid, iid, i, j) in zip(user_ids, item_ids, row_idx_list, col_idx_list):
            values = base[i, j]
            types = {0: "NA\tNA", 
                    -1: "EXPLOIT\tREGULAR",
                     2: "EXPLOIT\tSOLO",
                     1: "EXPLORE\tNA", }[base_type[i, j]]
            f.write(f'{uid}\t{iid}\t{types}\t{values}\n')

# Entities 
def get_user(path):
    customer = User()
    customer.read_csv(path)
    customer.preprocess(
            cust_filter_list=None, 
            time_alignment='age',
            na_preps={
                'gender_code': fill_mode,
                'income_range_code': fill_mode,
                'risk_type_code': fill_mode,
                'age': fill_median,
                'cust_vintage': fill_median
            }
    )
    customer_id_mapping = {}
    for customer_id in customer.get_id_list():
        customer_id_mapping.setdefault(customer_id, len(customer_id_mapping))

    return customer_id_mapping

def get_item(path):
    product = Item()
    product.read_csv(path)
    product.preprocess(
            fund_filter_list=None,
            na_preps={
                'can_rcmd_ind': identity,
                'prod_risk_code': identity ,
                'prod_detail_type_code': identity,
                'prod_ccy': prod_ccy_categorization,
                'mkt_rbot_ctg_ic': discard_categorization_100,
            }
    )
    product_id_mapping = {}
    for product_id in product.get_id_list():
        product_id_mapping.setdefault(product_id, len(product_id_mapping))

    return product_id_mapping

def get_interaction_matrix(path, user_mapping=None, item_mapping=None, regular=False):
    transaction = Interaction()
    transaction.read_csv(path)
    transaction.preprocess(
            min_required_items=0,
            min_required_users=0,
            used_txn_info=['txn_dt', 'txn_amt', 'deduct_cnt'],
            min_time_unit='day',
            utility='binary',
            do_regular=True 
    )
    transaction.truncate(
            cust_avail_list=list(user_mapping.keys()),
            prod_avail_list=list(item_mapping.keys())
    )

    # data to nd array (all)
    data = transaction.get_data()
    matrix_trans = COOMatrix(
            shape=(len(user_mapping), len(item_mapping)),
            dtype=np.float32
    )
    
    # print(user_mapping)
    matrix_trans.append_all(
            all_i=npmapping(data['users'], user_mapping),
            all_j=npmapping(data['items'], item_mapping),
            all_v=data['ratings']
    )

    if regular:
        data = transaction.get_regular_plan_data()
        matrix_regular = COOMatrix(
                shape=(len(user_mapping), len(item_mapping)),
                dtype=np.float32
        )
        matrix_regular.append_all(
                all_i=npmapping(data['users'], user_mapping),
                all_j=npmapping(data['items'], item_mapping),
                all_v=data['regular']
        )
        return (matrix_trans.tocoo().toarray(), matrix_regular.tocoo().toarray())
    else:
        return matrix_trans.tocoo().toarray()

def get_intersection(left_mat, 
                     right_mat, 
                     return_left=False, 
                     return_right=False):

    intersection = np.multiply(left_mat!=0, right_mat!=0)
    subset_overlapped = np.multiply(left_mat, intersection)

    if return_left:
        subset_remained = np.multiply(left_mat, ~intersection)
    if return_right:
        subset_remained = np.multiply(right_mat, ~intersection)

    return intersection, subset_overlapped, subset_remained

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_transaction', type=str)
    parser.add_argument('--path_transaction_eval', type=str)
    parser.add_argument('--path_rec_output', type=str)
    # parser.add_argument('--is_prediction', action='store_true', default=False)
    parser.add_argument('--path_user', type=str, default="customer_meta_features.csv")
    parser.add_argument('--path_item', type=str, default="item_meta_features.csv")
    args = parser.parse_args()
    main(args)
