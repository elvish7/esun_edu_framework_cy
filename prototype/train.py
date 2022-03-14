"""
Main function
"""
import argparse
import pandas as pd
import numpy as np
from lightfm.data import Dataset 
from lightfm import LightFM
from pysmore import SMORe
from datasets import Interaction, User, Item, RecDataset
from evaluation import Evaluation, EVAL_FUNCTION
from trainer import RecTrainer
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
    """
    Pipeline:
    1) Data preparations: 
        - Interaction of training 
        - User table with metas
        - Item table with metas
        - Interaction of evaluation
        - Interaction of testing (evaluation on N month)
    2) Datset prepration
        - Building matrix with interaction matrices
            * Building matrix with interaction "intersection"
        - Building matrix with feature matrices
    3) Evaluation object
        - loading truth ranksets, besides the evaluation but also diff (old and new)
    4) Model (includes: lightfm)
    5) Trainer
        - Pre-defined training configs
    6) Model training 
    """
    # Data preparation 
    ## interactions (training)
    transaction = Interaction()
    transaction.read_csv(args.path_transaction)
    transaction.preprocess(
            min_required_items=0,
            min_required_users=0,
            used_txn_info=['txn_dt', 'txn_amt', 'deduct_cnt'],
            min_time_unit='day',
            utility='binary',
            do_regular=True if args.evaluation_regular else False
    )
    import time

    NULL=None
    print(f"==========(1) Load Interaction ========== \
            \nTime cost: {NULL}\
            \nMemory cost: {NULL}\
            \n{transaction} \
            \n=======================")

    ## interactions (evaluation)
    transaction_eval = Interaction()
    transaction_eval.read_csv(args.path_transaction_eval)
    transaction_eval.preprocess(
        min_required_items=0,
        min_required_users=0,
        used_txn_info=['txn_dt', 'txn_amt', 'deduct_cnt'],
        min_time_unit='day',
        utility='binary',
        do_regular=True if args.evaluation_regular else False
    )
    print(transaction_eval)

    ## interactions (testing)
    ### [TBA] Use the time argument to implement the periodic spans
    if args.path_transaction_test is not None:
        transaction_test = Interaction()
        transaction_test.read_csv(args.path_transaction_test)
        transaction_test.preprocess(
            min_required_items=0,
            min_required_users=0,
            used_txn_info=['txn_dt', 'txn_amt', 'deduct_cnt'],
            min_time_unit='day',
            utility='binary'
        )
        print(transaction_test)

    # customer
    customer = User()
    customer.read_csv(args.path_user)
    customer.preprocess(
            cust_filter_list=None, 
            time_alignment='age',
            na_preps={
                'gender_code': fill_mode,
                'income_range_code': fill_mode,
                'risk_type_code': fill_mode,
                'age': fill_median,
                'cust_vintage': fill_median
            },
            meta_preps={
                'gender_code': identity,
                'income_range_code': identity,
                'risk_type_code': identity, 
                'age': median_binarization,
                # 'age': binning_10,
                'cust_vintage': median_binarization,
                # 'cust_vintage': quantile_binning_25,
            }
    )
    print(customer)


    # products (funds)
    product = Item()
    product.read_csv(args.path_item)
    product.preprocess(
            fund_filter_list=None,
            meta_preps={
                # 'can_rcmd_ind': identity,
                'prod_risk_code': identity ,
                'prod_detail_type_code': identity,
                'prod_ccy': prod_ccy_categorization,
                # 'prod_ccy': identity,
                'mkt_rbot_ctg_ic': discard_categorization_100,
                # 'mkt_rbot_ctg_ic': identity,
            }
    )
    print(product)
    
    # [DEBUG] truncate the out-user/out-item interactions
    transaction.truncate(
        cust_avail_list=customer.get_id_list(),
        prod_avail_list=product.get_id_list()
    )
    transaction_eval.truncate(
        cust_avail_list=customer.get_id_list(),
        prod_avail_list=product.get_id_list()
    )

    # RecDataset
    dataset = RecDataset(
            train_interaction_instance=transaction,
            eval_interaction_instance=transaction_eval,
            test_interaction_instance=None,
    )
    dataset.fit(
            user_instance=customer,
            item_instance=product,
            user_features=True,
            item_features=True,
    )

    print("Building interaction and user/item feature matrices ...\n")
    ## Interaction (training)
    dataset.build_auto_matrix(
            split='train', 
            data_name='interactions', 
            dict_key='ratings'
    )
    dataset.build_feature_matrix(
            split='train', 
            user_instance=customer,
            item_instance=product,
    )

    ## Interaction (evaluation)
    dataset.build_auto_matrix(
            split='eval',
            data_name='interactions',
            dict_key='ratings'
    )
    print(dataset)
    dataset.calculate_cold_start()

    # Evaluation 
    evaluation = Evaluation(
            dataset_instance=dataset,
            evaluate_train=args.evaluation_train,
            evaluate_diff=args.evaluation_diff,
            evaluate_regular=args.evaluation_regular,
            heuristics=[] if args.use_heuristic is None else args.use_heuristic
    )
    print(evaluation)

    # models
    if args.model == 'LightFM':
        model = LightFM(
                no_components=args.model_hidden_dimension,
                loss=args.model_loss,
                max_sampled=args.model_max_neg_sample,
        )
    
    if args.model == 'SMORe':
        # build the network file (to-be-trained), 
        # [NOTE] that in SMORe, there is only one split, which is 'train'
        dataset.build_interaction_network(
                split='train',
                dict_key='ratings',
                output_file=args.path_temp_network
        )
        model = SMORe(
                dimension_size=args.model_hidden_dimension,
                loss=args.model_loss,
                negative_samples=args.model_max_neg_sample,
                n_user=len(customer.get_id_list()),
                n_item=len(product.get_id_list()),
                input_file=args.path_temp_network,
                output_file=args.path_temp_reps
        )

    # Trainer
    trainer = RecTrainer(
            model=model,
            model_type=args.model_type,
            model_path=args.model_path,
            model_name=args.model,
            dataset=dataset,
            evaluation=evaluation,
            compute_metrics=args.evaluation_metrics
    )
    print(trainer)

    # ** start training **
    trainer.train(
            epochs=args.training_num_epochs,
            do_eval=args.training_do_evaluation,
            verbose=args.training_verbose,
            eval_per_epochs=args.training_eval_per_epochs
    )
    # ** End training **
    
    # save the reports
    evaluation.save_results_to_file(file_path=args.evaluation_results_csv)
    evaluation.save_recommedation_to_file(file_path=args.evaluation_rec_detail_report)
    print(evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--path_transaction', type=str, default="training_transaction.csv")
    parser.add_argument('--path_transaction_eval', type=str, default="evaluation_transaction.csv")
    parser.add_argument('--path_transaction_test', type=str, default=None)
    parser.add_argument('--path_user', type=str, default="customer_meta_features.csv")
    parser.add_argument('--path_item', type=str, default="item_meta_features.csv")

    # entity arguments
    # model arguments
    parser.add_argument('--model', type=str, default='LightFM')
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--model_hidden_dimension', type=int, default=128)
    parser.add_argument('--model_max_neg_sample', type=int, default=10)
    parser.add_argument('--model_loss', type=str, default='warp')

    # training arguments
    parser.add_argument('--training_do_evaluation', action='store_true', default=False)
    parser.add_argument('--training_verbose', action='store_true', default=False)
    parser.add_argument('--training_num_epochs', type=int, default=1)
    parser.add_argument('--training_eval_per_epochs',type=int, default=1)

    # evaluation arguments
    parser.add_argument('--evaluation_train', action='store_true', default=False)
    parser.add_argument('--evaluation_diff', action='store_true', default=False)
    parser.add_argument('--evaluation_regular', action='store_true', default=False)
    parser.add_argument('--evaluation_results_csv', type=str, default='evaluation.csv')
    parser.add_argument('--evaluation_rec_detail_report', type=str, default='rec_deatail.tsv')
    parser.add_argument('--evaluation_metrics', action='append')

    # other arguments
    parser.add_argument('--use_heuristic', action='append')
    parser.add_argument('--path_temp_network', type=str, default='temp/neworks.txt')
    parser.add_argument('--path_temp_reps', type=str, default='temp/smore_reps.txt')

    args = parser.parse_args()
    main(args)
