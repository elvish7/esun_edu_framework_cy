import os
import argparse
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from local_evaluation import Evaluation
from utils import w106_process, cust_process, recommendation_all, create_all_feature_pairs, build_feature_tuples
## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--date", default='2018-12-31', help="Recommendation date")
parser.add_argument("--train_span", type=int, default=6, help="Training Period")
parser.add_argument("--user_ft", help="Use user features", action='store_true')
parser.add_argument("--item_ft", help="Use item features", action='store_true')
parser.add_argument("--dim", default=128, type=int, help="feature emb. dimensions")
parser.add_argument("--epoch", default=20, type=int, help="epoch num")
parser.add_argument("--feature_i", default=1, type=int, help="selected feature")
args = parser.parse_args()
date = args.date
span = args.train_span
dim = args.dim
epoch = args.epoch
## Get data path
# path = os.getcwd()
path = '/tmp2/cywu/raw_datasets/'+ date + '_' + str(span)
train_path = path + '/train_w103.csv'
evaluation_path = path + '/evaluate_w103.csv'
if args.item_ft:
    item_feature_path = path + '/item_features.csv'
if args.user_ft:
    user_feature_path = path + '/user_features.csv'
## Read data
w103_df = pd.read_csv(train_path)
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()
if args.item_ft:
    w106_df = pd.read_csv(item_feature_path)
    w106_df = w106_process(w106_df)
if args.user_ft:
    cm_customer_m_df = pd.read_csv(user_feature_path)
    cm_customer_m_df = cust_process(cm_customer_m_df)
## Intersection of w103 & w106 wrt wm_prod_code
if args.item_ft:
    _filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
    w106_df_filter = w106_df[_filter]
    _selected_col = ['wm_prod_code','can_rcmd_ind', 'invest_type','prod_risk_code', 'prod_detail_type_code', 'mkt_rbot_ctg_ic', 'counterparty_code', 'prod_ccy', 'high_yield_bond_ind']
    w106_df_filter = w106_df_filter[['wm_prod_code', _selected_col[args.feature_i]]]
## Intersection of w103 & cm_customer_m wrt cust_no
if args.user_ft:
    _filter = cm_customer_m_df.cust_no.isin(w103_df['cust_no'].tolist())
    cust_df_filter = cm_customer_m_df[_filter]
    _selected_col = ['cust_no', 'age', 'gender_code', 'cust_vintage', 'income_range_code', 'risk_type_code', 'children_cnt', 'edu_code', 'wm_club_class_code']
    cust_df_selected = cust_df_filter[_selected_col].groupby('cust_no').tail(1)
## Create features
user_fts, item_fts = None, None
if args.user_ft:
    user_fts = create_all_feature_pairs(cust_df_selected)
if args.item_ft:
    item_fts = create_all_feature_pairs(w106_df_filter)
## Fit the dataset
dataset1 = Dataset()
dataset1.fit(
        w103_df['cust_no'].unique(), # all the users
        w103_df['wm_prod_code'].unique(), # all the items
        user_features = user_fts,
        item_features = item_fts
)
## Build features
user_features, item_features = None, None
if args.user_ft:
    user_tuples = build_feature_tuples(cust_df_selected)
    user_features = dataset1.build_user_features(user_tuples, normalize= False)
if args.item_ft:
    item_tuples = build_feature_tuples(w106_df_filter)
    item_features = dataset1.build_item_features(item_tuples, normalize= False)

## Build interactions
# (interactions, weights) = dataset1.build_interactions([(x[1], x[2], x[4]) for x in w103_df.values ])
(interactions, weights) = dataset1.build_interactions([(x[1], x[2], 1) for x in w103_df.values ])
## Get Id mappings 
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
## Train model
model = LightFM(no_components=dim, loss='warp')
model.fit(interactions, # spase matrix representing whether user u and item i interacted
      user_features= user_features, # we have built the sparse matrix above
      item_features= item_features, # we have built the sparse matrix above
      sample_weight= weights, # spase matrix representing how much value to give to user u and item i inetraction: i.e ratings
      epochs=epoch)
## Get user list for predictions
user_list = w103_df['cust_no'].unique().tolist()
## Recommend each user's top5 funds
pred = recommendation_all(model, interactions, user_list, user_id_map, item_id_map, user_features, item_features)
## Evaluate results
print("Evaluating Results...")
eva_df = pd.read_csv(evaluation_path)
evaluation = Evaluation(date, pred, purchase_hist, eva_df)
warm_user, cold_user = evaluation.warm_cold_list()
eval_mode = 'warm'
if eval_mode == 'warm':
    warm_pred = {k: v for k, v in pred.items() if k in warm_user}
    score, upper_bound = evaluation.results(warm_pred)
buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

print(f'Today: {date} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound} Used features: {_selected_col[args.feature_i]}\n')
# print(f'Today: {date} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound}\n')
print("Done!") 
