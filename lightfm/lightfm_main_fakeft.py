import os
import argparse
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.data import Dataset
from evaluation import Evaluation
from mlaas_tools.config_build import config_set
from db_connection.utils import get_conn
from utils import recommendation_all, load_w103, load_w106, w106_process, load_cust, cust_process, create_all_feature_pairs, build_feature_tuples, load_cust_pop_0205 #weighted

## Configure env
if not os.path.isfile('config.ini'):
    config_set()
## Add params
parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--train_span", type=int, default=18, help="Training Period")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--dim", default=128, type=int, help="feature emb. dimensions")
parser.add_argument("--epoch", default=20, type=int, help="epoch num")
parser.add_argument("--user_ft", help="Use user features", action='store_true')
parser.add_argument("--item_ft", help="Use item features", action='store_true')
parser.add_argument("--eval_mode", default='warm', type=str, help="choose warm or cold")
parser.add_argument("--feature_number", type=int, default=9, help="number of selected features")
parser.add_argument("--feature_i", default=1, type=int, help="selected feature")
args = parser.parse_args()
today = args.date
span = args.train_span
duration = args.eval_duration
dim = args.dim
epoch = args.epoch
eval_mode = args.eval_mode
f_num = args.feature_number

## Load db connection
rawdata_conn = get_conn('edu')
## Load data
print("Loading Data...")
w103_df = load_w103(today, rawdata_conn, span)
cm_customer_m_df = load_cust(today, rawdata_conn, span=span)
w103_df = w103_df[w103_df['cust_no'].isin(cm_customer_m_df.cust_no)]
print(w103_df.head(2), len(w103_df))
if args.user_ft:
    _filter = cm_customer_m_df.cust_no.isin(w103_df['cust_no'].tolist())
    cust_df_filter = cm_customer_m_df[_filter]
    _selected_col = ['cust_no', 'age', 'gender_code', 'cust_vintage', 'income_range_code', 'risk_type_code', 'children_cnt', 'edu_code', 'wm_club_class_code']
    cust_df_filter = cust_df_filter[_selected_col[:f_num]]
    
if args.item_ft:
    w106_df = load_w106(rawdata_conn)
    _filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
    w106_df_filter = w106_df[_filter]
    _selected_col = ['wm_prod_code','can_rcmd_ind', 'invest_type','prod_risk_code', 'prod_detail_type_code', 'mkt_rbot_ctg_ic', 'counterparty_code', 'prod_ccy', 'high_yield_bond_ind']
    w106_df_filter = w106_df_filter[['wm_prod_code', _selected_col[args.feature_i]]]

purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()

# construct a useful feature
eva = Evaluation(today, None, duration, purchase_hist, cm_customer_m_df)
eva_df = eva.read(today, get_conn('edu'), duration)
eva_df = eva_df[eva_df['cust_no'].isin(cm_customer_m_df.cust_no)]
eva_ans = eva_df[['cust_no', 'wm_prod_code']].groupby('cust_no').first()
fake_ft = cust_df_filter[['cust_no']].merge(eva_ans, how='left', on='cust_no').fillna(0)
cust_df_filter = fake_ft    
print('fake features', cust_df_filter.head(3))
print(w103_df.head(2), len(w103_df))

## Create features
user_fts, item_fts = None, None
if args.user_ft:
    user_fts = create_all_feature_pairs(cust_df_filter)
if args.item_ft:
    item_fts = create_all_feature_pairs(w106_df_filter)


## Fit dataset
#dataset1 = Dataset(False,False)
dataset1 = Dataset()
user_list = w103_df['cust_no'].unique() # all the users
item_list = w103_df['wm_prod_code'].unique() # all the items

dataset1.fit(
        user_list, # all the users
        item_list, # all the items
        user_features = user_fts,
        item_features = item_fts
)
# mapping 

## Build features
user_features, item_features = None, None
if args.user_ft:
	# e.g. [("USER1", [0, 1, ,1 ... ]), ("USER2", ["..."])]
    user_tuples = build_feature_tuples(cust_df_filter)
    user_features = dataset1.build_user_features(user_tuples, normalize= False)

if args.item_ft:
    item_tuples = build_feature_tuples(w106_df_filter)
    item_features = dataset1.build_item_features(item_tuples, normalize= False)

## Transform w103
#w103_df = weighted(w103_df)
## Build interactions
#(interactions, weights) = dataset1.build_interactions([(x[0], x[1], x[3]) for x in w103_df.values ])
(interactions, weights) = dataset1.build_interactions([(x[0], x[1], 1) for x in w103_df.values ])
## Get Id mappings 
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
## Train model
print("Training Model...")
## dim =128 epoch =20 is the best
model = LightFM(no_components=dim, loss='warp')
model.fit(interactions, # spase matrix representing whether user u and item i interacted
      user_features= user_features, # we have built the sparse matrix above
      item_features= item_features, # we have built the sparse matrix above
      sample_weight= weights, # spase matrix representing how much value to give to user u and item i inetraction: i.e ratings
      epochs=epoch)
## Predict top5 funs for each user
print("Predicting...")
user_list = w103_df['cust_no'].unique().tolist()
pred = recommendation_all(model, interactions, user_list, user_id_map, item_id_map, user_features, item_features)
## Evaluate results
# print("Evaluating Results...")
# evaluation = Evaluation(today, pred, duration)
# score, score_warm, upper_bound_weak, upper_bound_strong, upper_warm = evaluation.results()
# non_cold_start_user, cold_start_user = evaluation.coverage()
# print(f'Today: {today} Training-Span: {span} Non-Cold-Start-Users: {non_cold_start_user} Cold-Start-Users: {cold_start_user} Mean-Precision-all: {score} Mean-Precision-warm: {score_warm} Upper-Bound-weak: {upper_bound_weak} Upper-Bound-strong: {upper_bound_strong} Upper-Bound-Warm: {upper_warm}\n')

## Evaluate results
print("Evaluating Results...")
evaluation = Evaluation(today, pred, duration, purchase_hist, cm_customer_m_df)
warm_user, cold_user = evaluation.warm_cold_list()
if eval_mode == 'warm':
    warm_pred = {k: v for k, v in pred.items() if k in warm_user}
    score, upper_bound = evaluation.results(warm_pred)
else: # pass
    pass
buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

print(f'Today: {today} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound} Used features: {_selected_col[args.feature_i]}\n')
print("Done!") 
