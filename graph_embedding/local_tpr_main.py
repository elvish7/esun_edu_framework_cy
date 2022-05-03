import math, os, collections
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from Graph_module.SMORe import SMORe 
from Graph_module.TPR import TPR 
from Graph_module.TPR_notext import TPR_notext 
import argparse
from local_evaluation import Evaluation
## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--date", default='2018-12-31', help="Recommendation date")
parser.add_argument("--train_span", type=int, default=6, help="Training Period")
parser.add_argument("--mode", default='bpr', type=str, help="choose algorithms")
parser.add_argument("--model", default='tpr', type=str, help="choose tpr or smore or tpr_notext")
# tpr tune
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--neg", type=int, default=5)
parser.add_argument("--l", type=float, default=0.025)
parser.add_argument("--t", type=int, default=100)

args = parser.parse_args()
date = args.date
span = args.train_span
algo = args.mode
## Get data path
path = '/tmp2/cywu/raw_datasets/'+ date + '_' + str(span)
train_path = path + '/train_w103.csv'
evaluation_path = path + '/evaluate_w103.csv'
item_feature_path = path + '/item_features.csv'
user_feature_path = path + '/user_features.csv'
## Read data
w103_df = pd.read_csv(train_path)
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()
if args.model != 'smore':
    w106_df = pd.read_csv(item_feature_path)
    _filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
    w106_df_filter = w106_df[_filter]
    print(w106_df_filter.columns)
    if args.model == 'tpr':
        _selected_col = ['wm_prod_code', 'can_rcmd_ind', 'high_yield_bond_ind', 'counterparty_code', 
                            'invest_type', 'mkt_rbot_ctg_ic', 'prod_ccy', 'prod_detail_type_code', 'prod_risk_code']
    else:
        _selected_col = ['wm_prod_code']
    w106_df_filter = w106_df_filter[_selected_col]
    #cm_customer_m_df = pd.read_csv(user_feature_path)
## Init SMORe
if args.model == 'smore':
    model = SMORe(w103_df)
else:
    model = TPR(w103_df, w106_df_filter)
    
## Get user & item emb.
if args.model == 'smore':
    user_emb, item_emb = model.fit(mode=algo, lr=0.05, update_times=100)
else: 
    user_emb, item_emb = model.fit(d=args.dim, ns= args.neg, lr=args.l, update_times=args.t)
## Calculate cosine similarity of every (u, i) pair, n_user * n_item
scores = cosine_similarity(user_emb.fillna(0), item_emb.fillna(0))
## Recommend 5 funds for every user
id2user, prediction = model.user_dict, collections.defaultdict(list)
for i, score in enumerate(tqdm(scores, total=len(scores))):
   user = id2user[i] 
   prediction[user] = [i[1] for i in sorted(zip(score, model.items), reverse=True )][:5]
## Evaluate results
print("Evaluating Results...")
eva_df = pd.read_csv(evaluation_path)
evaluation = Evaluation(date, prediction, purchase_hist, eva_df)
warm_user, cold_user = evaluation.warm_cold_list()
eval_mode = 'warm'
if eval_mode == 'warm':
    warm_pred = {k: v for k, v in prediction.items() if k in warm_user}
    score, upper_bound = evaluation.results(warm_pred)
buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

# print(f'Today: {date} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound} Used features: {_selected_col}\n')
print(f'Today: {date} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound} params: {str(args.dim)+str(args.neg)+str(args.l)+str(args.t)}\n')
print("Done!") 
