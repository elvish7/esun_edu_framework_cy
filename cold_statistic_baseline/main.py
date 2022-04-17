import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
from evaluation import Evaluation
from mlaas_tools.config_build import config_set
from db_connection.utils import get_conn
from utils import load_w103, load_w106, load_cust, fast_topK
from sklearn.metrics.pairwise import cosine_similarity

## Configure env
if not os.path.isfile('config.ini'):
    config_set()
## Add params
parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--train_span", type=int, default=1, help="Training Period")
parser.add_argument("--mode", type=str, default='popularity', help="popularity or similarity")
parser.add_argument("--eval_mode", default='cold', type=str, help="choose warm or cold")
args = parser.parse_args()
today = args.date
duration = args.eval_duration
mode = args.mode
span = args.train_span
eval_mode = args.eval_mode
## Load db connection
rawdata_conn = get_conn('edu')
## Load data
print("Loading Data...")
# interaction train w103
w103_df = load_w103(today, rawdata_conn, span)
# user
cust_df = load_cust(today, rawdata_conn, span)
# intersection
user_filter = set(w103_df['cust_no'].tolist()) & set(cust_df['cust_no'].tolist())
w103_df = w103_df[w103_df['cust_no'].isin(user_filter)]
cust_df_filter = cust_df[cust_df['cust_no'].isin(user_filter)] 
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()
# evaluation w103
evaluation = Evaluation(today, None, duration, purchase_hist)
evaluate_w103 = evaluation.read(today, rawdata_conn, duration)
# evaluation user features
eva_cust_df = load_cust(today, rawdata_conn, span=None, mode='evaluation')
# intersection
eva_user_filter = set(evaluate_w103['cust_no'].tolist()) & set(eva_cust_df['cust_no'].tolist())
eva_cust_df_filter = eva_cust_df[eva_cust_df['cust_no'].isin(eva_user_filter)]
evaluate_w103 = evaluate_w103[evaluate_w103['cust_no'].isin(eva_user_filter)]

warm_users, cold_users = evaluation.warm_cold_list()
print('warm-users:', len(warm_users), 'cold-users:', len(cold_users))

# cold start users
user_list = evaluate_w103[evaluate_w103['cust_no'].isin(cold_users)]['cust_no'].unique().tolist()
print(len(user_list))

## Recommend each user's top5 funds
print("Predicting...")
popularity_dict = {}
## Popularity
if mode == 'popularity':
    # by unique user
    #top5_fund = w103_df.groupby(["cust_no","wm_prod_code"])["wm_prod_code"].count().sort_values(ascending=False).head(5).index.to_list()
    #top5_fund = [i[1] for i in top5_fund]
    # by all data
    top5_fund = w103_df.groupby("wm_prod_code")["wm_prod_code"].count().sort_values(ascending=False).head(5).index.to_list()
    pred = {u: top5_fund for u in user_list}

if mode == 'similarity':
    eva_cust_df_cold = eva_cust_df_filter[eva_cust_df_filter['cust_no'].isin(user_list)]
    cat_cols = ['gender_code', 'income_range_code','risk_type_code', 'edu_code', 'wm_club_class_code']
    all_dummy = pd.get_dummies(pd.concat([cust_df_filter, eva_cust_df_cold]), columns=cat_cols)
    cust_old_matrix = all_dummy.iloc[:len(cust_df_filter)].iloc[:, 1:].values
    cust_new_matrix = all_dummy.iloc[len(cust_df_filter):].iloc[:, 1:].values
    sim = cosine_similarity(cust_new_matrix, cust_old_matrix)
    # get the topk similar old users for each cold start user
    topk_sim_old_user = fast_topK(sim, 5)
    old_cust_idx = cust_df_filter.reset_index().cust_no.to_dict()
    for i in range(len(topk_sim_old_user)):
        topk_sim_old_user[i] = [old_cust_idx[j] for j in topk_sim_old_user[i]]
    # recommend by topk old user's bought funds
    recommend_funds = []
    for i in topk_sim_old_user:
        hist_fund = []
        for old_u in i:
            hist_fund += purchase_hist[old_u]
        recommend_funds.append(hist_fund[:5])
    pred = {u: v for u, v in zip(eva_cust_df_cold.cust_no, recommend_funds)}
    
    
## Evaluate results
print("Evaluating Results...")
evaluation = Evaluation(today, pred, duration, purchase_hist)
score, upper_bound = evaluation.results(pred)

buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

print(f'Today: {today} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound} \n')

print("Done!") 