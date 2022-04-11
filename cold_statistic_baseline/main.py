import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
from evaluation import Evaluation
from mlaas_tools.config_build import config_set
from db_connection.utils import get_conn
from utils import load_w103, load_w106, load_cust
## Configure env
if not os.path.isfile('config.ini'):
    config_set()
## Add params
parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--train_span", type=int, default=1, help="Training Period")
parser.add_argument("--mode", type=str, default='popularity', help="random or popularity")
parser.add_argument("--eval_mode", default='warm', type=str, help="choose warm or cold")
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
## Popularity
## Recommend each user's top5 funds
print("Predicting...")
popularity_dict = {}
if mode == 'popularity':
    # by unique user
    #top5_fund = w103_df.groupby(["cust_no","wm_prod_code"])["wm_prod_code"].count().sort_values(ascending=False).head(5).index.to_list()
    #top5_fund = [i[1] for i in top5_fund]
    # by all data
    top5_fund = w103_df.groupby("wm_prod_code")["wm_prod_code"].count().sort_values(ascending=False).head(5).index.to_list()
    pred = {u: top5_fund for u in user_list}
    
## Evaluate results
print("Evaluating Results...")
evaluation = Evaluation(today, pred, duration, purchase_hist)
score, upper_bound = evaluation.results(pred)

buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

print(f'Today: {today} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound} \n')

print("Done!") 