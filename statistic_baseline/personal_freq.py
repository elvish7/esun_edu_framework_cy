import os
import argparse
import pandas as pd
import numpy as np
import collections
from evaluation import Evaluation
from mlaas_tools.config_build import config_set
from db_connection.utils import get_conn
from utils import recommendation_all, load_w103, load_w106, load_cust_pop
## Configure env
if not os.path.isfile('config.ini'):
    config_set()
## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--train_span", type=int, default=1, help="Training Period")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--eval_mode", default='warm', type=str, help="choose warm or cold")
args = parser.parse_args()
today = args.date
span = args.train_span
duration = args.eval_duration
eval_mode = args.eval_mode
## Load db connection
rawdata_conn = get_conn('edu')
## Load data
print("Loading Data...")
w103_df = load_w103(today, rawdata_conn, span).groupby(["cust_no","wm_prod_code"], group_keys=False).apply(lambda x: x.sort_values(["deduct_cnt"],ascending=True).head(1))
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()
purchase_record = [(u, i) for u, i in zip(w103_df['cust_no'], w103_df['wm_prod_code'])]
purchase_df = pd.DataFrame([(*record, count) for record, count in collections.Counter(purchase_record).items()], columns = ['cust_no','wm_prod_code','count'])
purcahse_df = purchase_df.sort_values(['cust_no','count'])
## Get user list for predictions
user_list = w103_df['cust_no'].unique().tolist()
## Recommend each user's top5 funds
pred = purchase_df.groupby('cust_no')['wm_prod_code'].apply(lambda x: list(x)[:5]).to_dict() 
## Evaluate each user's precision@5
print("Evaluating Results...")
evaluation = Evaluation(today, pred, duration, purchase_hist)
warm_user, cold_user = evaluation.warm_cold_list()
if eval_mode == 'warm':
    warm_pred = {k: v for k, v in pred.items() if k in warm_user}
    score, upper_bound = evaluation.results(warm_pred)
else: # pass
    pass
buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

print(f'Today: {today} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mode: {eval_mode} Mean-Precision: {score} Upper-Bound: {upper_bound} \n')

print("Done!") 
