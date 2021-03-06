import math, os, collections
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from Graph_module.SMORe import SMORe 
from Graph_module.TPR import TPR 
import argparse
from evaluation import Evaluation
from db_connection.utils import get_conn
from utils import load_w103, load_w106 
from mlaas_tools.config_build import config_set

## Configure env
if not os.path.isfile('config.ini'):
    config_set()
## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--train_span", type=int, default=1, help="Training Period")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--mode", default='bpr', type=str, help="choose algorithms")
parser.add_argument("--model", default='tpr', type=str, help="choose tpr or smore")
args = parser.parse_args()
today = args.date
span = args.train_span
duration = args.eval_duration
algo = args.mode
## Load db connection
rawdata_conn = get_conn('edu')
## Load data
print("Loading Data...")
w103_df = load_w103(today, rawdata_conn, span)
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()
w106_df = load_w106(rawdata_conn)
_filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
w106_df_filter = w106_df[_filter]
_selected_col = ['wm_prod_code', 'prod_detail_type_code','prod_ccy','prod_risk_code','can_rcmd_ind']
w106_df_filter = w106_df_filter[_selected_col]
## Init SMORe
if args.model == 'smore':
    model = SMORe(w103_df)
else:
    model = TPR(w103_df, w106_df_filter)
## Get user & item emb.
print("Training Model...")
if args.model == 'smore':
    user_emb, item_emb = model.fit(mode=algo, lr=0.05, update_times=100)
else: 
    user_emb, item_emb = model.fit(lr=0.05, update_times=100)
## Calculate cosine similarity of every (u, i) pair, n_user * n_item
print("Predicting...")
scores = cosine_similarity(user_emb.fillna(0), item_emb.fillna(0))
## Recommend 5 funds for every user
id2user, prediction = model.user_dict, collections.defaultdict(list)
for i, score in enumerate(tqdm(scores, total=len(scores))):
   user = id2user[i] 
   prediction[user] = [i[1] for i in sorted(zip(score, model.items), reverse=True )][:5]

## Evaluate results
print("Evaluating Results...")
evaluation = Evaluation(today, prediction, duration, purchase_hist)
score, score_warm, upper_bound_weak, upper_bound_strong, upper_warm = evaluation.results()
buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

print(f'Today: {today} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mean-Precision-all: {score} Mean-Precision-warm: {score_warm} Upper-Bound-weak: {upper_bound_weak} Upper-Bound-strong: {upper_bound_strong} Upper-Bound-Warm: {upper_warm} Buy-old: {buy_old_user} Buy-new: {buy_new_user}\n')

print("Done!") 
