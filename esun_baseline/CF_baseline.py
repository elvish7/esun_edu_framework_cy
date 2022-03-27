#!/usr/bin/env python
# coding: utf-8

# # 舊戶基金推薦流程
import pandas as pd
import numpy as np
import argparse
import os
from experiments.old_cust_baseline.experiment_module import old_cust_CF_baseline
from utils import recommendation_all
from evaluation import Evaluation

from db_connection.utils import get_conn
from utils import load_w103
from mlaas_tools.config_build import config_set
from mlaas_tools.config_build import config_set

## Configure env
if not os.path.isfile('config.ini'):
    config_set()

parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--training_span", type=int, default=1, help="Training Period")
args = parser.parse_args()
today = args.date
span = args.training_span
duration = args.eval_duration

# ## 模型計算 

#assert today == '2019-06-30'
#model = old_cust_CF_baseline(today=today, mode='model')

#model.pipe.view(summary=False) 


# ### 模型結果 

# 基金對顧客的喜好分數 
#model.pipe.cust_item_rating_matrix.get(load_tmp=True) 

# 顧客id對應到position id 表
#model.pipe.user2nid.get(load_tmp=True) 


# 基金id對應到position id 表
#model.pipe.item2nid.get(load_tmp=True)  


# ## 排序與初步篩選 

rank_n_filter = old_cust_CF_baseline(today=today, span=span, mode='all')

#rank_n_filter.pipe.view(summary=False) 


# ### 排序與初篩後結果 

# 基金排序結果 
#rank_n_filter.pipe.cust_item_rank_matrix_out.get() 


# 顧客基金喜好分數 
#rank_n_filter.pipe.cust_item_rating_matrix_out.get()  


# 顧客id對應到position id 表
#rank_n_filter.pipe.user2nid.get(verbose=True, load_tmp=True)    


# 基金id對應到position id 表
#rank_n_filter.pipe.item2nid_out.get()


# 顧客id對應到position id 表
#no_filter_user_id_map = pd.Series(model.pipe.user2nid.get(verbose=True, load_tmp=True).index.values).to_dict()    
user_id_map = pd.Series(rank_n_filter.pipe.user2nid.get(verbose=True, load_tmp=True).index.values).to_dict()    


# 基金id對應到position id 表
#no_filter_item_id_map = pd.Series(model.pipe.item2nid.get().index.values).to_dict()
item_id_map = pd.Series(rank_n_filter.pipe.item2nid_out.get().index.values).to_dict()


# 顧客基金喜好分數 
#no_filter_score = model.pipe.cust_item_rating_matrix.get()
#score.shape

score = rank_n_filter.pipe.cust_item_rating_matrix_out.get()  
#score = rank_n_filter.pipe.cust_item_rank_matrix_out.get()  
#score = score.shape[1] - score

pred = recommendation_all(score, user_id_map, item_id_map)

# temp (to calculate warm/cold performance)
## Load db connection
rawdata_conn = get_conn('edu')
print("Loading Data...----")
w103_df = load_w103(today, rawdata_conn, span)
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()

#evaluation = Evaluation(today, pred, duration)
#score, upper = evaluation.results()
#print(f'Today: {today} Training-Span: {span} Mean-Precision: {score}\n')

## Evaluate results
print("Evaluating Results...")
evaluation = Evaluation(today, pred, duration, purchase_hist)
warm_user, cold_user = evaluation.warm_cold_list()
warm_pred = {k: v for k, v in pred.items() if k in warm_user}
score, upper_bound = evaluation.results(warm_pred)

buy_old_user, buy_new_user, warm_start_user, cold_start_user = evaluation.purchase_statistic()

print(f'Today: {today} Training-Span: {span} Warm-Start-Users: {warm_start_user} Cold-Start-Users: {cold_start_user} Mean-Precision: {score} Upper-Bound: {upper_bound} \n')

print("Done!") 