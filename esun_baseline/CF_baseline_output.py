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
from mlaas_tools.config_build import config_set
## Configure env
if not os.path.isfile('config.ini'):
    config_set()

parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
args = parser.parse_args()
today = args.date
duration = args.eval_duration

# ## 模型計算 

#assert today == '2019-06-30'
model = old_cust_CF_baseline(today=today, mode='model')

# 顧客id對應到position id 表
user_mapping = model.pipe.user2nid.get(load_tmp=True).to_dict()['index']
user_mapping = {v: k for (k, v) in user_mapping.items()}

# 基金id對應到position id 表
item_mapping = model.pipe.item2nid.get(load_tmp=True).to_dict()['index'] 
item_mapping = {v: k for (k, v) in item_mapping.items()}

scores = model.pipe.cust_item_rating_matrix.get(load_tmp=True) 
scores = scores.argsort().argsort()

df = pd.DataFrame(scores)
df.rename(index = user_mapping, inplace=True)
df.rename(columns = item_mapping, inplace=True)
df.reset_index(inplace=True)

today = today.replace("-", "")
df.to_csv(f"{today}_esun_cf.txt",  header=None, sep=' ', mode='a')

print(f'Today: {today} n_users: {len(df)}\n')

