import os
import argparse
import pandas as pd
import numpy as np
from evaluation import Evaluation
from mlaas_tools.config_build import config_set
from db_connection.utils import get_conn
from utils import load_w103, load_w106
from preprocess import convert_data 
from train import *
from recommend import predict_rate
import pickle
from utils_amcf import load_emb_weights

## Configure env
if not os.path.isfile('config.ini'):
    config_set()

## Add params
parser = argparse.ArgumentParser()
parser.add_argument("--date", default='2019-06-30', help="Recommendation date")
parser.add_argument("--train_span", type=int, default=1, help="Training Period")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--epoch", default=10, type=int, help="epoch num")
args = parser.parse_args()
today = args.date
duration = args.eval_duration
# dim = args.dim
epoch = args.epoch

## Load db connection
rawdata_conn = get_conn('edu')
## Load data
print("Loading Data...")
w103_df = load_w103(today, rawdata_conn)
w106_df = load_w106(rawdata_conn)

## pretrained data from lightFM
lightfm_path = 'lightfm_latent/'
item_repts = pickle.load(open(lightfm_path + today +'_item_latents.pkl', 'rb'))
user_repts = pickle.load(open(lightfm_path + today +'_user_latents.pkl', 'rb'))
user_id_map, item_id_map = pickle.load(open(lightfm_path+ today +'_id_map.pkl', 'rb'))
base_model_data = [item_repts, user_repts, user_id_map, item_id_map]

## Intersection of w103 & w106 wrt wm_prod_code
_filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
w106_df_filter = w106_df[_filter]
aspect_col = ['prod_detail_type_code', 'prod_risk_code', 'invest_type'] #'prod_ccy'
_selected_col = ['wm_prod_code'] + aspect_col
w106_df_filter = w106_df_filter[_selected_col]

## data preprocess
ratings, fund, user_n, item_n, user_dict, fund_dict = convert_data(w103_df, w106_df_filter, aspect_col)
asp_n = len(fund.columns)-1
print('total aspects:', asp_n)

## pretrained embedding weights
weights = load_emb_weights(user_dict, fund_dict, base_model_data)
# weights = None
## training
model, val_rmse, cos_sim = model_training(user_n, item_n, asp_n, ratings, fund, epoch, weights)

## Predict top5 funds for each user
print("Predicting...")
# all the users & items
user_list, item_list = ratings['uid'].unique().tolist(), fund['fid'].unique().tolist()
pred = predict_rate(user_list, item_list, model, fund, user_dict, fund_dict)

## Evaluate results
print("Evaluating Results...")
evaluation = Evaluation(today, pred, duration)
score = evaluation.results()
print(f'Today: {today} Mean Precision: {score}\n')

## Save results
# with open('amcf_results' + today + '.txt', 'a') as f_out:
#     f_out.write(f'{today} {score}\n')
#     f_out.write(f'{val_rmse} {cos_sim}\n')