import os
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
from evaluation import Evaluation
from mlaas_tools.config_build import config_set
from db_connection.utils import get_conn
from utils import recommendation_all, load_w103, load_w106, load_cust_pop, create_all_feature_pairs, build_feature_tuples, load_cust_pop_0205 #weighted
## Configure env
if not os.path.isfile('config.ini'):
    config_set()
## Add params
parser = argparse.ArgumentParser()
parser.add_argument("--date", help="Recommendation date")
parser.add_argument("--train_span", type=int, default=18, help="Training Period")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--user_ft", help="Use user features", action='store_true')
parser.add_argument("--item_ft", help="Use item features", action='store_true')
parser.add_argument("--eval_mode", default='warm', type=str, help="choose warm or cold")
args = parser.parse_args()
today = args.date
span = args.train_span
duration = args.eval_duration
eval_mode = args.eval_mode

## Load db connection
rawdata_conn = get_conn('edu')
path = '../../raw_datasets/'+today+'_'+str(span)
if not os.path.exists(path):
    os.makedirs(path)
## Load data
print("Loading Data...")
w103_df = load_w103(today, rawdata_conn, span)
w103_df.to_csv(path+'/train_w103.csv')
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()

