import os
import argparse
import pandas as pd
import numpy as np
from local_evaluation import Evaluation
from utils import recommendation_all
## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="training data path")
parser.add_argument("--evaluation", help="evaluation data path")
parser.add_argument("--mode", help="random or popularity")
args = parser.parse_args()
mode = args.mode
## Get data path
path = os.getcwd()
train_path =os.path.join(path, 'data', args.train)
evaluation_path =os.path.join(path, 'data', args.evaluation)
## Read data
w103_df = pd.read_csv(train_path)
## Get user/item list for predictions
user_list = w103_df['cust_no'].unique().tolist()
item_list = w103_df['wm_prod_code'].unique().tolist()
## Popularity
popularity_dict = {}
if mode == 'popularity':
    popularity_dict = w103_df.groupby("wm_prod_code")["wm_prod_code"].count().sort_values(ascending=False).head(5).to_dict()
## Recommend each user's top5 funds
pred = recommendation_all(popularity_dict, user_list, item_list, mode=mode)
## Evaluate each user's precision@5
evaluation = Evaluation('', evaluation_path, pred)
score = evaluation.results()
print(f'Mean Precision: {score}\n')
