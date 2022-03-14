import math, os, collections
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from SMORe import SMORe 
from TPR import TPR 
import argparse
from local_evaluation import Evaluation
## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--model", default='tpr', help="model type")
parser.add_argument("--train", help="training data path")
parser.add_argument("--evaluation", help="evaluation data path")
parser.add_argument("--item_ft", default='', type=str, help="w106 item static feature data path")
parser.add_argument("--user_ft", default='', type=str, help="cm_customer_m user static feature data path")
args = parser.parse_args()
## Get data path
path = os.getcwd()
train_path =os.path.join(path, 'data', args.train)
evaluation_path =os.path.join(path, 'data', args.evaluation)
item_feature_path =os.path.join(path, 'data', args.item_ft)
user_feature_path =os.path.join(path, 'data', args.user_ft)
## Read data
w103_df = pd.read_csv(train_path)
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()
if args.model == 'tpr':
    w106_df = pd.read_csv(item_feature_path)
    _filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
    w106_df_filter = w106_df[_filter]
    _selected_col = ['wm_prod_code', 'prod_detail_type_code','prod_ccy','prod_risk_code','can_rcmd_ind']
    w106_df_filter = w106_df_filter[_selected_col]
    #cm_customer_m_df = pd.read_csv(user_feature_path)
## Init SMORe
if args.model == 'tpr':
    model = TPR(w103_df, w106_df_filter)
else:
    model = SMORe(w103_df)
## Get user & item emb.
user_emb, item_emb = model.fit(lr=0.05, update_times=500)
## Calculate cosine similarity of every (u, i) pair, n_user * n_item
scores = cosine_similarity(user_emb.fillna(0), item_emb.fillna(0))
## Recommend 5 funds for every user
id2user, prediction = model.user_dict, collections.defaultdict(list)
for i, score in enumerate(tqdm(scores, total=len(scores))):
   user = id2user[i] 
   prediction[user] = [i[1] for i in sorted(zip(score, model.items), reverse=True )][:5]
## Evaluate results
evaluation = Evaluation('', evaluation_path, prediction, purchase_hist)
score, upper = evaluation.results()
print(f'Mean-Precision: {score} Upper-Bound: {upper}\n')

print("Done!") 
