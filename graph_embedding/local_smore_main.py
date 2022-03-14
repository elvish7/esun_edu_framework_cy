import math, os, collections
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from SMORe import SMORe 
import argparse
from local_evaluation import Evaluation
## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="training data path")
parser.add_argument("--evaluation", help="evaluation data path")
args = parser.parse_args()
## Get data path
path = os.getcwd()
train_path =os.path.join(path, 'data', args.train)
evaluation_path =os.path.join(path, 'data', args.evaluation)
## Read data
w103_df = pd.read_csv(train_path)
purchase_hist = w103_df.groupby("cust_no")["wm_prod_code"].apply(lambda x: list(set(x.values.tolist()))).to_dict()
## Init SMORe
smore = SMORe(w103_df)
## Get user & item emb.
user_emb, item_emb = smore.fit(mode='warp', lr=0.05, update_times=200)
## Calculate cosine similarity of every (u, i) pair, n_user * n_item
scores = cosine_similarity(user_emb, item_emb)
## Recommend 5 funds for every user
id2user, prediction = smore.user_dict, collections.defaultdict(list)
for i, score in enumerate(tqdm(scores, total=len(scores))):
   user = id2user[i] 
   prediction[user] = [i[1] for i in sorted(zip(score, smore.items), reverse=True )][:5]
## Evaluate results
evaluation = Evaluation('', evaluation_path, prediction, purchase_hist)
score, upper = evaluation.results()
print(f'Mean-Precision: {score} Upper-Bound: {upper}\n')

print("Done!") 
