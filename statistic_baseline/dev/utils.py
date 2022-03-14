import pandas as pd
import numpy as np
import datetime
import random
from tqdm import tqdm
from collections import defaultdict

def top5_recommendation_user(popularity_dict, user_id, 
                               item_list, mode):
    if mode == 'random':
        pred = random.sample(item_list, 5)

    else: 
        pred = [k for k, v in popularity_dict.items()]
    
    return pred

def recommendation_all(popularity_dict, user_list, item_list, mode='random'):

    predictions = defaultdict(list)

    for user_id in tqdm(user_list, total=len(user_list)):
        pred = top5_recommendation_user(popularity_dict, user_id, item_list, mode)
        predictions[user_id] = pred

    return predictions
