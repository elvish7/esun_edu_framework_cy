import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def top5_recommendation_user(pred, item_dict, nrec_items = 5):
    return [ item_dict[idx] for idx in pred[:nrec_items] ]

def recommendation_all(score, user_dict, item_dict):

    predictions = defaultdict(list)
    matrix = np.argsort(score, axis=1)[:,::-1]
    n_user, n_item = matrix.shape
    for u_idx, ranklist in tqdm(enumerate(matrix), total=n_user):
        user_id = user_dict[u_idx]
        pred = top5_recommendation_user(ranklist, item_dict, 5)
        predictions[user_id] = pred

    return predictions
