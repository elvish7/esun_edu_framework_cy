import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from collections import defaultdict

def cust_process(df):
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    df[df['children_cnt']>=4] = 4
    # continuous value
    df['age'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], labels=False)
    df['cust_vintage'] = pd.qcut(df['cust_vintage'], 4, labels=False, duplicates='drop')
    return df

def w106_process(df):
    # discard categorization
    discard_condition = {'counterparty_code': 100, 'mkt_rbot_ctg_ic': 200, 'prod_ccy': 500}
    for col, n in discard_condition.items(): 
        df.loc[df[col].value_counts()[df[col]].values<n, col] = col+'_other'
    # convert int to categorical
    df['high_yield_bond_ind'] = df['high_yield_bond_ind'].map({'Y': 'high_yield', 'N': 'not_high_yield'})
    df['can_rcmd_ind'] = df['can_rcmd_ind'].map({1:'can_rcmd', 0: 'can_rcmd_N'})
    return df

def create_all_feature_pairs(features):
    """
    Create list containing all possible feature_name,feature_value pairs
    """
    feature_pairs = []
    col = []
    unique_features = []
    for column in features.iloc[: , 1:]: #drop the first column
        col += [column]*len(features[column].unique())
        unique_features += list(features[column].unique())
    for x,y in zip(col, unique_features):
        pair = str(x)+ ":" +str(y)
        feature_pairs.append(pair)
    return feature_pairs
    

def concat_feature_colon_value(header, my_list):
    """
    Takes as input a list and prepends the columns names to respective values in the list.
    For example: if my_list = [1,1,0,'del'],
    resultant output = ['f1:1', 'f2:1', 'f3:0', 'loc:del']
   
    """
    result = []
    for x,y in zip(header,my_list):
        res = str(x) +""+ str(y)
        result.append(res)
    return result

def build_feature_tuples(features):
    """
    One user/item tuple: (item id, {feature name: feature weight})
    Returns a list of tuples
    """
    feature_subset = features.iloc[: , 1:] #drop the first column
    header = [f+':' for f in feature_subset.columns.tolist()]
    feature_list = [list(f) for f in feature_subset.values]
    feature_colon_value_list = []
    for ft in feature_list:
        feature_colon_value_list.append(concat_feature_colon_value(header, ft))
    feature_tuples = list(zip(features.iloc[:,0], feature_colon_value_list))     
    return feature_tuples

def top5_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict, user_features=None, item_features=None,threshold = 0,nrec_items = 5):
    
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items), user_features=user_features, item_features=item_features))
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    #known_items = list(pd.Series(interactions.loc[user_id,:] \
    #                             [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    
    #scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    pred = [k for k, v in item_dict.items() if v in return_score_list]
    
    return user_id, pred

def recommendation_all(model, intersections, user_li, user_dict, item_dict, user_features, item_features):

    predictions = defaultdict(list)

    for u in tqdm(user_li, total=len(user_li)):
        user_id, pred = top5_recommendation_user(model, intersections, u, user_dict, item_dict, user_features, item_features)
        predictions[user_id] = pred

    return predictions
