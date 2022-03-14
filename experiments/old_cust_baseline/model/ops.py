import gc 
import pandas as pd 
import numpy as np
from sklearn.metrics import pairwise_distances

def calculate_similarity(item_feature):
    '''
    Inputs: 
        - item_feature: 基金特徵表 
    Outputs: 
        - item_matrix: 相似度矩陣 (pd.DataFrame) 

    TODO:
        - [ ] item_matrix改名為 item_similarity_matrix 
        - [ ] allow n_job > 1 in pairwise_distances calculation 
            to speedup the process. 
    '''
    ### 取feature數值
    item_feature_val = item_feature.iloc[:,1:].values
    ### 相似度
    dist = 1-pairwise_distances(item_feature_val, metric='cosine')
    ### 存成df
    item_matrix = pd.DataFrame(dist, columns=item_feature.item_id)
    item_matrix['item_id'] = item_feature.item_id
    return item_matrix 

def limit_recmd_items(item_matrix, w106):
    '''
    By limiting the columns in the item_matrix, 
    we can restrict the items in 
    the cust-item matrix calculated by the 
    `matrix_mul_advance` function, and hence 
    limit the items that can be recommended. 
    
    Input:
        - item_matrix: the item similarity matrix 
        - w106: the table contain the information 
            of whether a fund (item) should be recommended
            in the `can_rcmd_ind` column. 
    Output: 
        - item_matrix: the resulting item_matrix. 
        
    TODO:
        - [ ] 這個限制item_matrix中item的措施應該要在產生item_matrix的時候，
            就要進行了。

    '''
    item_can_recmd_list = w106[
        w106['can_rcmd_ind']==1].wm_prod_code.tolist()
    item_matrix = item_matrix.loc[
        :, item_matrix.columns.isin(['item_id']+item_can_recmd_list)]
    del item_can_recmd_list
    gc.collect()
    return item_matrix