import gc 
import numpy as np 
import pandas as pd 
from experiments.old_cust_baseline.process_utils.mat_op import restrict_matrix
def reorganize_by_RR(cust_item_rank_matrix, 
                     cust_item_rating_matrix, 
                     user2nid, item2nid, w106, K = 5,
                     output_dataframe = True):
    '''
    Re-organize the matrices produced from 
    the CF model according to the product risk code. 
    Specifically, for each customer (user) and each rist code, 
    select only the top K ranked funds and rank the funds 
    for each customer according to the ranking of CF model. 
    
    Inputs: 
        - cust_item_rank_matrix: 
            a matrix where Mat[i, j] represent the recommendation 
            ranking of a fund j to user i. 
        - cust_item_rating_matrix: 
            a matrix where Mat[i, j] represent the recommendation 
            rataing of a fund j to user i. 
        - user2nid: 
            a pandas dataframe where each index is the user id and 
            its corresonding column the numeric id (nid) of the user. 
            nid stands for the location of user in the rank/rating 
            matrices. 
        - item2nid: 
            a pandas dataframe where each index is the item id and 
            its corresonding column the numeric id (nid) of an item. 
            nid stands for the location of item in the rank/rating 
            matrices. 
        - w106:  
            a pandas dataframe containing the product_risk_code 
            for each item (funds). 
        - K: 
            the number of funds should be selected for each customer
            and each risk code. 
        - output_dataframe: (bool)
            whether to convert the resulting matrices to pandas dataframe.
            
    Outputs: 
        - If output_dataframe == True:
            - cust_item_recmd: 
                a dataframe containing user ids, item ids, 
                rating between users and items, and rank of items to 
                users. 
        - Else: 
            - rate_matrix:
                a matrix where Mat[i, j] represent the recommendation 
                rating of a fund j (0<=j<=K-1) to user i. The funds are sorted 
                by rank in cust_item_rank_matrix. 
            - item_matrix:
                a matrix where Mat[i, j] represent the item id of 
                a fund j (0<=j<=K-1) that should be 
                recommend to user i. The funds are sorted 
                by rank in cust_item_rank_matrix. 
        
    '''
    w106_risk = w106[
        ['wm_prod_code','prod_risk_code']
    ].copy().rename(
        columns={'wm_prod_code':'item_id'})

    del w106
    gc.collect()

    matrix_dict, item2nid_dict = split_matrix_by_RR(
        [cust_item_rank_matrix, cust_item_rating_matrix], 
        ['rank', 'rate'], item2nid, w106_risk)

    rank_dict, rate_dict, item_dict = select_top_K_by_RR(
        matrix_dict, item2nid_dict, K=K) 
    del matrix_dict, item2nid_dict 
    gc.collect()

    rank_matrix, rate_matrix, item_matrix = concate_over_RR(
        rank_dict, rate_dict, item_dict)
    del rank_dict, rate_dict, item_dict
    gc.collect()

    rank_matrix, rate_matrix, item_matrix = sort_and_rerank(
        rank_matrix, rate_matrix, item_matrix)
    gc.collect()
    
    if output_dataframe:
        cust_item_recmd = to_dataframe(
            rank_matrix, rate_matrix, item_matrix, user2nid)
        del item_matrix, rate_matrix, rank_matrix 
        gc.collect()
        return cust_item_recmd
    else:
        return rate_matrix, item_matrix
def split_matrix_by_RR(matrices, labels, item2nid, w106_risk):
    '''
    Split cust_item_rank_matrix into four matrix, 
    each only contains funds of a risk category. 
    '''
    matrix_dict = dict()
    item2nid_dict = dict()
    for RR in ['RR1', 'RR2', 'RR3', 'RR4', 'RR5']:
        ## [ ] 切RR1~RR5可以在基金矩陣相乘前就做。(但要確認效果) 
        overlapped_items = set(
            w106_risk[w106_risk.prod_risk_code == RR].item_id.tolist()
        ) & set(item2nid.index.tolist())

        if overlapped_items:
            ans_dict = {}
            for label, mat in zip(labels, matrices):
                res_mat, res_item2nid = restrict_matrix(
                    mat,
                    restrict_to=overlapped_items, 
                    to_nid_table=item2nid,
                    mode='col'
                )
                ans_dict[label] = res_mat
            
            matrix_dict[RR] = ans_dict
            item2nid_dict[RR] = res_item2nid
    return matrix_dict, item2nid_dict

def select_top_K_by_RR(matrix_dict, item2nid_dict, K = 5):
    rank_dict = dict()
    rate_dict = dict()
    item_dict = dict()
    for RR in item2nid_dict.keys():
        rank_mat = matrix_dict[RR]['rank']
        rate_mat = matrix_dict[RR]['rate']
        rank_mat, rate_mat, items = select_top_K(
            rank_mat, rate_mat, item2nid_dict[RR], K=K)
        
        rank_dict[RR] = rank_mat 
        rate_dict[RR] = rate_mat 
        item_dict[RR] = items 
    return rank_dict, rate_dict, item_dict
# re_ranked = matrix_item2nid_for_each_RR['RR3']['re-ranked']

def select_top_K(rank_mat, rate_mat, item2nid, K=5):
    item_array = item2nid.index.to_numpy()
    # get nids of top 5 funds 
    select_item_nids = np.apply_along_axis(
        lambda arr: arr.argsort()[:K], 
        1,    
        rank_mat
    )
    # append to user ids 
    col_selector = ColWiseMatrixSelector(select_item_nids)
    selected_rank = col_selector.apply(rank_mat)
    selected_rate = col_selector.apply(rate_mat)
    selected_items = col_selector.broadcast_apply(item_array)
    
    return selected_rank, selected_rate, selected_items

def concate_over_RR(rank_dict, rate_dict, item_dict):
    # concate matrix by RR 
    rank_matrix = np.concatenate([rank_dict[x] for x in item_dict], axis = 1)
    rate_matrix = np.concatenate([rate_dict[x] for x in item_dict], axis = 1)
    item_matrix = np.concatenate([item_dict[x] for x in item_dict], axis = 1)
    return rank_matrix, rate_matrix, item_matrix 

def sort_and_rerank(rank_matrix, rate_matrix, item_matrix):
    # sort by rank 
    col_selector = ColWiseMatrixSelector(rank_matrix.argsort())
    rate_matrix = col_selector.apply(rate_matrix)
    item_matrix = col_selector.apply(item_matrix)
    gc.collect()
    # obtain new rank matrix
    rank_matrix = np.reshape(
        np.repeat([range(1, item_matrix.shape[1] + 1)],
        repeats = item_matrix.shape[0] # 
         ), 
        newshape = item_matrix.shape, # 
        order = 'F'
    )
    return rank_matrix, rate_matrix, item_matrix

def to_dataframe(rank_matrix, rate_matrix, item_matrix, user2nid):
    # squeeze matrix to array 
    user_array = np.repeat(user2nid.index.to_numpy(), 
              repeats = item_matrix.shape[1])

    item_array = np.reshape(item_matrix, 
               newshape = (item_matrix.shape[0] * item_matrix.shape[1]))

    rating_array = np.reshape(rate_matrix, 
               newshape = (rate_matrix.shape[0] * rate_matrix.shape[1]))

    rank_array = np.reshape(rank_matrix, 
               newshape = (rank_matrix.shape[0] * rank_matrix.shape[1]))

    del item_matrix, rate_matrix, rank_matrix 
    gc.collect()
    # building DataFrame from the arrays 
    cust_item_recmd = pd.DataFrame(
        zip(user_array, item_array, rating_array, rank_array))
    del user_array, item_array, rating_array, rank_array
    gc.collect()
    cust_item_recmd.columns = ['cust_no', 'fund_id', 'rating', 'rank']
    return cust_item_recmd

class ColWiseMatrixSelector:
    def __init__(self, selection_matrix): 
        nids = np.arange(0, selection_matrix.shape[0])
        self.rid_cids = np.concatenate(
            [np.reshape(
                nids, newshape = (len(nids), 1)),
             selection_matrix], axis=1)
    def apply(self, mat): 
        selected = np.apply_along_axis(
            lambda arr: mat[arr[0], arr[1:]],
            1,
            self.rid_cids
        )
        return selected
    def broadcast_apply(self, in_array): 
        selected = np.apply_along_axis(
            lambda arr: in_array[arr[1:]],
            1,
            self.rid_cids
        )
        return selected

def select_by_cust_risk(recmd, w107, w106, K = 5):
    cust_item_recmd = recmd.merge(
        w106[['wm_prod_code','prod_risk_code']].rename(
            columns={'wm_prod_code':'fund_id'}), 
        how='left', 
        on=['fund_id'])
    
    dic = {'RR1':1,'RR2':1,'RR3':2,'RR4':3,'RR5':4, None:0}
    cust_item_recmd['prod_risk_code'] = cust_item_recmd[
        'prod_risk_code'].map(dic)
    ### 串顧客風險、過風險等級
    cust_item_recmd = cust_item_recmd.merge(
        w107, on=['cust_no'], how='left')
    cust_item_recmd = cust_item_recmd[
        cust_item_recmd['cust_risk'] >= cust_item_recmd['prod_risk_code']]
    ### 排序 
    cust_item_recmd = cust_item_recmd.sort_values(
        ['cust_no','rank'], ascending=[1,1]) 
    cust_item_recmd['cnt'] = 1
    cust_item_recmd['rank'] = cust_item_recmd.groupby(['cust_no'])['cnt'].cumsum()
    ### 取前 K 檔基金
    output = cust_item_recmd[cust_item_recmd['rank']<=K]
    output = output[['cust_no','fund_id','rank']]
    return output