import numpy as np 
import pandas as pd 
import scipy 
from scipy.stats.stats import rankdata

from experiments.old_cust_baseline.process_utils.mat_op import convert_to_nid_on_user_item_matrix
from experiments.old_cust_baseline.process_utils.mat_op import convert_to_nid

def get_dig_cnt(txn_dig):
    '''
    取得各個基金數位通路交易之熱門排序(依顧客三個月內買過哪支基金)
    
    TODO: 
        - [ ] get_dig_cnt 改為 get_hot_rank_table 
    '''
    ### 
    txn_dig_ind = txn_dig[['cust_no','wm_prod_code']].copy()
    #txn_dig_ind = txn_dig_ind.rename(columns={'cust_no':'user_id', 'wm_prod_code':'item_id'})
    txn_dig_ind = txn_dig_ind.drop_duplicates(
        subset=['cust_no','wm_prod_code'], keep='last')
    txn_dig_ind =  txn_dig_ind.reset_index(drop=True)
    ### 依基金交易unique人數排序
    dig_cnt = txn_dig_ind.groupby(['wm_prod_code'])[['cust_no']].count(
        ).rename(columns={'cust_no':'txn_cust_cnt'})
    dig_cnt = dig_cnt.sort_values(by=['txn_cust_cnt'], ascending=False)
    dig_cnt = dig_cnt.reset_index()
    dig_cnt['no'] = 1
    dig_cnt['item_hot_rank'] = dig_cnt['no'].cumsum()
    dig_cnt = dig_cnt[['wm_prod_code','item_hot_rank']].copy()
    dig_cnt = dig_cnt.rename(columns={'wm_prod_code':'item_id'})
    dig_cnt['item_hot_rank'] = dig_cnt['item_hot_rank'].fillna(
        max(dig_cnt['item_hot_rank'])+1)
    ### 近三個月沒有被購買者合併後會nan,給一個排序最後面的數值(9999)
    return dig_cnt


def convert_rating_to_rank(matrix):
    return rankdata(1. - matrix, method='min', axis=0) 

def re_rank_by_hotness_n_novalty(cust_item_rank_matrix, user2nid, item2nid, 
            dig_cnt=None, cust_txn_exclude=None):
    _cust_item_rank_matrix = cust_item_rank_matrix.copy()
    _user2nid = user2nid.copy()
    _item2nid = item2nid.copy()
    ## TODO: [V] 以下程式放到一個 re_ranking的.py中 
    # 用  dig_cnt 的 item_hot_rank 來 做 次順位的排序 
    if dig_cnt is not None:
        hot_rank_matrix = _build_hot_rank_matrix(dig_cnt, _user2nid, _item2nid)
        _cust_item_rank_matrix = _merge_ranking_matrix(
            _cust_item_rank_matrix, hot_rank_matrix)
    
    if cust_txn_exclude is not None:
        # 把 cust_txn_exclude 中 'exclude_ind' == 0 的排到前面，
        # 其他的排到後面。
        cust_item_exclude_matrix = _convert_cust_txn_exclude_to_sparse_matrix(
            cust_txn_exclude, _user2nid, _item2nid)
        print(cust_item_exclude_matrix.shape)
        print(_cust_item_rank_matrix.shape)
        _cust_item_rank_matrix = _merge_ranking_matrix(
            cust_item_exclude_matrix, 
            _cust_item_rank_matrix)
        
    return _cust_item_rank_matrix

def _build_hot_rank_matrix(dig_cnt, user2nid, item2nid):
    # print('item2nid:', item2nid) 
    hot_rank_table = item2nid.reset_index()
    # print('dig_cnt:', dig_cnt) 
    # print('hot_rank_table', hot_rank_table)
    hot_rank_table = hot_rank_table.merge(
        dig_cnt, on='item_id', how='left')
    hot_rank_table = hot_rank_table.fillna(
        max(hot_rank_table.item_hot_rank) + 1)
    n_user, n_item = len(user2nid), len(item2nid)
    hot_rank_matrix = np.repeat([
        hot_rank_table.item_hot_rank.to_numpy()
    ], repeats = n_user, axis=0)
    return hot_rank_matrix

def _merge_ranking_matrix(main_mat, secondary_mat):
    assert main_mat.shape == secondary_mat.shape 
    # scale down the secondary ranking matrix 
    max_value = np.max(secondary_mat)
    secondary_mat = secondary_mat/(max_value + 1.)
    return rankdata(
        main_mat + secondary_mat, method='min', axis=0) 



def _convert_cust_txn_exclude_to_sparse_matrix(cust_txn_exclude, user2nid, item2nid):
    cust_txn_exclude = cust_txn_exclude[(
            cust_txn_exclude.item_id.isin(item2nid.index)
        ) & (
            cust_txn_exclude.user_id.isin(user2nid.index)
        )
    ]

    cust_txn_exclude, _, _ = convert_to_nid_on_user_item_matrix(
        cust_txn_exclude, 
        user2nid, 
        item2nid
    )

    if max(cust_txn_exclude.user_id) != max(user2nid['index']) or max(
        cust_txn_exclude.item_id) != max(item2nid['index']):
        max_row = pd.DataFrame((max(user2nid['index']), max(item2nid['index']), 0)).T
        max_row.columns = ['user_id', 'item_id', 'exclude_ind']
        cust_txn_exclude = pd.concat([cust_txn_exclude, max_row])

    cust_item_exclude_matrix = scipy.sparse.csr_matrix(
        (cust_txn_exclude.exclude_ind, (
        cust_txn_exclude.user_id, 
        cust_txn_exclude.item_id))
    )
    return cust_item_exclude_matrix

def filter_by_items(cust_item_rank_matrix, 
                                    item2nid, restricted_item_list):
    indices = item2nid.reset_index().item_id.isin(
        restricted_item_list).tolist()  
    filtering_indices = item2nid[indices]['index'].tolist()   
    filtered_cust_item_rank_matrix = cust_item_rank_matrix[
        :, filtering_indices] 
    filtered_item2nid = pd.DataFrame(item2nid[indices]['index']) 
    filtered_item2nid['index'] = range(len(filtered_item2nid)) 
    return filtered_cust_item_rank_matrix, filtered_item2nid 



def re_rank(cust_item_rank_matrix): 
    _cust_item_rank_matrix = cust_item_rank_matrix.copy()
    # Re_rank  
    cust_item_rank_matrix = rankdata(
        _cust_item_rank_matrix, method='min', axis=0) 
    return cust_item_rank_matrix
