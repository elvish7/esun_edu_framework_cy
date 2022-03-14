import gc 
import numpy as np 
import pandas as pd 
import scipy


def matrix_mul(cust_txn_rating, item_matrix):
    '''
    Inputs: 
        - cust_txn_rating 顧客最新一次交易回推一年的各基金申購交易占比 
        e.g., 
        |        | user_id                  | item_id   |   rating |\n
        |-------:|:-------------------------|:----------|---------:|\n
        | 453324 | ++++2rDzc1I9amUGBhmxLA== | FCG5      | 0.252261 |\n
        | 352670 | ++++2rDzc1I9amUGBhmxLA== | 5950      | 0.267704 |\n
        | 273746 | ++++2rDzc1I9amUGBhmxLA== | 9117      | 0.232072 |\n
        
        - item_matrix 基金相似度矩陣 
        e.g., 
        |    |      1101 |      1102 |      1103 |  ...      |  item_id |
        |---:|----------:|----------:|----------:|----------:|---------:|
        |  0 | 1         | 0.0618653 | 0.185933  | ...       | FCG5     |
        |  1 | 0.0618653 | 1         | 0.108893  | ...       | 5950     |
        |  2 | 0.185933  | 0.108893  | 1         | ...       | 9117     |
        
    Outputs: 
        - cust_item_rank 計算後的user對item的rating表 
        |    | user_id                  |   item_id |    rating |
        |---:|:-------------------------|----------:|----------:|
        |  0 | ++++2rDzc1I9amUGBhmxLA== |      1101 | 0.0138784 |
        |  1 | ++++2rDzc1I9amUGBhmxLA== |      1102 | 0.0213596 |
        |  2 | ++++2rDzc1I9amUGBhmxLA== |      1103 | 0.0204618 |
        |  3 | ++++2rDzc1I9amUGBhmxLA== |      1104 | 0.0229548 |
        |  4 | ++++2rDzc1I9amUGBhmxLA== |      1107 | 0.0273924 |
    TODO: 
        - [ ] 現有作法很耗費記憶空間和速度: 改用sparse matrix存cust_txn_rating，numpy matrix存item_matrix，然後用numpy的matrix multiplication 
            - cust_txn_rating 和 item_matrix left join 會存很多重複的item_matrix中的行。
        - [ ] 用left join 會產生很多NA的item rows 
        - [ ] directly output numpy matrix and take numpy item_matrix as input. 
        
    '''
    assert set(item_matrix.item_id) == set([col for col in item_matrix.columns if col != 'item_id'])
    ### 合併
    cust_item_rating = pd.merge(cust_txn_rating, item_matrix, on = ['item_id'], how='left')

    ### rating乘上基金相似度vector作為加權
    item_list = [x for x in item_matrix.columns if x not in ['item_id']]
    del item_matrix
    gc.collect()
    for item in item_list:
        cust_item_rating[item] = (cust_item_rating['rating'].values)*(cust_item_rating[item].values)
    ### sum rating
    cust_item_rating = cust_item_rating.groupby(['user_id'])[item_list].sum().reset_index()
    ### 轉置
    cust_item_rank = cust_item_rating.set_index(['user_id']).stack()
    del cust_item_rating
    gc.collect()
    cust_item_rank = cust_item_rank.reset_index(name='rating')
    cust_item_rank = cust_item_rank.rename(columns={'level_1':'item_id'})
    return cust_item_rank

def matrix_mul_advance(cust_txn_rating, item_matrix, output_generator = False):
    '''
    Inputs: 
        - cust_txn_rating 顧客最新一次交易回推一年的各基金申購交易占比 
        e.g., 
        |        | user_id                  | item_id   |   rating |\n
        |-------:|:-------------------------|:----------|---------:|\n
        | 453324 | ++++2rDzc1I9amUGBhmxLA== | FCG5      | 0.252261 |\n
        
        
        - item_matrix 基金相似度矩陣 
        e.g., 
        |    |      1101 |      1102 |      1103 |  ...      |  item_id |
        |---:|----------:|----------:|----------:|----------:|---------:|
        |  0 | 1         | 0.0618653 | 0.185933  | ...       | FCG5     |
        - output_generator 
            若是True，輸出一個產生(rating, user_id, item_id) 的generator。
            若是False，輸出 user_item_matrix, user2nid_mapper, item2nid_mapper
        
    Outputs: 
        - If output_generator == False: user_item_matrix, user2nid_mapper, item2nid_mapper 
        - If output_generator == True: generator: 一個產生(rating, user_id, item_id) 的generator
    '''
    _cust_txn_rating = cust_txn_rating.copy(deep=True)
    _item_matrix = item_matrix.copy()
    _item_matrix = drop_and_pad_item_to_item_matrix(
        _item_matrix, _cust_txn_rating)
    # convert user and item ids to nids 
    _cust_txn_rating, user2nid_mapper, item2nid_mapper_tmp = \
        convert_to_nid_on_user_item_matrix(_cust_txn_rating)
    _item_matrix, item2nid_mapper = convert_to_nid_on_item_matrix(
        _item_matrix, item2nid_mapper_tmp)
    gc.collect()
    # Apply matrix multiplication: 
    ## In our case, csr_matrix is faster than coo/csc_matrix in terms of 
    ## matrix multiplication. 
    user_item_sparse_mat = scipy.sparse.csr_matrix(
        (_cust_txn_rating.rating, (
            _cust_txn_rating.user_id, 
            _cust_txn_rating.item_id))
    )
    del _cust_txn_rating
    gc.collect() 
    item_np_matrix = _item_matrix.set_index('item_id').to_numpy()
    del _item_matrix
    gc.collect() 
    # adop matrix multiplication 
    assert user_item_sparse_mat.shape[1] == item_np_matrix.shape[0]
    user_item_matrix = user_item_sparse_mat @ item_np_matrix
    assert user_item_matrix.shape == (user_item_sparse_mat.shape[0], item_np_matrix.shape[1])
    del user_item_sparse_mat, item_np_matrix
    gc.collect()
    # print('item2nid_mapper:', item2nid_mapper)
    if output_generator:
        rating, user_ids, item_ids = get_table_series(
            user_item_matrix, user2nid_mapper, item2nid_mapper)
        return zip(rating, user_ids, item_ids)
    else:
        return user_item_matrix, user2nid_mapper, item2nid_mapper
    
def drop_and_pad_item_to_item_matrix(item_matrix, cust_txn_rating):
    '''
    Let item_matrix has rows with items 
    matched to those items in cust_txn_rating.
    '''
    item_should_be_added = list(
        set(
        cust_txn_rating.item_id
    ) - set(
        item_matrix.item_id
    ))

    item_should_be_removed = list(
        set(
        item_matrix.item_id
    ) - set(
        cust_txn_rating.item_id
    ))

    item_matrix = item_matrix.set_index('item_id') 
    item_matrix = item_matrix.T 

    for col in item_should_be_removed:
        del item_matrix[col] 
    for col in item_should_be_added:
        item_matrix[col] = 0.0
    item_matrix = item_matrix.T.reset_index()
    gc.collect()
    return item_matrix 

def get_to_nid_mapping_table(id_column):
    id_table = pd.DataFrame(id_column.unique()) 
    id_table.columns = [id_column.name] 
    gc.collect()
    return id_table.reset_index().set_index(id_column.name)

def convert_to_nid(table, column_name, to_nid_mapper = None):
    if to_nid_mapper is None:
        to_nid_mapper = get_to_nid_mapping_table(table[column_name])
    table[column_name] = to_nid_mapper.loc[table[column_name]].values
    gc.collect()
    return to_nid_mapper

def convert_to_nid_on_user_item_matrix(cust_txn_rating, user2nid = None, item2nid = None):
    user2nid = convert_to_nid(cust_txn_rating, 'user_id', user2nid)
    item2nid = convert_to_nid(cust_txn_rating, 'item_id', item2nid)
    gc.collect()
    return cust_txn_rating, user2nid, item2nid

def convert_to_nid_on_item_matrix(item_matrix, item2nid):
    assert 'item_id' in item_matrix.columns
    
    _ = convert_to_nid(item_matrix, 'item_id', item2nid)
    
    # transpose 
    item_matrix = item_matrix.set_index('item_id').T.reset_index()
    item_matrix.rename(columns={'index':'item_id'}, inplace=True)
    
    assert 'item_id' in item_matrix.columns
    sim_item2nid = convert_to_nid(item_matrix, 'item_id')
    item_matrix.sort_values('item_id', inplace=True) 
    # transpose 
    df = item_matrix.set_index('item_id').T
    item_matrix = df.reset_index().rename(columns={df.index.name:'item_id'})
    
    item_matrix.sort_values('item_id', inplace=True) 
    return item_matrix, sim_item2nid

def convert_user_item_matrix_to_table(
    user_item_matrix, user2nid_mapper, item2nid_mapper):
    n_user, n_item = user_item_matrix.shape
    rating = np.reshape(
        user_item_matrix, newshape = n_user * n_item)
    cust_item_rank = pd.DataFrame([rating])
    cust_item_rank.columns = ['rating']
    del user_item_matrix, rating
    gc.collect()
    
    user_ids = np.repeat(
        user2nid_mapper.index.to_numpy(), 
        n_item)
    cust_item_rank['user_id'] = user_ids
    del user_ids, user2nid_mapper
    gc.collect()
    item_ids = np.reshape(np.repeat(
        [item2nid_mapper.index.to_numpy()], 
        n_user, axis=0),
               newshape=(n_user*n_item,)
              )
    cust_item_rank['item_id'] = item_ids
    del item_ids, item2nid_mapper
    gc.collect()
    
    return cust_item_rank

def get_table_series(user_item_matrix, user2nid_mapper, item2nid_mapper):
    n_user, n_item = user_item_matrix.shape
    rating = np.reshape(
        user_item_matrix, newshape = n_user * n_item)
    del user_item_matrix
    gc.collect()

    user_ids = np.repeat(
        user2nid_mapper.index.to_numpy(), 
        n_item)
    del user2nid_mapper
    gc.collect()
    
    item_ids = np.reshape(np.repeat(
        [item2nid_mapper.index.to_numpy()], 
        n_user, axis=0),
               newshape=(n_user*n_item,)
              )
    del item2nid_mapper
    gc.collect()
    return rating, user_ids, item_ids


def convert_matrix_to_dataframe(matrix, row2nid, col2nid):
    dataframe = pd.DataFrame(matrix) 
    dataframe.columns = col2nid.index    
    index_name = row2nid.index.name  
    dataframe[index_name] = row2nid.index 
    dataframe = dataframe.set_index(index_name) 
    return dataframe


def restrict_matrix(matrix, restrict_to, to_nid_table, mode='row'):
    '''
    - Input: 
        - matrix: a numpy matrix of shape (#row, #col), where 
            each element is a recmd rating between a row and an col.
            
        - restrict_to: a list of cols or rows upon which the matrix 
            should be restricted. 
        
        - to_nid_table: a pd.DataFrame containing mapping from 
            col/row id to number id, which identicate the location 
            of col/row in the matrix. 
        
        - mode: "row" or "col", indicating whether to 
            restrict the matrix by row or by col. 
    - Output: 
        - new_matrix: the resulting restricted matrix. 
        - new_nid_table: the restricted to_nid_table
    '''
    assert mode == 'col' or mode == 'row'
    if restrict_to == None:
        return matrix, to_nid_table
    else:
        if mode =='row':
            new_matrix = matrix[np.squeeze(to_nid_table.loc[restrict_to].to_numpy()), :]
        if mode =='col':
            new_matrix = matrix[:, np.squeeze(to_nid_table.loc[restrict_to].to_numpy())]
        new_nid_table = to_nid_table.loc[restrict_to].copy()
        new_nid_table[new_nid_table.columns[0]] = list(range(len(new_nid_table)))
        return new_matrix, new_nid_table 