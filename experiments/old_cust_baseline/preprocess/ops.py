import pandas as pd 
import numpy as np 
import gc 

def fix_empty_string_on_column(df, column=None):
    # 基金市場(有一筆空值，已補)
    assert column 
    df_series = df[column]
    df[column] = df_series.replace(to_replace='', value='F0000')
    return df

def rename_column(table, columns=None):
    assert isinstance(columns, dict) 
    table = table.rename(columns=columns) 
    return table

def extract_cat_attrs(w106, id_name=None, attr_cols=[], one_hot_encode=True):
    '''
    Extract specific categorical item attributes 
        and one-hot encode tham if nessecery. 
    
    Inputs: 
     - id_name: the name of the item id column 
     - attr_cols: a list of names of the columns that need to be extracted. 
     - fix_cols: a list of names of the columns that need to be fixed. 
     - fix_methods: a list of methods for fixing the columns in `fix_null_cols` 
     
    Output:
     - item_attr: the one-hot encoded attributes, in the form of pandas.DataFrame. 
     
    Example: 
    item_attr = extract_cat_attrs(
        w106,
        id_name='wm_prod_code',
        attr_cols = ['mkt_rbot_ctg_ic','prod_detail_type_code','prod_ccy','prod_risk_code'], 
        fix_cols=['mkt_rbot_ctg_ic'],
        fix_methods=[fix_empty_string]
    )
    def fix_empty_string(df_series):
        # 基金市場(有一筆空值，已補)
        return df_series.replace(to_replace = '', value = 'F0000')
        
    TODO: 
     - [X] Move Data Cleaning Procedure to the w106 load function 
     - [X] Move the last Re-Naming of item id name to the w106 load function 
     - [X] Make one-hot encoding optional. 
    '''
    assert id_name
    assert attr_cols 
    ### 依欄位類型拆df
    df_id = w106[[id_name]].copy()
    df_cat = w106[attr_cols].copy()
    if one_hot_encode:
        ### one-hot encode 
        df_cat_dummy = pd.get_dummies(df_cat).astype(int)
        df_cat_dummy.columns = map(str.lower, df_cat_dummy.columns)
        ### 併回
        item_attr = pd.concat([df_id, df_cat_dummy], axis=1)
    else:
        item_attr = pd.concat([df_id, df_cat], axis=1)
    
    return item_attr

def encode_net_average_price(w118):
    '''
    對淨值進行處理 : 
        1) 抓出各基金淨值之量化統計數值型特徵 
        2) 對各特徵進行離散化
        3) 進行one-hot encoding 

    Inputs: 
        - w188 淨值資料表
    Outputs: 
        - 轉換後淨值資料 
    Example: 

        item_nav = preprocess_item_net_average_price(w118)

    TODO:
        - [ ] w118裡面有很多重複的rows ?? => 資訊處正在處理 #On-Hold 
        - [ ] w118的型態轉換和重複row的去除要在load_w118做完 #Next-up (新版本可以這樣做) 
        - [ ] 'product_code'轉'item_id'在load_w118完成
        - [ ] 淨值特徵經過description抽取統計量化指標後，會再進行離散化以及one-hot encoding的過程。
        - [ ] 前面提到的one-hot-encoding的過程要進行模組化。
        - [ ] 未來淨值可以用time-series模組來處理。
    '''
    ### 轉時間型別
    # w118['nav_date'] = pd.to_datetime(w118['nav_date'])
    # 轉型別
    w118['product_code'] = w118['product_code'].astype(str)
    w118['purchase_val'] = w118['purchase_val'].astype(float)
    # w118['redeem_val'] = w118['redeem_val'].astype(float)
    ### 跑 describe: 產生 count、mean、std、min、25%、50%、75%、max
    w118_describe = w118.groupby(['product_code'])['purchase_val'].describe()
    ### reset_index
    w118_describe = w118_describe.reset_index()
    ### std為nan者補0
    w118_describe['std'] = w118_describe['std'].fillna(0)
    ### 欄位型別
    w118_describe.columns = w118_describe.columns.astype(str)
    ### rename
    w118_describe = w118_describe.rename(columns={'product_code':'item_id'})

    #==================特徵轉換======================================
    ### 複製一份存淨值變化註記
    w118_ind = w118_describe[['item_id']].copy()
    ### 淨值變化註記function
    ### 跑淨值變化註記function
    features_nav = [
        'count', 
        'mean', 'std', 'min', 
        '25%', '50%', '75%', 
        'max'
    ]
    for feature in features_nav:
        w118_ind[feature] = _nav_ind_quar(w118_describe[feature])
    # 進行 one-hot encoding 
    ### 依欄位類型拆df
    df_id = w118_ind[['item_id']].copy()
    df_nav = w118_ind[features_nav].copy()
    ### get dum
    df_nav_dummy = pd.get_dummies(df_nav)
    ### 併回
    item_nav = pd.concat([df_id, df_nav_dummy], axis=1)
    return item_nav



def _nav_ind_quar(describ):
    '''
    把淨值的min, max, mean, 25%, 50%, 75%等description數值，離散化成4級。
    離散化的分界為該指標於整個基金所計算出的Q1,Q2,Q3。

    TODO: 
        - [ ] Input 一個pandas series而不要是一個column name而已。
    '''
    col_name = describ.name
    val_max = describ.max()
    val_min = describ.min()
    # 四分位數
    quar = np.percentile(describ, [25,50,75])
    q1 = quar[0]
    q2 = quar[1]
    q3 = quar[2]
    conditions = [
        (val_min <= describ) & (describ < q1),
        (q1 <= describ) & (describ < q2),
        (q2 <= describ) & (describ < q3),
        (q3 <= describ) & (describ <= val_max)]
    choices = [
        col_name+'_flag1', 
        col_name+'_flag2', 
        col_name+'_flag3',
        col_name+'_flag4']
    return np.select(conditions, choices, default = col_name+'_flag0')


def extract_dig_channel_purchase(w103, topsales_start_dt=None, topsales_end_dt=None):
    '''
    篩選出近三個月數位通路交易
    Inputs: 
        - w103: 基金購買紀錄
        - topsales_start_dt: 資料起算日
        - topsales_end_dt: 資料結算日
    Outputs: 
        - txn_dig: 數位通路基金購買紀錄
    '''
    txn_dig = w103[(w103['txn_channel_code'].isin(['B0','B1','B2'])) & (w103['dta_src']!='A0')].copy()
    ### 轉時間型別
    txn_dig['txn_dt'] = pd.to_datetime(txn_dig['txn_dt'])
    txn_dig = txn_dig[(txn_dig['txn_dt']>=topsales_start_dt) & (txn_dig['txn_dt']<=topsales_end_dt)].copy()
    txn_dig = txn_dig.reset_index(drop=True) 
    return txn_dig 


def extract_purchase_amount_rank(txn_dig, max_rank = 50):
    '''
    * 取得基金交易金額排名
    - 同一檔基金的總交易金額相加後，進行排序，並挑出前50個並且標上rank，最後再適時把rank做one-hot encoding。

    Inputs: 
        - txn_dig: 數位通路基金購買紀錄
        - max_rank: 考量的最大rank數

    Outputs: 
        - ranking_table: 基金交易金額排序特徵

    TODO:
        - [ ] ranking 不應該用one-hot encoding的方式表示，因為其有大小順序的差異
               (若是不用one-hot encoding也不用擔心需要取前50個來限制維度) 
        - [ ] wm_prod_code 轉 item_id 應提早做完
        - [ ] item_top改名為item_ranking
        - [ ] one-hot encoding 應該要是optional的。
    '''
    _txn_dig = txn_dig[['wm_prod_code','txn_amt']].copy()
    _txn_dig = _txn_dig.groupby(['wm_prod_code'])[['txn_amt']].sum().sort_values(by=['txn_amt'], ascending=False)
    _txn_dig = _txn_dig.reset_index()
    _txn_dig = _txn_dig[:max_rank]
    _txn_dig['no'] = 1
    _txn_dig['rank'] = _txn_dig['no'].cumsum()
    del _txn_dig['no'], _txn_dig['txn_amt']
    ###  併上註記
    # - 把rank 做 one-hot encoding. 
    rank_dummy =  pd.get_dummies(_txn_dig['rank']).astype(int)
    rank_dummy.columns = rank_dummy.columns.astype(str)
    # 把　item-id/rank和one-code encode後的rank 並起來。
    del _txn_dig['rank']
    item_ranking = pd.concat([_txn_dig, rank_dummy], axis=1) 
    item_ranking = item_ranking.rename(
        columns={'wm_prod_code':'item_id'})
    return item_ranking


def obtain_user_item_matrix(w103, drop_duplicate = True):
    '''
    註記顧客一年半內 是否購買過該基金 或購買過幾筆該基金 (如果drop_duplicate==False)。
    
    Inputs: 
        - w103 : the phurchase table 
        - drop_duplicate : whether or not to drop the same purchase for 
            each user. 
    Outputs: 
        - txn_matrix : the user-item matrix. If drop_duplicate == True, 
            a boolean rating matrix is resulted; o.w., 
            a purchase count matrix is resulted. 

    TODO: 
        - [V] what w103, txn, txn_matrix look like. 
        - [V] describe w103, and txn_matrix in a better way. 
              - w103: phurchase table 
              - txn_matrix : user-to-item matrix, where each element encode 
                  the purchase affinity (whether not the user purchase the 
                  item or the number of times he/she purchase.) 
        - [V] add option for determinate whether repeated phurchase shall be 
                dropped. 
        - [ ] use some kind of sparse matrix rather than pandas to store 
                the user-item matrix. 
        - [ ] 欄位名稱的變更改在load_w106/103...的時候做
    '''
    txn = w103[['cust_no','wm_prod_code']].copy()
    txn = txn.rename(columns={'cust_no':'user_id', 'wm_prod_code':'item_id'})
    # 去除重複購買
    if drop_duplicate:
        txn = txn.drop_duplicates(subset=['user_id','item_id'], keep='last')
        txn =  txn.reset_index(drop=True)
    # 新增 rating欄位，作為是否有購買之註記 
    txn['rating'] = 1
    rate_grouped_txn = txn.groupby(['user_id','item_id'])['rating'].sum()
    ### 轉置
    txn_matrix = rate_grouped_txn.unstack('user_id').fillna(0).astype(int)
    return txn_matrix.rename(columns=str).reset_index()

def merge_fix_features(
    txn_matrix, 
    item_attr, 
    item_ranking, 
    item_nav):
    result = _merge_features(
        [txn_matrix, 
        item_attr, 
        item_ranking, 
        item_nav])
    return result

def _merge_features(tables = []):
    '''
    把基金特徵表進行合併

    Inputs: 
        - tables: 要被合併的資料表
    Outputs: 
        - item_feature: 合併特徵資料表

    Example: 
        item_feature = merge_features([
            txn_matrix,
            item_attr,
            item_ranking,
            item_nav
        ])
    '''
    assert tables
    for i, table in enumerate(tables):
        if i == 0: 
            pass 
        elif i == 1:
            item_feature = tables[0].merge(table, on='item_id', how='left') 
        else:
            item_feature = item_feature.merge(table, on='item_id', how='left') 
    item_feature = item_feature.fillna(0)
    return item_feature

def normalize_amt_by_cust(cust_txn_amt):
    """
    Inputs: 
        - cust_txn_amt: 顧客最新一次交易回推一年的基金申購交易
       e.g., 
        |    | cust_no | wm_prod_code   | txn_amt |\n
        |---:|:------------|:-----------|--------:|\n
        |  0 | rS/Gpmn+3hjYhNfqiAKFuw== | BB39           |     45000 |\n
        |  1 | G63O5M05MUQ3pJ17JRjeHQ== | 4010           |         1 |\n
        |  2 | wHoFRwxUDGqz1ovAhiAwdA== | 2531           |    226000 |
    Outputs: 
        - cust_txn_rating: 顧客最新一次交易回推一年的各基金申購交易占比 
       e.g., 
       |        | user_id  | item_id   |   rating |\n
       |-------:|:-------------------------|:----------|---------:|\n
       | 453324 | ++++2rDzc1I9amUGBhmxLA== | FCG5      | 0.252261 |\n
       | 352670 | ++++2rDzc1I9amUGBhmxLA== | 5950      | 0.267704 |\n
       | 273746 | ++++2rDzc1I9amUGBhmxLA== | 9117      | 0.232072 |\n
       |  40158 | ++++2rDzc1I9amUGBhmxLA== | 9809      | 0.247963 |\n
       | 191344 | +++A6VBLjJZGaIcDJLu8oA== | KK42      | 1        |
    TODO:
        - [ ] `cust_txn_rating = cust_txn_rating[['user_id','item_id','rating']].copy()`用del的方式處理 
        - [ ] 轉型別在get_cust_txn_amt做 
        - [ ] 欄位名稱轉換在get_cust_txn_amt做: 
            'cust_no':'user_id', 'wm_prod_code':'item_id'。
    """
    ### 轉型別
    cust_txn_amt['txn_amt'] = cust_txn_amt['txn_amt'].astype(float)
    ### 顧客交易總金額
    cust_txn_amt_sum = cust_txn_amt.groupby(['cust_no'])[['txn_amt']].sum().rename(columns={'txn_amt':'sum_txn_amt'})
    cust_txn_amt_sum = cust_txn_amt_sum.reset_index()
    ### 將每人對於所有基金交易總金額併回
    cust_txn_rating = cust_txn_amt.merge(cust_txn_amt_sum, on='cust_no', how='left')
    ### del
    del cust_txn_amt_sum
    gc.collect()
    ### 計算交易金額占比
    cust_txn_rating['rating'] = cust_txn_rating.apply(
        lambda x:(
            round(x['txn_amt']/x['sum_txn_amt'], 6
                 ) if x['sum_txn_amt']>0 else 0), 
        axis=1)
    cust_txn_rating = cust_txn_rating.rename(
        columns={'cust_no':'user_id', 'wm_prod_code':'item_id'})
    cust_txn_rating = cust_txn_rating[['user_id','item_id','rating']].copy()
    ### 型別
    cust_txn_rating['rating'] = cust_txn_rating['rating'].astype(float)
    del cust_txn_amt
    gc.collect()
    return cust_txn_rating