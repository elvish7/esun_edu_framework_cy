import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from db_connection.utils import get_conn
from db_connection.utils import get_data_start_dt

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

def load_w103(today, rawdata_conn=None, span=1):
    txn_start_dt, txn_end_dt = get_data_start_dt(today, span), today  
    #assert datetime.strptime(
    #    txn_start_dt, '%Y-%m-%d') <= datetime.strptime(
    #    '2019-07-31', '%Y-%m-%d') 
    #assert datetime.strptime(
    #    txn_end_dt, '%Y-%m-%d') <= datetime.strptime(
    #    '2019-07-31', '%Y-%m-%d') 
    assert rawdata_conn 
    sql = """
    with 
        cte1 as (select cust_id as cust_no, 
                        replace(wm_prod_code, ' ', '') as wm_prod_code, 
                        txn_dt, 
                        (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as txn_amt,
                        dta_src,
                        deduct_cnt,
                        etl_dt
                from sinica.witwo103_hist 
                where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}'), 
        cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                from sinica.witwo106
                where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')) 
    select cte1.cust_no, cte1.wm_prod_code, cte1.txn_dt, cte1.txn_amt, cte1.dta_src, cte1.deduct_cnt, cte1.etl_dt
    from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code order by cust_no
    """.format(d_start=txn_start_dt, d_end=txn_end_dt)
    w103 = pd.read_sql(sql, rawdata_conn)
    return w103