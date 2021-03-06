import pandas as pd
import numpy as np
import datetime
import random
from tqdm import tqdm
from collections import defaultdict
from db_connection.utils import get_conn
from db_connection.utils import get_data_start_dt

def load_w103(today, rawdata_conn=None, span=18):
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
                        deduct_cnt,
                        etl_dt
                from sinica.witwo103_hist 
                where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}'), 
        cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                from sinica.witwo106
                where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')) 
    select cte1.cust_no, cte1.wm_prod_code, cte1.txn_dt, cte1.txn_amt, cte1.deduct_cnt, cte1.etl_dt
    from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code order by cust_no
    """.format(d_start=txn_start_dt, d_end=txn_end_dt)
    w103 = pd.read_sql(sql, rawdata_conn)
    return w103

def load_w103_eval(date, conn, d_start, d_end):
        sql = """
            with 
                cte1 as (select cust_id as cust_no, 
                                replace(wm_prod_code, ' ', '') as wm_prod_code, 
                                txn_dt, 
                                (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as txn_amt,
                                deduct_cnt,
                                etl_dt
                        from sinica.witwo103_hist 
                        where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}' and deduct_cnt <=1), 
                cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                        from sinica.witwo106
                        where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')) 
            select cte1.cust_no, cte1.wm_prod_code, cte1.txn_dt, cte1.txn_amt, cte1.deduct_cnt, cte1.etl_dt 
            from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code order by cust_no
            """.format(d_start=d_start, d_end=d_end)
        return pd.read_sql(sql, conn)  

def load_w106(rawdata_conn = None):
    """
    ?????????????????????: 
    
    Note: 
    
    1) ????????????DB???w106??????????????? prod_attr_5_ind (????????????) ??? prod_detail2_type_code     (??????/??????/?????????????????????)?????????can_rcmd_ind???????????????????????????????????? 
    
    * ??????????????? can_rcmd_ind = 1????????????
        ???????????????prod_attr_5_ind='Y' 
        ????????? ???????????????????????? ??? 
            (substring(channel_web_ind, 1, 1)='Y' and 
            substring(channel_web_ind, 5, 1)='Y' and 
            substring(channel_mobile_ind, 1, 1)='Y' and 
            substring(channel_mobile_ind, 5, 1)='Y')
        ????????? ?????????????????????fee_type_code='A' 
        ????????? ?????????????????????????????????prod_detail2_type_code in ('01','02') 
    
    * ???????????? can_rcmd_ind = 1????????????
        ?????????????????????????????? ??? 
            (substring(channel_web_ind, 1, 1)='Y' and 
            substring(channel_web_ind, 5, 1)='Y' and 
            substring(channel_mobile_ind, 1, 1)='Y' and 
            substring(channel_mobile_ind, 5, 1)='Y')
        ????????? ?????????????????????fee_type_code='A' 
    2) ??????DB???????????????????????????????????????????????????????????????(???500???)??? 
    """
    sql = """
            select
                replace(wm_prod_code, ' ', '') as wm_prod_code,
                mkt_rbot_ctg_ic,
                replace(prod_detail_type_code, ' ','') as prod_detail_type_code,
                prod_ccy,
                prod_risk_code,
                (case when substring(channel_web_ind, 1, 1)='Y' and substring(channel_web_ind, 5, 1)='Y' and
                    substring(channel_mobile_ind, 1, 1)='Y' and substring(channel_mobile_ind, 5, 1)='Y' and
                    fee_type_code='A' 
                then 1 else 0 end) as can_rcmd_ind
            from sinica.witwo106
            where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD');
            """
    w106 = pd.read_sql(sql, rawdata_conn)
    return w106

def load_w118(nav_start_dt, nav_end_dt, rawdata_conn=None):
    '''
    * ?????? w118 ????????????????????????
    
    Inputs: 
    - nav_start_dt, nav_end_dt: 
        The start and end timestamp of net average price. 
    - rawdata_conn: obtained from conns.get_rawdata_db_conn(). 
    
    Outputs: 
    - w118 ????????????????????????
    
    Notes: 
    - ??????????????????????????????'2019-07-31'???
        ??????nav_start_dt, nav_end_dt???????????????????????????
    
    '''
    assert datetime.strptime(
        nav_start_dt, '%Y-%m-%d') <= datetime.strptime(
        '2019-07-31', '%Y-%m-%d') 
    assert datetime.strptime(
        nav_end_dt, '%Y-%m-%d') <= datetime.strptime(
        '2019-07-31', '%Y-%m-%d') 
    assert rawdata_conn 
    # w118
    sql = """
            select
                nav_date, 
                replace(product_code, ' ', '') as product_code, 
                purchase_val,
                redeem_val
            from sinica.witwo118
            where nav_date>='{d_start}' and nav_date<='{d_end}';
            """.format(d_start=nav_start_dt, d_end=nav_end_dt)

    w118 = pd.read_sql(sql, rawdata_conn)
    return w118

def load_cust_dummy(today, rawdata_conn=None):
    '''
    cte1:???????????????????????? ; ????????????encoding???na???xxx0??????cust_vintage??????????????????
    cte2:??????duplicates:???????????????cust_no??????????????????????????? ; cust_vintage: normalization
    '''
    txn_start_dt, txn_end_dt = get_data_start_dt(today, 18), today  
    sql = """
        with
            cte0 as (select distinct cust_id as cust_no 
                     from sinica.witwo103_hist 
                    where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}'), 
            cte1 as(
                select distinct
                    cust_no,
                    etl_dt as data_dt,
                    age,
                    gender_code,
                    (case when gender_code = 'M' then 1 else 0 end)::numeric as gender_code1,
                    (case when gender_code = 'F' then 1 else 0 end)::numeric as gender_code2,
                    (case when gender_code is null then 1 else 0 end)::numeric as gender_code0,
                    coalesce((select avg(cust_vintage) from sinica.cm_customer_m), cust_vintage) as cust_vintage,
                    income_range_code,
                    (case when income_range_code = '1' then 1 else 0 end)::numeric as income_range_code1,
                    (case when income_range_code = '2' then 1 else 0 end)::numeric as income_range_code2,
                    (case when income_range_code = '3' then 1 else 0 end)::numeric as income_range_code3,
                    (case when income_range_code = '4' then 1 else 0 end)::numeric as income_range_code4,
                    (case when income_range_code is null then 1 else 0 end)::numeric as income_range_code0
                from sinica.cm_customer_m
                where (age between 20 and 69)
                and biz_line_code = 'P' 
                and cust_no in (select cust_no from cte0)
                )
            select cust_no,
                    data_dt,
                    age,
                    gender_code,
                    gender_code1,
                    gender_code2,
                    gender_code0,
                    cust_vintage,
                    income_range_code,
                    income_range_code1,
                    income_range_code2,
                    income_range_code3,
                    income_range_code4,
                    income_range_code0
            from cte1
        """.format(d_start=txn_start_dt, d_end=txn_end_dt)
    cust_pop = pd.read_sql(sql, rawdata_conn)
    return cust_pop

#cust_pop modified version
def load_cust_pop(today, rawdata_conn=None):
    '''
    cte1:???????????????????????? ; ????????????encoding???na???xxx0??????cust_vintage??????????????????
    cte2:??????duplicates:???????????????cust_no??????????????????????????? ; cust_vintage: normalization
    '''
    txn_start_dt, txn_end_dt = get_data_start_dt(today, 18), today  
    sql = """
        with
            cte0 as (select distinct cust_id as cust_no 
                     from sinica.witwo103_hist 
                    where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}'), 
            cte1 as(
                select distinct
                    cust_no,
                    etl_dt as data_dt,
                    age,
                    gender_code,
                    (case when gender_code = 'M' then 1 else 0 end)::numeric as gender_code1,
                    (case when gender_code = 'F' then 1 else 0 end)::numeric as gender_code2,
                    (case when gender_code is null then 1 else 0 end)::numeric as gender_code0,
                    coalesce((select avg(cust_vintage) from sinica.cm_customer_m), cust_vintage) as cust_vintage,
                    income_range_code,
                    (case when income_range_code = '1' then 1 else 0 end)::numeric as income_range_code1,
                    (case when income_range_code = '2' then 1 else 0 end)::numeric as income_range_code2,
                    (case when income_range_code = '3' then 1 else 0 end)::numeric as income_range_code3,
                    (case when income_range_code = '4' then 1 else 0 end)::numeric as income_range_code4,
                    (case when income_range_code is null then 1 else 0 end)::numeric as income_range_code0
                from sinica.cm_customer_m
                where (age between 20 and 69)
                and biz_line_code = 'P' 
                and cust_no in (select cust_no from cte0)
                ),
            cte2 as (
                select
                    cust_no,
                    data_dt,
                    age,
                    gender_code,
                    gender_code1,
                    gender_code2,
                    gender_code0,
                    ((cust_vintage
                    - (select min(cust_vintage) from sinica.cm_customer_m))
                    /((select max(cust_vintage) from sinica.cm_customer_m)
                    - (select min(cust_vintage) from sinica.cm_customer_m))) as cust_vintage,
                    income_range_code,
                    income_range_code1,
                    income_range_code2,
                    income_range_code3,
                    income_range_code4,
                    income_range_code0,
                    row_number() over (partition by cust_no order by age desc,
                                                                     cust_vintage desc,
                                                                     income_range_code1 desc,
                                                                     income_range_code2 desc,
                                                                     income_range_code3 desc,
                                                                     income_range_code4 desc,
                                                                     income_range_code0 desc,
                                                                     gender_code1 desc,
                                                                     gender_code2 desc,
                                                                     gender_code0 desc) as rank
                from cte1
                )
            select cust_no,
                    data_dt,
                    age,
                    gender_code,
                    gender_code1,
                    gender_code2,
                    gender_code0,
                    cust_vintage,
                    income_range_code,
                    income_range_code1,
                    income_range_code2,
                    income_range_code3,
                    income_range_code4,
                    income_range_code0
            from cte2
            where rank = 1
        """.format(d_start=txn_start_dt, d_end=txn_end_dt)
    cust_pop = pd.read_sql(sql, rawdata_conn)
    return cust_pop

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
