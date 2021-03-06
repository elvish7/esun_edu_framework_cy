import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from collections import defaultdict
from db_connection.utils import get_conn
from db_connection.utils import get_data_start_dt

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

def w106_process(df):
    # discard categorization
    discard_condition = {'counterparty_code': 100, 'mkt_rbot_ctg_ic': 200, 'prod_ccy': 500}
    for col, n in discard_condition.items(): 
        df.loc[df[col].value_counts()[df[col]].values<n, col] = col+'_other'
    # convert int to categorical
    df['high_yield_bond_ind'] = df['high_yield_bond_ind'].map({'Y': 'high_yield', 'N': 'not_high_yield'})
    df['can_rcmd_ind'] = df['can_rcmd_ind'].map({1:'can_rcmd', 0: 'can_rcmd_N'})
    del df['invest_limited_code']
    return df

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
                (case when substring(channel_web_ind, 1, 1)='Y' and substring(channel_web_ind, 5, 1)='Y' and
                    substring(channel_mobile_ind, 1, 1)='Y' and substring(channel_mobile_ind, 5, 1)='Y' and
                    fee_type_code='A' 
                then 1 else 0 end) as can_rcmd_ind,
                high_yield_bond_ind,
                counterparty_code,
                invest_limited_code,
                invest_type,
                mkt_rbot_ctg_ic,
                prod_ccy,
                replace(prod_detail_type_code, ' ','') as prod_detail_type_code,
                prod_risk_code
            from sinica.witwo106
            where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD');
            """
    w106 = pd.read_sql(sql, rawdata_conn)
    w106 = w106_process(w106)
    return w106

# for evaluation data loading
def get_data_dt(etl_time_string, backward_months):
    today = pd.to_datetime(etl_time_string, format='%Y-%m-%d')
    data_start_dt = today + relativedelta(days=1)
    # data_1m_end_dt = today + relativedelta(months=backward_months)
    # Modified
    data_1m_end_dt = data_start_dt + relativedelta(months=backward_months) - relativedelta(days=1)
    data_7d_end_dt = today + relativedelta(days=7)
    data_start = data_start_dt.strftime('%Y-%m-%d')
    data_1m_end = data_1m_end_dt.strftime('%Y-%m-%d')
    data_7d_end = data_7d_end_dt.strftime('%Y-%m-%d')
    return data_start, data_1m_end, data_7d_end

def cust_process(df):
    df[df['children_cnt']>=4] = 4
    # continuous value
    df['age'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], labels=False)
    df['cust_vintage'] = pd.cut(df['cust_vintage'], bins=[0, 100, 200, 300], labels=False)
    #df['cust_vintage'] = pd.qcut(df['cust_vintage'], 4, labels=False, duplicates='drop')
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return df

def load_cust(today, rawdata_conn=None, span=18, mode='train'):
    '''
    cte1:??????duplicates:???????????????cust_no??????????????????????????? ; 
    '''
    if mode == 'train':
        txn_start_dt, txn_end_dt = get_data_start_dt(today, span), today  
    else: # evaluation
        after_1d_dt, after_1m_dt, after_7d_dt = get_data_dt(today, 1)
        txn_start_dt, txn_end_dt = after_1d_dt, after_1m_dt
    print('loading', txn_start_dt, txn_end_dt, 'data.')
        
    sql = """
        with
            cte0 as (select distinct cust_id as cust_no 
                     from sinica.witwo103_hist 
                    where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}'), 
            cte1 as(
                select
                    cust_no,
                    etl_dt,
                    age,
                    gender_code,
                    cust_vintage,
                    income_range_code,
                    risk_type_code,
                    children_cnt,
                    edu_code,
                    wm_club_class_code,
                    row_number() over (partition by cust_no order by etl_dt desc,
                                                                     age desc,
                                                                     cust_vintage desc,
                                                                     income_range_code asc) as rank
                from sinica.cm_customer_m
                where cust_no in (select cust_no from cte0)
                )
            select cust_no,
                    age,
                    gender_code,
                    cust_vintage,
                    income_range_code,
                    risk_type_code,
                    children_cnt,
                    edu_code,
                    wm_club_class_code
                    from cte1 
                    where rank = 1
        """.format(d_start=txn_start_dt, d_end=txn_end_dt)
    cust_df = pd.read_sql(sql, rawdata_conn)
    cust_df = cust_process(cust_df)
    return cust_df

