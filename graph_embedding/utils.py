import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from collections import defaultdict
from db_connection.utils import get_conn
from db_connection.utils import get_data_start_dt

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
def cust_process(df):
    df[df['children_cnt']>=4] = 4
    # continuous value
    df['age'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], labels=False)
    df['cust_vintage'] = pd.cut(df['cust_vintage'], bins=[0, 100, 200, 300], labels=False)
    #df['cust_vintage'] = pd.qcut(df['cust_vintage'], 4, labels=False, duplicates='drop')
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return df

def load_cust(today, rawdata_conn=None, span=18):
    '''
    cte1:??????duplicates:???????????????cust_no??????????????????????????? ; 
    '''
    train_txn_start_dt, train_txn_end_dt = get_data_start_dt(today, span), today  
    after_1d_dt, after_1m_dt, after_7d_dt = get_data_dt(today, 1) # evaluate
    print('loading', train_txn_start_dt, after_1m_dt, 'user feature data.')
        
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
        """.format(d_start=train_txn_start_dt, d_end=after_1m_dt)
    cust_df = pd.read_sql(sql, rawdata_conn)
    cust_df = cust_process(cust_df)
    return cust_df

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
                               item_dict,threshold = 0,nrec_items = 5):
    
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    #known_items = list(pd.Series(interactions.loc[user_id,:] \
    #                             [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    
    #scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    pred = [k for k, v in item_dict.items() if v in return_score_list]
    
    return user_id, pred

def recommendation_all(model, intersections, user_li, user_dict, item_dict):

    predictions = defaultdict(list)

    for u in tqdm(user_li, total=len(user_li)):
        user_id, pred = top5_recommendation_user(model, intersections, u, user_dict, item_dict)
        predictions[user_id] = pred

    return predictions
