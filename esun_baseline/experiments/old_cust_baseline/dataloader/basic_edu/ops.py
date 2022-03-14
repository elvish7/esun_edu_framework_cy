import pandas as pd 
from dateutil.relativedelta import relativedelta
import pandas as pd 
from datetime import datetime  
def load_w106(rawdata_conn = None):
    """
    產製基金產品檔: 
    
    Note: 
    
    1) 因為產學DB的w106表目前缺少 prod_attr_5_ind (精選與否) 和 prod_detail2_type_code     (境內/境外/貨幣型基金標註)，因此can_rcmd_ind的產製和線上版本有差異。 
    
    * 線上版本的 can_rcmd_ind = 1　代表：
        －　精選：prod_attr_5_ind='Y' 
        －　且 可於數位通路購買 ： 
            (substring(channel_web_ind, 1, 1)='Y' and 
            substring(channel_web_ind, 5, 1)='Y' and 
            substring(channel_mobile_ind, 1, 1)='Y' and 
            substring(channel_mobile_ind, 5, 1)='Y')
        －　且 為前收型基金：fee_type_code='A' 
        －　且 為境內外非貨幣型基金：prod_detail2_type_code in ('01','02') 
    
    * 此版本的 can_rcmd_ind = 1　代表：
        －　可於數位通路購買 ： 
            (substring(channel_web_ind, 1, 1)='Y' and 
            substring(channel_web_ind, 5, 1)='Y' and 
            substring(channel_mobile_ind, 1, 1)='Y' and 
            substring(channel_mobile_ind, 5, 1)='Y')
        －　且 為前收型基金：fee_type_code='A' 
    2) 產學DB的資料為歷史資料，因此缺少許多新上架的基金(約500檔)。 
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
    * 得到 w118 基金淨值相關特徵
    
    Inputs: 
    - nav_start_dt, nav_end_dt: 
        The start and end timestamp of net average price. 
    - rawdata_conn: obtained from conns.get_rawdata_db_conn(). 
    
    Outputs: 
    - w118 基金淨值相關特徵
    
    Notes: 
    - 產學資料的最後一天是'2019-07-31'，
        因此nav_start_dt, nav_end_dt都不能超過這一天。
    
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


def load_w103(txn_start_dt, txn_end_dt, rawdata_conn=None):
    '''
    * 得到 w103 txn 基金購買紀錄
    
    Inputs: 
        - txn_start_dt, txn_end_dt: 
            The start and end timestamp of purchase history 
        - rawdata_conn: obtained from conns.get_rawdata_db_conn(). 
    Outputs: 
        - w103 txn 基金購買紀錄
        
    Notes: 
    和線上版本的差異是: 
    1) 產學DB的w103沒有 txn_channel_code (交易通路) 
    2) 產學資料的最後一天是'2019-07-31'，
        因此 txn_start_dt, txn_end_dt 都不能超過這一天。
    '''
    assert datetime.strptime(
        txn_start_dt, '%Y-%m-%d') <= datetime.strptime(
        '2019-07-31', '%Y-%m-%d') 
    assert datetime.strptime(
        txn_end_dt, '%Y-%m-%d') <= datetime.strptime(
        '2019-07-31', '%Y-%m-%d') 
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
    from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code
    """.format(d_start=txn_start_dt, d_end=txn_end_dt)
    w103 = pd.read_sql(sql, rawdata_conn)
    return w103

def load_cust_txn_amt(data_end_dt, rawdata_conn=None):
    '''
    * 一年中，各顧客各基金申購金額 (排除定期定額非第一次購買的紀錄)  
    
    Notes: 
    和線上版本的差異是: 
    2) 產學資料的最後一天是'2019-07-31'，
        因此 data_end_dt 不能超過這一天。
        
    '''
    assert datetime.strptime(
        data_end_dt, '%Y-%m-%d') <= datetime.strptime(
        '2019-07-31', '%Y-%m-%d') 
    assert rawdata_conn
    raw_cursor = rawdata_conn.cursor()
    ### w103申購最近一次交易時間&前一年日期(with寫法)
    sql="""
        with 
            cte1 as (select cust_id as cust_no, 
                            replace(wm_prod_code, ' ', '') as wm_prod_code, 
                            txn_dt,
                            (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as wms_txn_amt_twd,
                            deduct_cnt

                    from sinica.witwo103_hist 
                    where wm_txn_code='1' and txn_dt <= '{d_end}'),

            cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                    from sinica.witwo106
                    where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')),

            cte3 as (select cte1.cust_no, cte1.wm_prod_code, cte1.txn_dt, cte1.wms_txn_amt_twd, cte1.deduct_cnt
                    from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code),

            cte4 as(select cust_no, txn_dt 
                    from cte3
                    where deduct_cnt in (0,1)),

            cte5 as (select cust_no, 
                     max(txn_dt) as max_txn_dt,
                     (max(txn_dt) - INTERVAL '1 year')::date as start_txn_dt
                     from cte4
                     group by cust_no)

            select cte3.cust_no, cte3.wm_prod_code, sum(cte3.wms_txn_amt_twd) as txn_amt
            from cte3 left join cte5 on cte3.cust_no=cte5.cust_no
            where cte3.txn_dt >= cte5.start_txn_dt and cte3.txn_dt <= cte5.max_txn_dt
            group by cte3.cust_no, cte3.wm_prod_code
        """.format(d_end=data_end_dt)
    raw_cursor.execute(sql)
    #result = raw_cursor.fetchall()
    cust_txn_amt = pd.read_sql(sql, rawdata_conn)
    raw_cursor.close()
    return cust_txn_amt



    