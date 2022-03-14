import pandas as pd 
from dateutil.relativedelta import relativedelta

def load_w106(rawdata_conn = None):
    """產製基金產品檔"""
    sql = """
            select
                replace(wm_prod_code, ' ', '') as wm_prod_code,
                mkt_rbot_ctg_ic,
                replace(prod_detail_type_code, ' ','') as prod_detail_type_code,
                prod_ccy,
                prod_risk_code,
                (case when prod_attr_5_ind='Y' and
                    substring(channel_web_ind, 1, 1)='Y' and substring(channel_web_ind, 5, 1)='Y' and
                    substring(channel_mobile_ind, 1, 1)='Y' and substring(channel_mobile_ind, 5, 1)='Y' and
                    fee_type_code='A' and
                    prod_detail2_type_code in ('01','02')
                then 1 else 0 end) as can_rcmd_ind
            from mlaas_rawdata.witwo106
            where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD');
            """
    '''
    Note: 
    － can_rcmd_ind = 1　代表：
        －　精選：prod_attr_5_ind='Y'　
        －　可於數位通路購買：
            －　substring(channel_web_ind, 1, 1)='Y' and substring(channel_web_ind, 5, 1)='Y' 
            －　substring(channel_mobile_ind, 1, 1)='Y' and substring(channel_mobile_ind, 5, 1)='Y' 
        －　前收型基金：fee_type_code='A' 
        －　為境內外基金：prod_detail2_type_code in ('01','02')
    '''
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
    
    '''
    assert rawdata_conn 
    # w118
    sql = """
            select
                nav_date, 
                replace(product_code, ' ', '') as product_code, 
                purchase_val,
                redeem_val
            from mlaas_rawdata.witwo118
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
    '''
    assert rawdata_conn 
    sql = """
    with 
        cte1 as (select cust_id as cust_no, 
                        replace(wm_prod_code, ' ', '') as wm_prod_code, 
                        txn_dt, 
                        (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as txn_amt,
                        txn_channel_code, 
                        dta_src,
                        deduct_cnt,
                        etl_dt
                from mlaas_rawdata.witwo103_hist 
                where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}'), 
        cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                from mlaas_rawdata.witwo106
                where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')) 
    select cte1.cust_no, cte1.wm_prod_code, cte1.txn_dt, cte1.txn_amt, cte1.txn_channel_code, cte1.dta_src, cte1.deduct_cnt, cte1.etl_dt
    from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code
    """.format(d_start=txn_start_dt, d_end=txn_end_dt)
    w103 = pd.read_sql(sql, rawdata_conn)
    return w103

def load_cust_txn_amt(data_end_dt, rawdata_conn=None):
    raw_cursor = rawdata_conn.cursor()
    ### w103申購最近一次交易時間&前一年日期(with寫法)
    sql="""
        with 
            cte1 as (select cust_id as cust_no, 
                            replace(wm_prod_code, ' ', '') as wm_prod_code, 
                            txn_dt,
                            (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as wms_txn_amt_twd,
                            deduct_cnt

                    from mlaas_rawdata.witwo103_hist 
                    where wm_txn_code='1' and txn_dt <= '{d_end}'),

            cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                    from mlaas_rawdata.witwo106
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



    