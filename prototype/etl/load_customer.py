import pandas as pd 
from dateteim import datetime

def load_products(start_dt, end_dt, rawdata_conn=None):
    pass


def load_customer(start_dt, end_dt, rawdata_conn=None):
    sql = """
    select
        nav_date,
        replace(product_code, ' ', '') as product_code,
        purchase_val,
        redeem_val
    """.format(d_start=nav_start_dt, d_end=nav_end_dt)
    w118 = pd.read_sql(sql, rawdata_conn)

    return w118

def load_transactions(rawdata_conn=None):
    assert datetime.strptime(txn_start_dt, '%Y-%m-%d') <= \
            datetime.strptime('2019-07-31', '%Y-%m-%d')
    assert datetime.strptime(txn_end_dt, '%Y-%m-%d') <= \
            datetime.strptime('2019-07-31', '%Y-%m-%d')
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
