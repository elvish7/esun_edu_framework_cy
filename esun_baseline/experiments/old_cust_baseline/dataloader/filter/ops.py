import pandas as pd 
from dateutil.relativedelta import relativedelta

def load_featured_funds(rawdata_conn = None):
    """精選基金"""
    sql = """
            select
                replace(wm_prod_code, ' ', '') as wm_prod_code
            from mlaas_rawdata.witwo106
            where prod_attr_5_ind='Y' and replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD');
            """
    funds = pd.read_sql(sql, rawdata_conn).wm_prod_code.tolist()
    return funds
def load_available_funds(rawdata_conn = None):
    """精選基金"""
    sql = """
            select
                replace(wm_prod_code, ' ', '') as wm_prod_code
            from mlaas_rawdata.witwo106
            where substring(channel_web_ind, 1, 1)='Y' and substring(channel_web_ind, 5, 1)='Y' and
                    substring(channel_mobile_ind, 1, 1)='Y' and substring(channel_mobile_ind, 5, 1)='Y' and
                    fee_type_code='A' and
                    prod_detail2_type_code in ('01','02') and replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD');
            """
    funds = pd.read_sql(sql, rawdata_conn).wm_prod_code.tolist()
    return funds

def load_w107(rawdata_conn = None): 
    # w107
    sql="""select cust_id as cust_no,
            case when cust_risk_code='01' then 1 
            when cust_risk_code='02' then 2
            when cust_risk_code='03' then 3
            when cust_risk_code='04' then 4 else 0 end as cust_risk from mlaas_rawdata.witwo107"""
    w107=pd.read_sql(sql, rawdata_conn)
    return w107 

def load_cust_txn_exclude(d_start, d_end, rawdata_conn=None):
    """
    說明：
        撈取顧客近3個月申購/贖回之基金標的
        「申購」係以單筆申購日期、定期定額首扣日期來看
        「贖回」則不論”部分贖回”或”全部贖回”均以交易日計算
    """

    txn_fund_sql = """
    with
        cte1 as (select cust_id as user_id,
                        replace(wm_prod_code, ' ', '') as item_id,
                        1 as exclude_ind
                from mlaas_rawdata.witwo103_hist
                where wm_txn_code in ('1','2')
                  and txn_dt>='{d_start}' and txn_dt<='{d_end}' 
                  and deduct_cnt in (0,1)
                  ),
        cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                from mlaas_rawdata.witwo106
                where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD'))
    select distinct cte1.user_id, cte1.item_id, cte1.exclude_ind
    from cte1 inner join cte2 on cte1.item_id=cte2.wm_prod_code
    """.format(d_start=d_start, d_end=d_end)
    cust_txn_exclude = pd.read_sql(txn_fund_sql, rawdata_conn)
    return cust_txn_exclude

def load_last_6m_new_cust(d_start, end_dt, rawdata_conn = None):
    assert rawdata_conn
    sql = """
        with
            cte1 as (
                select
                    cust_id as cust_no,
                    min(txn_dt) as min_txn_dt
                from witwo103_hist t1
                where exists (
                        select
                            wm_prod_code
                        from mlaas_rawdata.witwo106 as t2
                        where replace(t1.wm_prod_code, ' ','') = replace(t2.wm_prod_code, ' ','')
                        and t2.prod_detail_type_code in ('FNDF','FNDD')
                        and t1.WM_Txn_Code='1'
                        and t1.txn_channel_code in ('B0','B1','B2')
                        and t1.dta_src != 'A0'
                        and t1.deduct_cnt in (0,1))
                group by cust_id)
            select distinct cust_no
            from cte1
            where min_txn_dt <= '{d_end}' and min_txn_dt >= '{d_start}'
        """.format(d_end=end_dt, d_start=d_start)
    with rawdata_conn.cursor() as raw_cursor:
        raw_cursor.execute(sql)
        last_6m_new_cust = raw_cursor.fetchall()
        last_6m_new_cust = [''.join(i) for i in last_6m_new_cust]
    return last_6m_new_cust
    