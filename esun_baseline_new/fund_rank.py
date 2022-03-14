import sys
import pytz
import pandas
import numpy
from sql_tools import SQLHelper
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from mlaas_tools.feature_tool import FeatureBase
from psycopg2 import sql, Error
from psycopg2.extras import execute_values

class FeatureETL(FeatureBase):
    """
    FeatureETL
    """
    def __init__(self, etl_dt):
        super().__init__(etl_dt)
        self.today = etl_dt

    def check(self):
        self.logger.info('check rawdata start')
        etl_3m_start = self.etl_period - relativedelta(months=3)
        rawdata_conn = self.dbset.get_rawdata_db_conn()
        feature_conn = self.dbset.get_feature_db_conn()
        # 近三個月
        try:
            self.logger.info('witwo103_hist: check rawdata start')
            ck_sql = """
                    SELECT count(distinct wm_prod_code) as cnt 
                    FROM mlaas_rawdata.witwo103_hist t1
                    WHERE exists 
                        (SELECT wm_prod_code FROM mlaas_rawdata.witwo106 as t2
                        WHERE replace(t1.wm_prod_code, ' ','')=replace(t2.wm_prod_code, ' ','')
                        and replace(t2.prod_detail_type_code, ' ','') in ('FNDF','FNDD')
                        and WM_Txn_Code='1'
                        and txn_dt >='{0}'
                        and txn_channel_code in ('B0','B1','B2')
                        and dta_src != 'A0'
                        and deduct_cnt in (0,1))
                    """.format(etl_3m_start)
            rawdata_cur = rawdata_conn.cursor()
            rawdata_cur.execute(ck_sql)
            raw_ck_result = rawdata_cur.fetchall()

            ftr_sql_check = """SELECT count(fund_id) FROM eb_ofsn_wm_fund.fund_rank"""
            feature_cur = feature_conn.cursor()
            feature_cur.execute(ftr_sql_check)
            ftr_ck_result = feature_cur.fetchall()
        except:
            self.logger.error('witwo103_hist: check rawdata error', exc_info=True)
            return False
        else:
            # 筆數>原筆數*0.5
            if (list(raw_ck_result[0])[0] >= list(ftr_ck_result[0])[0]*0.5):
                self.logger.info('witwo103_hist: check rawdata is normal')
                return True
            else:
                self.logger.info('witwo103_hist: check rawdata is abnormal')
                return False
        finally:
            rawdata_conn.close()
            feature_conn.close()
            self.logger.info('witwo103_hist: check rawdata end')

    def run(self):
        self.logger.info('fund_rank:ETL start')
        etl_3m_start = self.etl_period - relativedelta(months=3)
        try:
            raw_sql = """
                with max_etl_dt as (select max(etl_dt) as etl_dt  from mlaas_rawdata.witwo103_hist)

                select replace(t1.wm_prod_code, ' ', '') as fund_id,
                    row_number() over(order by count(distinct cust_id) desc) as rank,
                    (select etl_dt from max_etl_dt) as data_dt,
                    now()::date as etl_dt
                from mlaas_rawdata.witwo103_hist t1
                where exists (select wm_prod_code from mlaas_rawdata.witwo106 as t2
                        where replace(t1.wm_prod_code, ' ','')=replace(t2.wm_prod_code, ' ','')
                        and replace(t2.prod_detail_type_code, ' ','') in ('FNDF','FNDD')
                        and WM_Txn_Code='1'
                        and txn_dt >='{0}'
                        and txn_channel_code in ('B0','B1','B2')
                        and dta_src != 'A0'
                        and deduct_cnt in (0,1))
                group by replace(t1.wm_prod_code, ' ', '')
                order by count(distinct cust_id) desc
            """.format(etl_3m_start)
            
            sql_job = SQLHelper(etl_dt=self.today, table_name='fund_rank', sql=raw_sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=False, delete=True, update_time=True)
        except:
            self.logger.error('fund_rank:ETL error', exc_info=True)
            raise

        self.logger.info('fund_rank:ETL end')

if __name__ == '__main__':
    etl = FeatureETL(date.today())
    if etl.check():
        etl.run()
    else:
        etl.logger.info("Table didn't update because rawdata is abnormal")