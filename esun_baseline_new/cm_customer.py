import sys
import pytz
import pandas as pd
from datetime import date, datetime
from sql_tools import SQLHelper
from mlaas_tools.feature_tool import FeatureBase
from psycopg2.extras import execute_values


class FeatureETL(FeatureBase):
    '''繼承FeatureBase: FeatureBase'''
    def __init__(self, etl_dt):
        super().__init__(etl_dt)
        self.today = etl_dt

    def check(self):
        self.logger.info('cm_customer:check rawdata start')
        raw_conn = self.dbset.get_rawdata_db_conn()
        feature_conn = self.dbset.get_feature_db_conn()

        # 計算資料筆數
        raw_sql = """select count(cust_no) as cnt from mlaas_rawdata.cm_customer"""
        feature_sql = """select count(cust_no) as cnt from eb_ofsn_wm_fund.cm_customer"""
        try:
            raw_cur = raw_conn.cursor()
            raw_cur.execute(raw_sql)
            raw_result = raw_cur.fetchall()

            feature_cur = feature_conn.cursor()
            feature_cur.execute(feature_sql)
            feature_result = feature_cur.fetchall()
        except:
            self.logger.error('cm_customer:check rawdata error', exc_info=True)
            return False
        else:
            # raw 資料筆數 >= feature 資料筆數*0.9
            if list(raw_result[0])[0] >= list(feature_result[0])[0]*0.9:
                self.logger.info('cm_customer: {0} records'.format(format(list(raw_result[0])[0],',')))
                self.logger.info('cm_customer:check rawdata is normal')
                return True
            else:
                self.logger.error('cm_customer:check rawdata is abnormal')
                return False
        finally:
            self.logger.info('cm_customer:check rawdata end')
            raw_conn.close()
            feature_conn.close()

    def run(self):
        """cm_customer篩選個人戶，作為全新戶判斷"""
        self.logger.info('cm_customer:ETL start')
        try:
            raw_sql = """with etl_dt as (select max(etl_dt) as etl_dt  from mlaas_rawdata.cm_customer)
                         SELECT distinct cust_no,
                                (select etl_dt from etl_dt) as data_dt,
                                now()::date as etl_dt
                         FROM mlaas_rawdata.cm_customer 
                         WHERE biz_line_code='P' 
                         and age between 20 and 69
                    """
            sql_job = SQLHelper(etl_dt=self.today, table_name='cm_customer', sql=raw_sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=False, delete=True, update_time=True)
        except:
            self.logger.error('cm_customer:ETL error', exc_info=True)
            raise
        else:
            self.logger.info('cm_customer:ETL end')
        
if __name__ == '__main__':
    etl = FeatureETL(date.today())
    if etl.check():
        etl.run()
    else:
        etl.logger.info("Table didn't update because rawdata is abnormal")