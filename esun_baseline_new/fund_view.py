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
    def __init__(self, etl_dt):
        super().__init__(etl_dt)
        self.today = etl_dt

    def get_conn(self, db_name):
        """
        get DB connection
        """
        try:
            if db_name =='rawdata':
                rawdata_conn = self.dbset.get_rawdata_db_conn()
                rawdata_conn.autocommit = False
                return rawdata_conn
            elif db_name =='feature':
                feature_conn = self.dbset.get_feature_db_conn()
                feature_conn.autocommit = False
                return feature_conn

        except (Exception, psycopg2.Error) as error:
            self.logger.error(error, exc_info=True)
            raise

    def check(self):
        self.logger.info('web_event:check rawdata start')
        etl_before_1d = self.etl_period - relativedelta(days=1)
        rawdata_conn = self.dbset.get_rawdata_db_conn()
        try:
            ck_sql = "select max(etl_dt) as data_dt from mlaas_rawdata.web_event"
            rawdata_cur = rawdata_conn.cursor()
            rawdata_cur.execute(ck_sql)
            ck_result = rawdata_cur.fetchall()
            if ck_result[0][0] >= etl_before_1d:
                self.logger.info('web_event:check rawdata is normal')
                return True
            else:
                self.logger.info('web_event:check rawdata is abnormal')
                return False
        except:
            self.logger.error('web_event:check rawdata error', exc_info=True)
            return False
        finally:
            self.logger.info('web_event:check rawdata end')
            rawdata_conn.close()
        
    def run(self):
        self.logger.info('fund_view:ETL start')
        try:
            raw_sql = """
                     with cte1 as (
                     SELECT
                         case when length(hits_eventinfo_eventlabel)=4 then hits_eventinfo_eventlabel else substr(hits_eventinfo_eventlabel,7,4) end AS wm_prod_code,
                         REPLACE(to_char(count(distinct customDimensions_value), '9,999'), ' ', '') AS cnt,
                         count(distinct customDimensions_value) as cust_cnt
                     FROM mlaas_rawdata.web_event t1
                     LEFT JOIN mlaas_rawdata.web_custno_mapping t2 ON t1.fullvisitorid=t2.fullvisitorid AND t1.clientid=t2.clientid
                     WHERE t1.visitdate >= '2020-03-27'
                     AND hits_eventinfo_eventcategory in ('基金e指選加入觀察','基金e指選立即申購')
                     AND customDimensions_value is not null
                     GROUP BY hits_eventinfo_eventlabel
                     ),
                     cte2 as (
                     SELECT max(etl_dt) AS data_dt FROM mlaas_rawdata.web_event
                     )
                     SELECT
                         wm_prod_code as fund_id,
                         concat('超過', cnt, '位顧客感興趣') as card_view,
                         (select data_dt from cte2) as data_dt,
                         now()::date as etl_dt
                     FROM cte1
                     WHERE cust_cnt >=100
                     """
            sql_job = SQLHelper(etl_dt=self.today, table_name='fund_view', sql=raw_sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=False, delete=True, update_time=True)
        except:
            self.logger.error('fund_view:ETL error', exc_info=True)
            raise
        else:
            self.logger.info('fund_view:ETL end')

if __name__ == '__main__':
    etl = FeatureETL(date.today())
    if etl.check():
        etl.run()
    else:
        etl.logger.info("Table didn't update because rawdata is abnormal")