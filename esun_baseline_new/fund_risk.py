import sys
import pytz
from datetime import date, datetime
from psycopg2 import sql
from sql_tools import SQLHelper
from psycopg2.extras import execute_values
from mlaas_tools.feature_tool import FeatureBase

class FeatureETL(FeatureBase):
    """create fund_risk table"""

    def __init__(self, etl_dt):
        '''init'''
        super().__init__(etl_dt)
        self.today = etl_dt

    def check(self):
        '''check rawdata of witwo106'''
        try:
            self.logger.info('witwo106:check rawdata start')
            rawdata_con = self.dbset.get_rawdata_db_conn()
            feature_con = self.dbset.get_feature_db_conn()
            
            rawdata_cur = rawdata_con.cursor()
            sql_check = """
                        SELECT count(wm_prod_code) 
                        FROM mlaas_rawdata.witwo106
                        WHERE prod_detail_type_code in ('FNDD','FNDF')
                        """
            rawdata_cur.execute(sql_check)
            raw_rownum = rawdata_cur.fetchall()

            feature_cur = feature_con.cursor()
            ftr_sql_check = """SELECT count(fund_id) FROM eb_ofsn_wm_fund.fund_risk"""
            feature_cur.execute(ftr_sql_check)
            ftr_rownum = feature_cur.fetchall()            
        except:
            self.logger.error('witwo106:check rawdata error', exc_info=True)
            return False
        else:
            # 筆數>原筆數*0.7
            if (list(raw_rownum[0])[0] >= list(ftr_rownum[0])[0]*0.7):
                self.logger.info('witwo106:check rawdata is normal')
                return True
            else:
                self.logger.error('witwo106:check rawdata is abnormal')
                return False
        finally:
            self.logger.info('witwo106:check rawdata end')
            rawdata_con.close()
            feature_con.close()

    def run(self):
        '''run'''
        try:
            self.logger.info('fund_risk:ETL start.')
            # 新增資料
            sqlstr = """
                    select distinct
                        replace(wm_prod_code, ' ', '')::character varying as fund_id,
                        (case when prod_risk_code = 'RR1' then 1
                              when prod_risk_code = 'RR2' then 1
                              when prod_risk_code = 'RR3' then 2
                              when prod_risk_code = 'RR4' then 3
                              when prod_risk_code = 'RR5' then 4
                         end)::numeric as fund_risk,
                         etl_dt::date as data_dt,
                         current_date as etl_dt
                    from witwo106
                    where prod_detail_type_code in ('FNDD','FNDF')
                """
            sql_job = SQLHelper(etl_dt=self.today, table_name='fund_risk', sql=sqlstr, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=False, delete=True, update_time=True)
        except:
            self.logger.error('fund_risk:ETL error', exc_info=True)
            raise

        self.logger.info('fund_risk:ETL end')
        
if __name__ == '__main__':
    etl = FeatureETL(date.today())
    if etl.check():
        etl.run()
    else:
        etl.logger.info("Table didn't update because rawdata is abnormal")