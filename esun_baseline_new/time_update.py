import pandas as pd
from datetime import date, datetime
from mlaas_tools.feature_tool import FeatureBase


class FeatureETL(FeatureBase):
    def __init__(self, etl_dt):
        super().__init__(etl_dt)
        self.today = etl_dt

    def get_etldt(self, dt_sql=str)->str:
        """
        To get rawdata db conn.
        Parameters:
        dt_sql - sql statement
        Returns:
        return a string of rawdata etl_dt
        Raises:
        KeyError - raises an exception
        """
        try:
            self.logger.info('get feature conn.')
            with self.dbset.get_feature_db_conn() as feature_conn:
                feature_cur = feature_conn.cursor()
                feature_cur.execute(dt_sql)
                feature_etl_dt = feature_cur.fetchone()[0]
                feature_etl_dt_string = feature_etl_dt.strftime("%Y-%m-%d")
            return feature_etl_dt_string
        except:
            self.logger.error('get feature conn was failed.', exc_info=True)

    def compare_etldt(self, table_name=str):
        '''
        Check if etl_time of time_update_table is today.
        Parameters:
        table_name - table_name
        Returns:
        no return, just log
        Raises:
        KeyError - logging warning
        '''
        
        #取 time_update_table etl_time
        dt_sql = """select max(etl_time) from eb_ofsn_wm_fund.time_update_table where table_name = '{table}';""".format(table=table_name)
        feature_etl_dt_string = self.get_etldt(dt_sql=dt_sql)
        
        #確認 etl_time是否為今天日期
        if feature_etl_dt_string == str(self.today):
            self.logger.info('check {table} etl_dt: etl_dt is today'.format(table=table_name))
        else:
            self.logger.warning('check {table} etl_dt: etl_dt not today'.format(table=table_name))

    def run(self):
        '''check rawdata etl_dt'''
        #cm_customer
        self.logger.info('check rawdata etl_dt of cm_customer is going to start.')
        self.compare_etldt(table_name = 'cm_customer')
        #cust_similarity
        self.logger.info('check rawdata etl_dt of cust_similarity_var is going to start.')
        self.compare_etldt(table_name = 'cust_similarity_var')
        #fund_rank
        self.logger.info('check rawdata etl_dt of fund_rank is going to start.')
        self.compare_etldt(table_name = 'fund_rank')
        #fund_risk
        self.logger.info('check rawdata etl_dt of fund_risk is going to start.')
        self.compare_etldt(table_name = 'fund_risk')
        #new_cust_list
        self.logger.info('check rawdata etl_dt of new_cust_list is going to start.')
        self.compare_etldt(table_name = 'new_cust_list')
        #cust_fund_rating
        self.logger.info('check rawdata etl_dt of cust_fund_rating is going to start.')
        self.compare_etldt(table_name = 'cust_fund_rating')
        #old_cust_recommend
        self.logger.info('check rawdata etl_dt of old_cust_recommend is going to start.')
        self.compare_etldt(table_name = 'old_cust_recommend')
        
        self.logger.info('check time_update_table was finished.')

if __name__ == '__main__':
    etl = FeatureETL(date.today())
    etl.run()