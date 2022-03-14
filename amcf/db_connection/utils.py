import pandas as pd 
from dateutil.relativedelta import relativedelta
from mlaas_tools.config_build import config_set
from mlaas_tools.config_info import ConfigPass
from mlaas_tools.db_tool import DatabaseConnections
from db_connection.crv_tools import db_tool as edu_db_tool
# config_set()

def get_data_start_dt(etl_time_string, forward_months):
    """ 產製時間參數 """
    data_end_dt = pd.to_datetime(etl_time_string, format='%Y-%m-%d')
    data_start_dt = data_end_dt - relativedelta(
        months=forward_months) + relativedelta(days=1)
    data_start = data_start_dt.strftime('%Y-%m-%d') 
    return data_start
def get_conn(db_name):
    """
    get DB connection
    """
    
    try:
        configs = ConfigPass()._configsection
        conns = DatabaseConnections(configs)
        if db_name =='rawdata':
            rawdata_conn = conns.get_rawdata_db_conn()
            rawdata_conn.autocommit = False
            return rawdata_conn
        elif db_name =='feature':
            feature_conn = conns.get_feature_db_conn()
            feature_conn.autocommit = False
            return feature_conn
        elif db_name == 'edu':
            edu_conn = edu_db_tool.get_conn()
            edu_conn.autocommit = False 
            return edu_conn 
    except (Exception, psycopg2.Error) as error:
        raise
    
    
