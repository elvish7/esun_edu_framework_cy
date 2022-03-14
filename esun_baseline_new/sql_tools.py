# coding: utf-8
import re
import pytz
import psycopg2
import sqlparse
from datetime import datetime
from ddlparse.ddlparse import DdlParse
from psycopg2.extras import execute_values
from mlaas_tools.feature_tool import FeatureBase


class DDLHelper():
    """Parser the given DDL file.

    Attributes:
        ddl: string
            The first statment of the given ddl file.
        indexes: list
            The index definitions from the given ddl file.
        table: string or list
        columns: list
        insert_sql: string
    """
    def __init__(self, table_name):
        self.__table_name = table_name
        self.__all_statement = list() 
        clean_statement = re.sub(' +', ' ', open('schemas.sql', mode='r').read()).replace('\n', '') 
        for statement in sqlparse.split(clean_statement):
            self.__all_statement.append(statement.replace(';', ''))
        self.__all_statement_sign = [x for x in self.__all_statement if 'eb_ofsn_wm_fund.' + self.__table_name + '(' in x]
        self.__ddl = self.__all_statement_sign[0]
        self.__table = self.tokens[-2][0]
        self.__schema = self.tokens[-2][0].split('.')[0]

    @property
    def ddl(self):
        return self.__ddl
    
    @property
    def indexes(self):
        if len(self.__all_statement) > 1:
            idx = self.__all_statement[1:]
        else:
            idx = None
        return idx
        
    @property
    def table(self):
        return self.__table

    @property
    def schema(self):
        return self.__schema
    
    @property
    def table_name(self):
        table_info = DdlParse().parse(self.__ddl)
        self.__table_name = table_info.name
        return self.__table_name
    
    @property
    def tokens(self):
        self.__tokens = list() 
        for token in sqlparse.parse(self.ddl)[0].tokens:
            if str(token.ttype) != 'Token.Text.Whitespace':
                self.__tokens.append((token.value, token.ttype))
        return self.__tokens
    
    @staticmethod
    def check_column_name_rule(name):
        temp = re.findall('^[a-zA-Z_][a-zA-Z_0-9]*$', name)
        if len(temp):
            result = True
        else:
            result = False
        return result
    
    def __get_columns_string(self):
        for token, ttype in self.tokens:
            if ttype is None and token[0]=='(':
                columns_string = token[1:-1].strip()
                break
        return columns_string
    
    @property
    def columns(self):
        self.__columns = list()
        sqlstring = self.__get_columns_string()
        for token in sqlstring.split(','):
            name = token.strip().split(' ')[0]
            if not self.check_column_name_rule(name):
                continue
            if name.upper() == 'PRIMARY':
                break
            self.__columns.append(name)
        return self.__columns
    
    @property
    def insert_sql(self):
        self.__nsert_sql = 'INSERT INTO {table} ({columns}) values %s'.format(table=self.table, columns=','.join(self.columns))
        return self.__nsert_sql

    
class SQLHelper(FeatureBase):
    def __init__(self, etl_dt, table_name, sql=None, source_db=None, target_db=None):
        super().__init__(etl_dt)
        self.__table_name = table_name
        self.__sql = sql
        self.__ddl_helper = DDLHelper(self.__table_name)
        self.__source_db = source_db
        self.__target_db = target_db
        self.__source_conn = None
        self.__target_conn = None
        
    def get_conn(self, db_name):
        """get DB connection"""
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
            
    def calculate_time(func):
        def wrap(self, *args, **kwds):
            stime = datetime.now()
            func(self, *args, **kwds)
            etime = datetime.now()
            dif_time = etime - stime
            self.logger.info('{table}:execution time of {func}() -> {time}'.format(func=func.__name__, time=dif_time, table=self.ddl_helper.table_name))
        return wrap

    @property
    def sql(self):
        return self.__sql
    
    @property
    def ddl_helper(self):
        return self.__ddl_helper
            
    @staticmethod
    @calculate_time
    def move_data_db2db(self, truncate_table, delete_table, select_sql, insert_sql, page_size=10000):
        """select data from src db and insert data target db."""
        self.logger.info('{table}:query data start'.format(table=self.ddl_helper.table_name))
        self.__source_conn = self.get_conn(self.__source_db)
        
        with self.__source_conn.cursor() as source_cur:
            try:
                source_cur.itersize = page_size
                source_cur.execute(select_sql)
                self.logger.info('{table}:query data end'.format(table=self.ddl_helper.table_name))
                self.__target_conn = self.get_conn(self.__target_db)
                ### delete table
                if truncate_table:
                    self.truncate_target_table()
                elif delete_table:
                    self.delete_target_table()
                ### insert table
                with self.__target_conn.cursor() as target_cur:
                    self.logger.info('{table}:insert data start'.format(table=self.ddl_helper.table_name))
                    while True:
                        records = source_cur.fetchmany(page_size)
                        if len(records) > 0:
                            execute_values(target_cur, insert_sql, records, page_size=page_size)
                        else:
                            break
            except Exception as error:
                if self.__target_conn is not None:
                    self.__target_conn.rollback()
                self.logger.error('{table}:insert data error'.format(table=self.ddl_helper.table_name))
                raise error
            else:
                self.__target_conn.commit()
        self.logger.info('{table}:insert data end'.format(table=self.ddl_helper.table_name))

    @staticmethod
    @calculate_time
    def move_data_df2db(self, truncate_table, delete_table, input_df, insert_sql, page_size=10000):
        """說明：將 dataframe insert DB"""
        try:
            self.__target_conn = self.get_conn(self.__target_db)
            ### delete table
            if truncate_table:
                self.truncate_target_table()
            elif delete_table:
                self.delete_target_table()
            ### insert table
            with self.__target_conn.cursor() as target_cur:
                self.logger.info('{table}:insert data start'.format(table=self.ddl_helper.table_name))
                df = input_df.values.tolist()
                execute_values(target_cur, insert_sql, df, page_size=page_size)
        except Exception as error:
            if self.__target_conn is not None:
                self.__target_conn.rollback()
            self.logger.error('{table}:insert data error'.format(table=self.ddl_helper.table_name))
            raise error
        else:
            self.__target_conn.commit()
        self.logger.info('{table}:insert data end'.format(table=self.ddl_helper.table_name))
        

    def to_target_db(self, df=None, page_size=10000, truncate=False, delete=False, update_time=False):
        """
        說明：
            更新 Feature DB Table 使用，會將所有執行步驟寫入log中。
            所有參數預設皆為 False
        參數: 
            1.df:寫入 DB 的 dataframe
            2.page_size：每次寫入筆數
            3.truncate：是否使用 truncat 方式刪除資料
            4.delete：是否使用 delete 方式刪除資料
            5.update_time：是否將 Table 更新時間寫入 eb_ofsn_wm_fund.time_update_table(table_name, etl_time)
        """        
        insert_sql = self.__ddl_helper
        try:            
            if isinstance(df, type(None)):
                self.move_data_db2db(self, 
                                     truncate_table=truncate,
                                     delete_table=delete,
                                     select_sql=self.__sql,
                                     insert_sql=self.__ddl_helper.insert_sql,
                                     page_size=page_size)
            else:
                self.move_data_df2db(self,
                                     truncate_table=truncate,
                                     delete_table=delete,
                                     input_df=df,
                                     insert_sql=self.__ddl_helper.insert_sql,
                                     page_size=page_size)
        except Exception as error:
            raise(error)
        else:
            if update_time:
                self.write_updatetime()
        finally:
            if self.__source_conn:
                self.__source_conn.close()
            if self.__target_conn:
                self.__target_conn.close()
        
    def truncate_target_table(self):
        try:
            self.logger.info('{table}:truncate table start'.format(table=self.ddl_helper.table_name))
            with self.__target_conn.cursor() as target_cur:
                truncate_sql = 'TRUNCATE TABLE {table}'.format(table=self.ddl_helper.table)
                target_cur.execute(truncate_sql)
        except:
            self.logger.error('{table}:truncate table error'.format(table=self.ddl_helper.table_name))
        else:
            self.logger.info('{table}:truncate table end'.format(table=self.ddl_helper.table_name))
        
    def delete_target_table(self):
        try:
            self.logger.info('{table}:delete table start'.format(table=self.ddl_helper.table_name))
            with self.__target_conn.cursor() as target_cur:
                delete_sql = 'DELETE FROM {table}'.format(table=self.ddl_helper.table)
                target_cur.execute(delete_sql)
        except:
            self.logger.error('{table}:delete table error'.format(table=self.ddl_helper.table_name))
        else:
            self.logger.info('{table}:delete table end'.format(table=self.ddl_helper.table_name))
        
    def write_updatetime(self):
        """
        說明：記錄每張 feature DB Table 更新時間，以利 time_update 排程檢核資料是否有更新
        """
        if self.__target_conn is None:
            raise ValueError("Should setting 'target_conn'")
        else:
            target_conn = self.__target_conn
        try:
            self.logger.info('{table}:insert time_update_table start'.format(table=self.ddl_helper.table_name))
            tw = pytz.timezone('Asia/Taipei')
            result = [('{table}'.format(table=self.ddl_helper.table_name), datetime.now(tw).replace(microsecond=0))]
            with target_conn.cursor() as target_cur:
                sql = """INSERT INTO eb_ofsn_wm_fund.time_update_table(table_name, etl_time) values %s"""
                execute_values(target_cur, sql, result, page_size=200000)
        except:
            self.logger.error('{table}:insert time_update_table error'.format(table=self.ddl_helper.table_name), exc_info=True)
            raise
        else:
            target_conn.commit()
            self.logger.info('{table}:insert time_update_table end'.format(table=self.ddl_helper.table_name))