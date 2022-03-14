import sys
import pytz
import pandas as pd
import numpy as np
import scipy as sp
from datetime import date, datetime
from mlaas_tools.feature_tool import FeatureBase
from dateutil.relativedelta import relativedelta
from sklearn.metrics.pairwise import cosine_similarity
from psycopg2 import sql, Error
from psycopg2.extras import execute_values


class FeatureETL(FeatureBase):
    
    def __init__(self,etl_dt):
        super().__init__(etl_dt) 
        self.feature_conn=self.dbset.get_feature_db_conn()

    def check(self):
        self.logger.info('check rawdata start')
        ### 檢查w103
        self.logger.info('check rawdata (cust_similarity_var)')
        try:
            feature_conn = self.dbset.get_feature_db_conn()
            feature_cur = feature_conn.cursor()
            sql = """select count(1) as cnt from eb_ofsn_wm_fund.cust_similarity_var"""
            feature_cur.execute(sql)
            row_cnt = feature_cur.fetchone()
            row_cnt = int(row_cnt[0])
        except:
            self.logger.error('check rawdata: check rawdata cust_similarity_var error', exc_info=True)
            raise
        finally:
            feature_conn.close()

        ### 檢查資料源是否備齊
        if row_cnt >= 0:
            self.logger.info('rawdata is OK!!!')
            self.logger.info('check rawdata end')
            return True
        else:
            self.logger.error('rawdata is wrong!!!')
            self.logger.error('check rawdata end')
            return False
         
    def get_data(self):
        """取資料
            1.近半年第一次交易的舊戶相似度變數
            2.新戶總數
            3.資料更新時間
        """
        ### 近半年第一次交易的舊戶
        feature_conn = self.dbset.get_feature_db_conn()
        feature_cur = feature_conn.cursor()
        try:
            sql = """
                    select cust_no, age, gender_code0, gender_code1, gender_code2,
                           cust_vintage, income_range_code0, income_range_code1,
                           income_range_code2, income_range_code3, income_range_code4,
                           world_card_ind, pure_cc_ind, payroll_acct_ind, insure_ind,
                           gov_teacher_ind, esa_ind, pure_house_mtg_ln_ind, employer_ind,
                           curr_traveler_card_ind, cc_vip_ind, md_ind, my_ind, fd_ind,
                           fy_ind, cc_ind, ln_ind, aum_twd_new, aum_twd_mf1, aum_twd_mf2,
                           aum_twd_in, aum_twd_gd, aum_twd_etf, aum_twd_bo, aum_m1,
                           fund_m1, bond_m1, insurance_m1, gold_m1, etf_m1, aum_m3,
                           fund_m3, bond_m3, insurance_m3, gold_m3, etf_m3
                    from eb_ofsn_wm_fund.cust_similarity_var
                    where last_6m_new_ind = 1
                """
            cust_old = pd.read_sql(sql, feature_conn)
            # 近半年舊戶cust_no轉list
            cust_old_list = list(cust_old['cust_no'])
        except:
            self.logger.error('get rawdata cust_similarity_var: get cust_similarity_var table error', exc_info=True)
            raise
        else:
            self.logger.info('get rawdata cust_similarity_var table done')

        ### 新戶總數
        try:
            sql = """
                    select count(1)
                    from eb_ofsn_wm_fund.cust_similarity_var
                    where last_6m_new_ind = 0
                  """
            feature_cur.execute(sql)
            num_new_cust = feature_cur.fetchone()
            num_new_cust = int(num_new_cust[0])
        except:
            self.logger.error('get rawdata cust_similarity_var: get number of new cust error', exc_info=True)
            raise
        self.logger.info('get number of new cust done')

        ### 取 data_dt
        try:
            sql = """
                    select data_dt
                    from eb_ofsn_wm_fund.cust_similarity_var
                    limit 1
                """
            feature_cur.execute(sql)
            data_dt = feature_cur.fetchone()
            var_data_dt = data_dt[0]
        except:
            self.logger.error('get rawdata data_dt: get data_dt error', exc_info=True)
            raise
        finally:
            feature_conn.close()
        
        self.logger.info('get data_dt done')
        return cust_old, cust_old_list, num_new_cust, var_data_dt

    def fast_topK(self, sim, K):
        """取相似度TopK的顧客"""
        partial_sorting_sim = np.argpartition(-sim, K, 1)
        col_index = np.arange(sim.shape[0])[:, None]
        argsort_K = np.argsort(sim[col_index, partial_sorting_sim[:, 0:K]], 1)
        return (partial_sorting_sim[:, 0:K][col_index, argsort_K]).tolist()
    
    def truncate_table(self):
        try:
            self.logger.info('new_cust_list：truncate table start')
            sql = """truncate table eb_ofsn_wm_fund.new_cust_list"""
            with self.feature_conn.cursor() as feature_cur:
                feature_cur.execute(sql)
        except:
            self.logger.error('new_cust_list：truncate table error', exc_info=True)
            raise
        else:
            self.logger.info('new_cust_list：truncate table end')
    
    def loop_calculate(self, num_bins, num_new_rows, var_data_dt, cust_old_list, cust_old_matrix):
        try:
            self.logger.info('new_cust_list：calcalate similarity and insert data start')
            for i in range(0, (num_new_rows//num_bins+1), 1):
                num_offset = i*num_bins
                # get eb_ofsn_wm_fund.cust_similarity_var
                var_sql = """
                        select age, gender_code0, gender_code1, gender_code2,
                            cust_vintage, income_range_code0, income_range_code1,
                            income_range_code2, income_range_code3, income_range_code4,
                            world_card_ind, pure_cc_ind, payroll_acct_ind, insure_ind,
                            gov_teacher_ind, esa_ind, pure_house_mtg_ln_ind, employer_ind,
                            curr_traveler_card_ind, cc_vip_ind, md_ind, my_ind, fd_ind,
                            fy_ind, cc_ind, ln_ind, aum_twd_new, aum_twd_mf1, aum_twd_mf2,
                            aum_twd_in, aum_twd_gd, aum_twd_etf, aum_twd_bo, aum_m1,
                            fund_m1, bond_m1, insurance_m1, gold_m1, etf_m1, aum_m3,
                            fund_m3, bond_m3, insurance_m3, gold_m3, etf_m3
                        from eb_ofsn_wm_fund.cust_similarity_var
                        where last_6m_new_ind = 0 
                        order by cust_no limit {0} offset {1}
                    """.format(num_bins, num_offset)

                cust_sql = """
                            select cust_no
                            from eb_ofsn_wm_fund.cust_similarity_var
                            where last_6m_new_ind = 0 
                            order by cust_no limit {0} offset {1}
                        """.format(num_bins, num_offset)

                with self.feature_conn.cursor() as feature_cur:
                    feature_cur.execute(var_sql)
                    # 相似度矩陣
                    cust_new_matrix = feature_cur.fetchall()
                    feature_cur.execute(cust_sql)
                    new_cust_list = map(lambda x: x[0], feature_cur.fetchall())
                    new_cust_list = list(new_cust_list)

                # 計算相似度 (row為新戶，column為舊戶)
                sim = cosine_similarity(cust_new_matrix, cust_old_matrix)
                # top 10 index
                sim_top10 = self.fast_topK(sim, 10)

                # top 10 cust_no
                cust_top10_list = []
                capitalizer = lambda i:str(cust_old_list[i])

                for k in range(len(sim_top10)):
                    m1 = sim_top10[k]
                    # 去除順序
                    m1 = list(set(m1))
                    # one top 10 list
                    x = list(map(capitalizer, m1))
                    # add to whloe top10 list
                    cust_top10_list.append(x)

                # 時間存成一串list
                list_data_dt = [var_data_dt for i in range(num_bins)]
                list_etl_dt = [date.today() for i in range(num_bins)]

                # convert to tuple
                tp = tuple(zip(new_cust_list, cust_top10_list, list_data_dt, list_etl_dt))
                # insert DB
                insert_sql = "insert into eb_ofsn_wm_fund.new_cust_list (cust_no, cust_list, data_dt, etl_dt) values %s"
                with self.feature_conn.cursor() as feature_cur:
                    execute_values(feature_cur, insert_sql, tp, page_size=100000)
        except:
            self.logger.error('new_cust_list: insert table error', exc_info=True)
            raise
        else:
            self.feature_conn.commit()
            self.logger.info('new_cust_list：calcalate similarity and insert data end')
    
    def write_updatetime(self):
        """
        說明：記錄每張 feature DB Table 更新時間，以利 time_update 排程檢核資料是否有更新
        """
        try:
            self.logger.info('new_cust_list:insert time_update_table start')
            tw = pytz.timezone('Asia/Taipei')
            result = [('new_cust_list', datetime.now(tw).replace(microsecond=0))]
            with self.feature_conn.cursor() as target_cur:
                sql = """INSERT INTO eb_ofsn_wm_fund.time_update_table(table_name, etl_time) values %s"""
                execute_values(target_cur, sql, result, page_size=200000)
        except:
            self.logger.error('new_cust_list:insert time_update_table error', exc_info=True)
            raise
        else:
            self.feature_conn.commit()
            self.logger.info('new_cust_list:insert time_update_table end')
    
    def run(self):
        try:
            self.logger.info('new_cust_similarity:ETL start')
            ### select data
            cust_old, cust_old_list, num_new_rows, var_data_dt = self.get_data()
            ### 舊戶相似度矩陣
            cust_old_matrix = cust_old.iloc[:,1:].values
            ### truncate table
            self.truncate_table()
            ### 計算相似度
            self.loop_calculate(50000, num_new_rows, var_data_dt, cust_old_list, cust_old_matrix)
            ### 紀錄更新時間
            self.write_updatetime()
        except:
            self.logger.error('new_cust_similarity:ETL error')
            raise
        else:
            self.logger.info('new_cust_similarity:ETL end')
        finally:
            if self.feature_conn:
                self.feature_conn.close()

if __name__ == '__main__':
    etl = FeatureETL(date.today())
    if etl.check():
        etl.run()