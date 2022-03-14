import pandas as pd
import time
import ast
from collections import Counter
from datetime import date, datetime
from psycopg2.extras import execute_values
from mlaas_tools.feature_tool import FeatureBase


class FeatureETL(FeatureBase):
    def __init__(self, etl_dt):
        super().__init__(etl_dt)
        self.today = etl_dt
        self.feature_conn = None
        self.rawdata_conn = None

    def delete_table(self, truncate_table=None, delete_table=None):
        '''更新之前先truncate table'''
        if truncate_table:
            sql_truncate = f"""truncate table eb_ofsn_wm_fund.{truncate_table};"""
            self.feature_cur.execute(sql_truncate)
        if delete_table:
            sql_delete = f"""delete from eb_ofsn_wm_fund.{delete_table};"""
            self.feature_cur.execute(sql_delete)
        

    def get_cust_list(self):
        '''Get unique cust_list from eb_ofsn_wm_fund.new_cust_list'''
        sql = """
                select distinct cust_list
                from eb_ofsn_wm_fund.new_cust_list
            """
        self.feature_cur.execute(sql)
        cust_list_result = self.feature_cur.fetchall()
        cust_list = [list(cust_list_result[i])[0].replace('{','').replace('}','').split(",") for i in range(len(list(cust_list_result)))]
        # self.logger.info(f'Num of cust_list: {len(cust_list)}')
        return cust_list

    def quote(self, string):
        '''sql用的格式轉換'''
        return "('{}')".format(string)

    def get_fund_rating(self, cust_set):
        '''Get fund_rating of each cust_list'''
        sql = """
                select fund_rating
                from eb_ofsn_wm_fund.cust_fund_rating A
                where (exists (select * from (values {0}) vals(v) where A.cust_no = v))
            """.format(', '.join(map(self.quote, cust_set)))

        self.feature_cur.execute(sql)
        fund_rating_result = self.feature_cur.fetchall()
        rating_list = [ast.literal_eval(list(fund_rating_result[i])[0]) for i in range(len(fund_rating_result))] #10人
        # self.logger.info(f'Num of rating_list: {len(rating_list)}')
        return rating_list

    def sum_rating(self, rating_list):
        '''加總每支基金的Rating'''
        sum_dict = Counter(dict())
        for j in range(len(rating_list)):
            dict_ = Counter(rating_list[j])
            sum_dict = Counter(dict(sum_dict + dict_))
        sum_rating_dict = dict(sum_dict)
        return sum_rating_dict

    def convert2score(self, sum_rating_dict):
        '''
        把rating轉換成score
        score = (x-min)/(max-min)
        '''
        # min-max
        rating_max_key = max(sum_rating_dict, key=sum_rating_dict.get)
        rating_max = sum_rating_dict.get(rating_max_key)
        rating_min_key = min(sum_rating_dict, key=sum_rating_dict.get)
        rating_min = sum_rating_dict.get(rating_min_key)
        rating_base = rating_max - rating_min # 分母

        # score = (x-min)/(max-min)
        score_dict = {k: (v-rating_min)/rating_base for k, v in sum_rating_dict.items() if rating_base > 0}
        return score_dict

    def get_fund_risk(self):
        '''Get risk of fund from w106'''
        # 基金風險屬性
        rawdata_conn = self.dbset.get_rawdata_db_conn()
        raw_cursor = rawdata_conn.cursor()
        sql = """
            select
                wm_prod_code as fund_id,
                prod_risk_code
            from mlaas_rawdata.witwo106
            where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')
            and prod_attr_5_ind='Y'
            and substring(channel_web_ind, 1, 1)='Y'
            and substring(channel_web_ind, 5, 1)='Y'
            and substring(channel_mobile_ind, 1, 1)='Y'
            and substring(channel_mobile_ind, 5, 1)='Y'
            and fee_type_code='A'
            and prod_detail2_type_code in ('01','02')
            """
        raw_cursor.execute(sql)
        w106_risk = raw_cursor.fetchall()
        w106_risk = dict(w106_risk)
        raw_cursor.close()
        rawdata_conn.close()
        return w106_risk
    
    def fund_by_risk(self, w106_risk, sort_rating_dict, num_fund):
        '''各風險等級取5支基金'''
        risk_dict = {'RR2':[],'RR3':[],'RR4':[], 'RR5':[]}
        fund_dict = dict()
        if 'RR1' in w106_risk.values():
            risk_dict['RR1'] = []
        threshold = num_fund*len(risk_dict) #每個風險等級取5支基金

        c = 0
        for key, score in sort_rating_dict.items():
            len_value = sum([len(x) for x in risk_dict.values()])
            if len_value < threshold:
                R = w106_risk[key] #RR5
                if len(risk_dict[R]) < 5:
                    if key not in risk_dict[R]:
                        score_int = int(round(score*100, 0)) if score < 1 else 99
                        score_str = f'好感度 {str(score_int)}%' if score_int >= 70 else ''
                        risk_dict[R].append({key:score_str}) #score四捨五入至整數&加上中文，分數小於70留空字串
                        fund_dict[key] = score_str
                c+=1
        # self.logger.info(f'取風險等級的基金共取了幾次 iter:{c}')
        return fund_dict

    def insert_db(self, cust_set, fund_list):
        tuples = [(str(cust_set).replace('[','{').replace(']','}').replace("'","").replace(" ",""),
                    str(fund_list),
                    date.today())]

        sql = """INSERT INTO  
                eb_ofsn_wm_fund.cust_fund_list(cust_list,
                                                fund_list,
                                                etl_dt) values %s"""
        if not self.feature_conn:
            self.feature_conn = self.dbset.get_feature_db_conn()
            self.feature_cur = self.feature_conn.cursor()
        execute_values(self.feature_cur, sql, tuples, page_size=10000)
    
    def join_fund_list(self):
        '''把完成的fund_list 跟new_cust_list join在一起，inesrt到new_cust_recommend，delete and insert一起commit'''
        self.logger.info('join_fund_list is going to start')
        sql = """
                insert into 
                eb_ofsn_wm_fund.new_cust_recommend(
                    select A.cust_no,
                            B.fund_list,
                            A.data_dt,
                            current_date as etl_dt
                    from eb_ofsn_wm_fund.new_cust_list A
                    left join eb_ofsn_wm_fund.cust_fund_list B
                    on A.cust_list = B.cust_list);
                """
        self.delete_table(delete_table='new_cust_recommend')
        self.feature_cur.execute(sql)
        self.feature_conn.commit()
        self.logger.info('join_fund_list was finished')
    
    def run(self):
        try:
            self.logger.info('Lets go to start')
            self.feature_conn = self.dbset.get_feature_db_conn()
            self.feature_cur = self.feature_conn.cursor()
            # 先truncate cust_fund_list, new_cust_recommend，等到後面insert時一起commit
            self.delete_table(truncate_table='cust_fund_list')

            #取30萬cust_list
            cust_list = self.get_cust_list()

            # 取基金風險等級
            w106_risk = self.get_fund_risk()

            #把每組10人的cust_fund_rating抓出來
            for i in range(len(cust_list)):
                rating_list = self.get_fund_rating(cust_list[i])

                # rating 加總
                sum_rating_dict = self.sum_rating(rating_list)
                
                # 計算score: (x-min)/(max-min)
                score_dict = self.convert2score(sum_rating_dict)
                
                # 排序
                sort_rating_dict = dict(sorted(score_dict.items(), key=lambda x:x[1], reverse=True))
                
                # 各風險等級取5支基金
                fund_dict = self.fund_by_risk(w106_risk, sort_rating_dict, num_fund=5)
                
                #insert feature db
                self.insert_db(cust_set=cust_list[i], fund_list=fund_dict)

                if i % 10000 == 0:
                    self.feature_conn.commit()
                    self.logger.info(f'Insert data: no.{i+1}')
            #最後剩下的的資料，一起commit
            self.feature_conn.commit()
            # 最後 left join回new_cust_list->new_cust_recommend
            self.join_fund_list()
        except:
            self.logger.error('new_cust_recommend ETL was error',exc_info=True)
        finally:
            if self.feature_conn:
                self.feature_cur.close()
                self.feature_conn.close()
            if self.rawdata_conn:
                self.rawdata_cur.close()
                self.rawdata_conn.close()
            self.logger.info('new_cust_recommend ETL was finished')

if __name__ == '__main__':
    etl = FeatureETL(date.today())
    etl.run()