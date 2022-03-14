import sys
import gc
import pytz
import functools
import pandas as pd
import numpy as np
import scipy as sp
from sql_tools import SQLHelper
from datetime import date, datetime
from mlaas_tools.feature_tool import FeatureBase
from dateutil.relativedelta import relativedelta
from sklearn.metrics import pairwise_distances
from psycopg2 import sql, Error
from psycopg2.extras import execute_values


class FeatureETL(FeatureBase):
    def __init__(self, etl_dt):
        super().__init__(etl_dt)
        self.today = str(etl_dt)
        self.txn_start_dt = self.get_data_start_dt(self.today, 18) # 一年半
        self.before_3m_dt = self.get_data_start_dt(self.today, 3) # 三個月(90天)
        self.before_1m_dt = self.get_data_start_dt(self.today, 1) # 一個月(30天)

    def log_recorder(func):
        """記錄log裝飾器"""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.logger.info("function:" + func.__name__ + " is going to start")
            try:
                result = func(self, *args, **kwargs)
            except:
                self.logger.error("function:" + func.__name__ + " was failed", exc_info=True)
                raise
            else:
                self.logger.info("function:" + func.__name__ + " was finished")
                return result
        return wrapper
    
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
        ### 檢查三資料源(w103, w106, w118)是否都備齊
        try:
            self.logger.info('witwo103_hist, witwo106, witwo118:check rawdata start')
            raw_conn = self.get_conn('rawdata')
            ### select w103
            try:
                self.logger.info('witwo103_hist:select rawdata start')
                self.logger.info('witwo103_hist:select rawdata from {0} to {1}'.format(self.txn_start_dt, self.today))
                raw_sql = """select count(distinct extract(month from txn_dt)) as cnt from mlaas_rawdata.witwo103_hist where txn_dt between '{0}' and '{1}'
                        """.format(self.txn_start_dt, self.today)
                check_w103 = pd.read_sql(raw_sql, raw_conn)
            except:
                self.logger.error('witwo103_hist: select rawdata error', exc_info=True)
                raw_conn.close()
                return False
            finally:
                self.logger.info('witwo103_hist:select rawdata end')
            ### select w106
            try:
                self.logger.info('witwo106:select rawdata start')
                raw_sql = """select count(distinct replace(wm_prod_code, ' ', '')) as cnt from mlaas_rawdata.witwo106"""
                check_w106=pd.read_sql(raw_sql, raw_conn)
            except:
                self.logger.error('witwo106:select rawdata error', exc_info=True)
                raw_conn.close()
                return False
            finally:
                self.logger.info('witwo106:select rawdata end')
            ### select w118
            try:
                self.logger.info('witwo118:select rawdata start')
                self.logger.info('witwo118:select rawdata from {0} to {1}'.format(self.before_1m_dt, self.today))
                raw_sql = """select count(distinct nav_date) as cnt from mlaas_rawdata.witwo118 where nav_date between '{0}' and '{1}'
                        """.format(self.before_1m_dt, self.today)
                check_w118 = pd.read_sql(raw_sql, raw_conn)
            except:
                self.logger.error('witwo118:select rawdata error', exc_info=True)
                raw_conn.close()
                return False
            finally:
                self.logger.info('witwo118:select rawdata end')
            ### 檢查是否符合條件
            if check_w103['cnt'].values >= 12 and check_w106['cnt'].values > 0 and check_w118['cnt'].values > 0:
                self.logger.info('witwo103_hist, witwo106, witwo118:check rawdata is normal')
                del check_w103, check_w106, check_w118
                gc.collect()
                return True
            else:
                if check_w103['cnt'].values < 12:
                    self.logger.info('witwo103_hist:check rawdata is abnormal')
                if check_w106['cnt'].values == 0:
                    self.logger.info('witwo106:check rawdata is abnormal')
                if check_w118['cnt'].values == 0:
                    self.logger.info('witwo118:check rawdata is abnormal')
                self.logger.info('witwo103_hist, witwo106, witwo118:check rawdata is abnormal')
                del check_w103, check_w106, check_w118
                gc.collect()
                return False
        except:
            self.logger.error('witwo103_hist, witwo106, witwo118:check rawdata error', exc_info=True)
            return False
        finally:
            self.logger.info('witwo103_hist, witwo106, witwo118:check rawdata end')
            if raw_conn:
                raw_conn.close()
        
    def get_data_start_dt(self, etl_time_string, forward_months):
        """ 產製時間參數 """
        data_end_dt = pd.to_datetime(etl_time_string, format='%Y-%m-%d')
        data_start_dt = data_end_dt - relativedelta(months=forward_months) + relativedelta(days=1)
        data_start = data_start_dt.strftime('%Y-%m-%d')
        return data_start
    
    def get_w103(self):
        """產製近OO月基金申購交易"""
        raw_conn = self.get_conn('rawdata')
        try:
            sql = """
                with
                    cte1 as (select cust_id as cust_no,
                                    replace(wm_prod_code, ' ', '') as wm_prod_code,
                                    txn_dt,
                                    (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as txn_amt,
                                    txn_channel_code,
                                    dta_src,
                                    etl_dt
                            from mlaas_rawdata.witwo103_hist
                            where wm_txn_code='1'and txn_dt>='{d_start}'),
                    cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                            from mlaas_rawdata.witwo106
                            where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD'))
                select cte1.cust_no, cte1.wm_prod_code, cte1.txn_dt, cte1.txn_amt, cte1.txn_channel_code, cte1.dta_src, cte1.etl_dt
                from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code
                """.format(d_start=self.txn_start_dt)
            w103 = pd.read_sql(sql, raw_conn)
        except:
            self.logger.error('get rawdata w103: get witwo103_hist(fund similarity) table error', exc_info=True)
            raise
        finally:
            raw_conn.close()

        self.logger.info('get rawdata w103(fund similarity) done')
        return w103
    
    def get_w106(self):
        """產製基金產品檔"""
        raw_conn = self.get_conn('rawdata')
        try:
            sql = """
                    select
                        replace(wm_prod_code, ' ', '') as wm_prod_code,
                        mkt_rbot_ctg_ic,
                        replace(prod_detail_type_code, ' ','') as prod_detail_type_code,
                        prod_ccy,
                        prod_risk_code,
                        (case when prod_attr_5_ind='Y' and
                            substring(channel_web_ind, 1, 1)='Y' and substring(channel_web_ind, 5, 1)='Y' and
                            substring(channel_mobile_ind, 1, 1)='Y' and substring(channel_mobile_ind, 5, 1)='Y' and
                            fee_type_code='A' and
                            prod_detail2_type_code in ('01','02')
                        then 1 else 0 end) as can_rcmd_ind
                    from mlaas_rawdata.witwo106
                    where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD');
                    """
            w106 = pd.read_sql(sql, raw_conn)
        except:
            self.logger.error('get rawdata w106: get witwo106 table error', exc_info=True)
            raise
        finally:
            raw_conn.close()
        self.logger.info('get rawdata w106 done')
        return w106
    
    def get_w118(self):
        """產製近三個月基金淨值"""
        raw_conn = self.get_conn('rawdata')
        try:
            sql = """
                select
                    replace(product_code, ' ', '') as product_code,
                    purchase_val
                from mlaas_rawdata.witwo118
                where nav_date>='{d_start}';
                """.format(d_start=self.before_1m_dt)
            w118 = pd.read_sql(sql, raw_conn)
        except:
            self.logger.error('get rawdata w118: get witwo118 table error', exc_info=True)
            raise
        finally:
            raw_conn.close()
        self.logger.info('get rawdata w118 done')
        return w118
        
    def get_cust_txn_amt(self):
        """產製顧客最新一次交易回推一年的基金申購交易"""
        try:
            raw_conn = self.get_conn('rawdata')
            sql="""
                with
                    cte1 as (select cust_id as cust_no,
                                    replace(wm_prod_code, ' ', '') as wm_prod_code,
                                    txn_dt,
                                    (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as wms_txn_amt_twd,
                                    deduct_cnt
                            from mlaas_rawdata.witwo103_hist
                            where wm_txn_code='1'),
                    cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                            from mlaas_rawdata.witwo106
                            where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')),
                    cte3 as (select cte1.cust_no, cte1.wm_prod_code, cte1.txn_dt, cte1.wms_txn_amt_twd, cte1.deduct_cnt
                            from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code),
                    cte4 as(select cust_no, txn_dt
                            from cte3
                            where deduct_cnt in (0,1)),
                    cte5 as (select cust_no,
                             max(txn_dt) as max_txn_dt,
                             (max(txn_dt) - INTERVAL '1 year')::date as start_txn_dt
                             from cte4
                             group by cust_no)
                    select cte3.cust_no, cte3.wm_prod_code, sum(cte3.wms_txn_amt_twd) as txn_amt
                    from cte3 left join cte5 on cte3.cust_no=cte5.cust_no
                    where cte3.txn_dt >= cte5.start_txn_dt and cte3.txn_dt <= cte5.max_txn_dt
                    group by cte3.cust_no, cte3.wm_prod_code
            """
            cust_txn_amt = pd.read_sql(sql, raw_conn)
        except:
            self.logger.error('get rawdata w103: get witwo103_hist(cust_txn_amt) table error', exc_info=True)
            raise
        finally:
            raw_conn.close()
        self.logger.info('get rawdata w103(cust_txn_amt) done')
        return cust_txn_amt
        
    def get_last_6m_new_cust(self):
        """產製近半年新戶"""
        raw_conn = self.get_conn('rawdata')
        try:
            sql = """
                with
                    cte1 as (
                        select
                            cust_id as cust_no,
                            min(txn_dt) as min_txn_dt,
                            current_date - interval '6 month' as per_mon6
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
                    where min_txn_dt <= current_date and min_txn_dt >= per_mon6
                """
            with raw_conn.cursor() as raw_cursor:
                raw_cursor.execute(sql)
                last_6m_new_cust = raw_cursor.fetchall()
                last_6m_new_cust = [''.join(i) for i in last_6m_new_cust]
        except:
            self.logger.error('get rawdata w103: get witwo103_hist(last_6m_new_cust) list error', exc_info=True)
            raise
        finally:
            raw_conn.close()
        self.logger.info('get rawdata w103(last_6m_new_cust) done')

        self.logger.info('load DB done')
        return last_6m_new_cust
    
    def get_cust_txn_3m(self):
        """
        說明：
            撈取顧客近3個月申購/贖回之基金標的
            「申購」係以單筆申購日期、定期定額首扣日期來看
            「贖回」則不論”部分贖回”或”全部贖回”均以交易日計算
        """
        raw_conn = self.get_conn('rawdata')
        try:
            txn_fund_sql = """
            with
                cte1 as (select cust_id as user_id,
                                replace(wm_prod_code, ' ', '') as item_id,
                                1 as exclude_ind
                        from mlaas_rawdata.witwo103_hist
                        where wm_txn_code in ('1','2')
                          and txn_dt>='{d_start}'
                          and deduct_cnt in (0,1)
                          ),
                cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                        from mlaas_rawdata.witwo106
                        where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD'))
            select distinct cte1.user_id, cte1.item_id, cte1.exclude_ind
            from cte1 inner join cte2 on cte1.item_id=cte2.wm_prod_code
            """.format(d_start=self.before_3m_dt)
            cust_txn_exclude = pd.read_sql(txn_fund_sql, raw_conn)
        except:
            self.logger.error('get rawdata w103: get witwo103_hist(cust_txn_exclude) table error', exc_info=True)
            raise
        finally:
            raw_conn.close()

        self.logger.info('get rawdata w103(cust_txn_exclude) done')
        return cust_txn_exclude
        
                       
    
    def nav_ind_quar(self, w118_describe, col_name):
        """淨值變化註記function"""
        val_max = w118_describe[col_name].max()
        val_min = w118_describe[col_name].min()
        quar = np.percentile(w118_describe[col_name], [25,50,75])
        q1 = quar[0]
        q2 = quar[1]
        q3 = quar[2]
        conditions = [
            (val_min <= w118_describe[col_name]) & (w118_describe[col_name] < q1),
            (q1 <= w118_describe[col_name]) & (w118_describe[col_name] < q2),
            (q2 <= w118_describe[col_name]) & (w118_describe[col_name] < q3),
            (q3 <= w118_describe[col_name]) & (w118_describe[col_name] <= val_max)]
        choices = [col_name+'_flag1', col_name+'_flag2', col_name+'_flag3',col_name+'_flag4']
        return np.select(conditions, choices, default = col_name+'_flag0')
    
    @log_recorder
    def item_similarity(self, w103, w106, w118):
        """產製基金相似矩陣"""

        ### 註記顧客有無購買註記(一年半內是否購買過該基金)
        txn = w103[['cust_no','wm_prod_code']].copy()
        txn = txn.rename(columns={'cust_no':'user_id', 'wm_prod_code':'item_id'})
        txn = txn.drop_duplicates(subset=['user_id','item_id'], keep='last')
        txn =  txn.reset_index(drop=True)
        txn['rating'] = 1
        txn_matrix = txn.groupby(['user_id','item_id'])['rating'].sum().unstack('user_id')
        txn_matrix = txn_matrix.fillna(0)
        txn_matrix = txn_matrix.astype(int)
        txn_matrix = txn_matrix.rename(columns=str).reset_index()
        self.logger.info('get user txn ind done')
        del txn
        gc.collect()
        
        ### 基金基本屬性(feature1)
        col_id = ['wm_prod_code']
        col_cat = ['mkt_rbot_ctg_ic','prod_detail_type_code','prod_ccy','prod_risk_code']
        df_id = w106[col_id].copy()
        df_cat = w106[col_cat].copy()
        df_cat['mkt_rbot_ctg_ic'] = df_cat['mkt_rbot_ctg_ic'].replace(to_replace = '', value = 'F0000')
        df_cat_dummy = pd.get_dummies(df_cat)
        df_cat_dummy = df_cat_dummy.astype(int)
        df_cat_dummy.columns = map(str.lower, df_cat_dummy.columns)
        item_attr = pd.concat([df_id, df_cat_dummy], axis=1)
        item_attr = item_attr.rename(columns={'wm_prod_code':'item_id'})
        self.logger.info('get feature(fund attr) done')
        del col_id, col_cat, df_id, df_cat, df_cat_dummy
        gc.collect()
        
        ### 數位通路熱銷排名(feature2)
        txn_dig = w103[(w103['txn_channel_code'].isin(['B0','B1','B2'])) & (w103['dta_src']!='A0')].copy()
        txn_dig['txn_dt'] = pd.to_datetime(txn_dig['txn_dt'])
        txn_dig = txn_dig[(txn_dig['txn_dt']>=self.before_3m_dt) & (txn_dig['txn_dt']<=self.today)].copy()
        txn_dig = txn_dig.reset_index(drop=True)
        top = txn_dig.groupby(['wm_prod_code'])[['txn_amt']].sum().sort_values(by=['txn_amt'], ascending=False)
        top = top.reset_index()
        top = top[:50]
        top['no'] = 1
        top['rank'] = top['no'].cumsum()
        top = top[['wm_prod_code','rank']].copy()
        top_dummy =  pd.get_dummies(top['rank'])
        top_dummy = top_dummy.astype(int)
        item_top = pd.concat([top, top_dummy], axis=1)
        item_top = item_top[[x for x in item_top.columns if x not in ['rank']]]
        item_top = item_top.rename(columns={'wm_prod_code':'item_id'})
        item_top.columns = item_top.columns.astype(str)
        self.logger.info('get feature(top sales) done')
        del w103, top, top_dummy
        gc.collect()

        ### 淨值變化註記(feature3)
        w118['product_code'] = w118['product_code'].astype(str)
        w118['purchase_val'] = w118['purchase_val'].astype(float)
        w118_describe = w118.groupby(['product_code'])['purchase_val'].describe()
        w118_describe = w118_describe.reset_index()
        w118_describe['std'] = w118_describe['std'].fillna(0)
        w118_describe.columns = w118_describe.columns.astype(str)
        w118_describe = w118_describe.rename(columns={'product_code':'item_id'})
        w118_ind = w118_describe[['item_id']].copy()
        w118_ind = w118_ind.reset_index(drop=True)
        features_nav = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for feature in features_nav:
            w118_ind[feature] = self.nav_ind_quar(w118_describe, feature)
        col_nav = features_nav
        df_id = w118_ind[['item_id']].copy()
        df_nav = w118_ind[col_nav].copy()
        df_nav_dummy = pd.get_dummies(df_nav)
        item_nav = pd.concat([df_id, df_nav_dummy], axis=1)
        self.logger.info('get feature(nav) done')
        del w118, w118_describe, w118_ind, features_nav, col_nav, df_id, df_nav, df_nav_dummy
        gc.collect()

        ### 合併三features至交易註記
        item_feature = txn_matrix.merge(item_attr, on='item_id', how='left')
        item_feature = item_feature.merge(item_top, on='item_id', how='left')
        item_feature = item_feature.merge(item_nav, on='item_id', how='left')
        item_feature = item_feature.fillna(0)
        self.logger.info('get all feature done')
        del txn_matrix, item_attr, item_top, item_nav
        gc.collect()

        ### 計算基金相似度
        item_feature_val = item_feature.iloc[:,1:].values
        dist = 1-pairwise_distances(item_feature_val, metric='cosine')
        item_matrix = pd.DataFrame(dist, columns=item_feature.item_id)
        item_matrix['item_id'] = item_feature.item_id
        self.logger.info('item-item similarity matrix done')
        del item_feature, item_feature_val, dist
        gc.collect()
        return item_matrix, txn_dig
        
    @log_recorder   
    def cust_item_recmd(self, cust_txn_amt, item_matrix, txn_dig, w106, last_6m_new_cust, cust_txn_exclude):
        """產製舊戶推薦結果及近半年第一買基金顧客的推薦結果"""
        ### 交易量占比
        cust_txn_amt['txn_amt'] = cust_txn_amt['txn_amt'].astype(float)
        cust_txn_amt_sum = cust_txn_amt.groupby(['cust_no'])[['txn_amt']].sum().rename(columns={'txn_amt':'sum_txn_amt'})
        cust_txn_amt_sum = cust_txn_amt_sum.reset_index()
        cust_txn_rating = cust_txn_amt.merge(cust_txn_amt_sum, on='cust_no', how='left')
        cust_txn_rating['rating'] = cust_txn_rating.apply(lambda x:(round(x['txn_amt']/x['sum_txn_amt'], 6) if x['sum_txn_amt']>0 else 0), axis=1)
        cust_txn_rating = cust_txn_rating.rename(columns={'cust_no':'user_id', 'wm_prod_code':'item_id'})
        cust_txn_rating = cust_txn_rating[['user_id','item_id','rating']].copy()
        cust_txn_rating['rating'] = cust_txn_rating['rating'].astype(float)
        self.logger.info('get txn rating done')
        del cust_txn_amt, cust_txn_amt_sum
        gc.collect()

        ### 挑出可推薦的基金
        item_can_recmd_list = w106[w106['can_rcmd_ind']==1].wm_prod_code.tolist()
        item_matrix = item_matrix.loc[:, item_matrix.columns.isin(['item_id']+item_can_recmd_list)]
        del item_can_recmd_list
        gc.collect()

        ### 合併再計算加權的基金相似度vector
        cust_item_rating = pd.merge(cust_txn_rating, item_matrix, on = ['item_id'], how='left')
        item_list = [x for x in item_matrix.columns if x not in ['item_id']]
        for item in item_list:
            cust_item_rating[item] = (cust_item_rating['rating'].values) *(cust_item_rating[item].values)
        cust_item_rating = cust_item_rating.groupby(['user_id'])[item_list].sum()
        cust_item_rating = cust_item_rating.reset_index()
        
        ### return 近半年新戶 
        cust_fund_rating_6m = cust_item_rating[cust_item_rating['user_id'].isin(last_6m_new_cust)].copy()
        
        ### 舊戶rating轉置
        cust_item_rank = cust_item_rating.set_index(['user_id'])
        del cust_item_rating
        gc.collect()

        cust_item_rank = cust_item_rank.stack()
        cust_item_rank = cust_item_rank.reset_index(name='rating')
        cust_item_rank = cust_item_rank.rename(columns={'level_1':'item_id'})

        ### 近三個月熱銷
        txn_dig_ind = txn_dig[['cust_no','wm_prod_code']].copy()
        txn_dig_ind = txn_dig_ind.drop_duplicates(subset=['cust_no','wm_prod_code'], keep='last')
        txn_dig_ind =  txn_dig_ind.reset_index(drop=True)
        dig_cnt = txn_dig_ind.groupby(['wm_prod_code'])[['cust_no']].count().rename(columns={'cust_no':'txn_cust_cnt'})
        dig_cnt = dig_cnt.sort_values(by=['txn_cust_cnt'], ascending=False)
        dig_cnt = dig_cnt.reset_index()
        dig_cnt['no'] = 1
        dig_cnt['item_hot_rank'] = dig_cnt['no'].cumsum()
        dig_cnt = dig_cnt[['wm_prod_code','item_hot_rank']].copy()
        dig_cnt = dig_cnt.rename(columns={'wm_prod_code':'item_id'})
        del txn_dig, txn_dig_ind
        gc.collect()

        ### 合併基金rating及熱銷，並排序
        cust_item_rank = cust_item_rank.merge(dig_cnt, how='left', on=['item_id'])
        cust_item_rank = cust_item_rank.merge(cust_txn_exclude, how='left', on=['user_id','item_id'])
        cust_item_rank['item_hot_rank'] = cust_item_rank['item_hot_rank'].fillna(9999)
        cust_item_rank['exclude_ind'] = cust_item_rank['exclude_ind'].fillna(0)
        ### exclude_ind 小到大; rating 大到小; item_hot_rank 小到大 
        cust_item_rank = cust_item_rank.sort_values(['user_id','exclude_ind','rating','item_hot_rank'], ascending=[1,1,0,1])
        cust_item_rank = cust_item_rank.reset_index(drop=True)
        cust_item_rank['cnt'] = 1
        cust_item_rank['item_comb_rank'] = cust_item_rank.groupby(['user_id'])['cnt'].cumsum()
        cust_item_rank = cust_item_rank.drop(['item_hot_rank','cnt','exclude_ind'], axis=1) # rating 先不刪以計算 score
        self.logger.info('cust ranking done')
        del dig_cnt
        gc.collect()

        ### 加入基金風險屬性，每個風險留下top5，輸出推薦結果
        w106_risk = w106[['wm_prod_code','prod_risk_code']].copy().rename(columns={'wm_prod_code':'item_id'})
        cust_item_rank = cust_item_rank.merge(w106_risk, how='left', on=['item_id'])
        cust_item_recmd = cust_item_rank.groupby(['user_id','prod_risk_code']).head(5)
        cust_item_recmd = cust_item_recmd.reset_index(drop=True)
        cust_item_recmd['cnt'] = 1
        cust_item_recmd['rank'] = cust_item_recmd.groupby(['user_id'])['cnt'].cumsum()
        cust_item_recmd = cust_item_recmd.drop(['item_comb_rank','prod_risk_code','cnt'], axis=1)
        cust_item_recmd = cust_item_recmd.rename(columns={'user_id':'cust_no','item_id':'fund_id'})
        del w106_risk, w106, cust_item_rank
        gc.collect()
        
        return cust_fund_rating_6m, cust_item_recmd
    
    @log_recorder    
    def insert_cust_fund_rating(self, cust_fund_rating_6m):
        """insert DB:cust_fund_rating"""
        try:
            # 格式欄位名稱調整
            cust_fund_rating_6m = cust_fund_rating_6m.reset_index(drop=True)
            cust_fund_rating_6m = cust_fund_rating_6m.rename(columns={'user_id':'cust_no'})
            # fund_rating 轉文字
            cust_fund_rating_6m_dict = cust_fund_rating_6m.set_index('cust_no').T.to_dict('dict')
            cust_fund_rating_6m_str = [str(x) for x in cust_fund_rating_6m_dict.values()]
            # create DataFrame、新增時間日期欄位
            input_df = pd.DataFrame()
            input_df['cust_no'] = list(cust_fund_rating_6m_dict.keys())
            input_df['fund_rating'] = cust_fund_rating_6m_str
            input_df['data_dt'] = self.data_dt # w103的etl_dt
            input_df['etl_dt'] = date.today()
            # insert DB
            sql_job = SQLHelper(etl_dt=self.today, table_name='cust_fund_rating', sql=None, source_db=None, target_db='feature')
            sql_job.to_target_db(df=input_df, page_size=100000, truncate=False, delete=True, update_time=True)
            del cust_fund_rating_6m, cust_fund_rating_6m_dict, cust_fund_rating_6m_str
            gc.collect()
        except:
            self.logger.error('cust_fund_rating：update table error', exc_info=True)
            raise
    
    @log_recorder    
    def add_card_info(self, cust_item_recmd):
        """加入卡片文案"""
        # min, max, base
        score_min = cust_item_recmd[['cust_no','rating']].groupby(['cust_no'])[['rating']].min().reset_index().rename(columns={'rating':'min'})
        score_max = cust_item_recmd[['cust_no','rating']].groupby(['cust_no'])[['rating']].max().reset_index().rename(columns={'rating':'max'})
        # merge
        score_df = score_min.merge(score_max, on='cust_no', how='left')
        score_df['base']= score_df.apply(lambda x: x['max']-x['min'], axis=1)
        # left join to cust_item_recmd
        cust_item_recmd = cust_item_recmd.merge(score_df, on='cust_no', how='left')
        # new score column & 計算 score
        cust_item_recmd.insert(4, 'score', 0)
        cust_item_recmd['score'] = cust_item_recmd.apply(lambda x: (x['rating']-x['min'])/x['base'] if x['base'] > 0 else 0, axis=1)
        # 四捨五入至整數
        cust_item_recmd['score'] = cust_item_recmd.apply(lambda x: int(round(x['score']*100, 0)) if x['score'] < 1 else 99, axis=1)
        # 加上中文，分數小於70留空字串
        cust_item_recmd['score'] = cust_item_recmd.apply(lambda x: '好感度 {}%'.format(str(x['score'])) if x['score'] >= 70 else '', axis=1)
        # rename
        cust_item_recmd = cust_item_recmd.rename(columns={'score':'card_score'})
        # 去除不要的 columns
        cust_item_recmd = cust_item_recmd.drop(['rating','min','max','base'], axis=1)
        
        ### 調整型別 & 加入data_dt, etl_dt
        cust_item_recmd['cust_no'] = cust_item_recmd['cust_no'].astype(str)
        cust_item_recmd['fund_id'] = cust_item_recmd['fund_id'].astype(str)
        cust_item_recmd['rank'] = cust_item_recmd['rank'].astype(int)
        cust_item_recmd['data_dt'] = self.data_dt # w103的etl_dt
        cust_item_recmd['etl_dt'] = date.today()
        del score_min, score_max, score_df
        gc.collect()
        return cust_item_recmd
        
    def run(self):
        self.logger.info('old_cust_recommend:ETL start')
        try:
            ### select data:基金資料源
            w103 = self.get_w103()
            w106 = self.get_w106()
            w118 = self.get_w118()
            cust_txn_amt =  self.get_cust_txn_amt()
            ### select data:近半年第一次申購基金
            last_6m_new_cust = self.get_last_6m_new_cust()

            ### 記錄來源資料表更新時間
            self.data_dt = max(w103['etl_dt'])

            ### 產製基金相似矩陣、數位通路基金交易
            item_matrix, txn_dig = self.item_similarity(w103, w106, w118)
            
            ### 產製近三個月申購贖回基金註記
            cust_txn_exclude = self.get_cust_txn_3m()
            
            ### 產製近半年第一次申購基金舊戶推薦結果、所有舊戶推薦結果
            cust_fund_rating_6m, cust_item_recmd = self.cust_item_recmd(cust_txn_amt, item_matrix, txn_dig, w106, last_6m_new_cust, cust_txn_exclude)
        
            ### insert DB:cust_fund_rating
            self.insert_cust_fund_rating(cust_fund_rating_6m)

            ### 加入卡片文案
            old_cust_recommend = self.add_card_info(cust_item_recmd)

            ### insert DB:old_cust_recommend
            sql_job = SQLHelper(etl_dt=self.today, table_name='old_cust_recommend', sql=None, source_db=None, target_db='feature')
            sql_job.to_target_db(df=old_cust_recommend, page_size=100000, truncate=False, delete=True, update_time=True)
        except:
            self.logger.error('old_cust_recommend:ETL error', exc_info=True)
            raise
        else:
            self.logger.info('old_cust_recommend:ETL end')

if __name__ == '__main__':
    etl = FeatureETL(date.today())
    if etl.check():
        etl.run()
    else:
        etl.logger.info("Table didn't update because rawdata is abnormal")