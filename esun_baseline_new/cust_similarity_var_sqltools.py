import sys
import pytz
import pandas as pd
from datetime import date, datetime
from psycopg2 import sql
from sql_tools import SQLHelper
from mlaas_tools.feature_tool import FeatureBase
from psycopg2.extras import execute_values

class FeatureETL(FeatureBase):
    '''cust_similarity_var etl'''
    def __init__(self, etl_dt):
        '''init'''
        super().__init__(etl_dt)
        self.etl_dt = etl_dt

        
    def check(self):
        '''check rawdata of cm_customer'''
        self.logger.info('The "check rawdata of cm_customer" is going to start.')
        rconn = self.dbset.get_rawdata_db_conn()
        try:
            r_cur = rconn.cursor()
            sqlstring = """select count(cust_no) from mlaas_rawdata.cm_customer"""
            r_cur.execute(sqlstring)
            num = r_cur.fetchall()
        except:
            self.logger.error('rawdata of cm_customer was failed', exc_info=True)
            raise
        finally:
            rconn.close()
        # 筆數非0筆
        if list(num[0])[0] > 0:
            self.logger.info('rawdata of cm_customer is ok.')
            return True
        else:
            self.logger.error('rawdata of cm_customer is wrong.')
            return False
        self.logger.info('check rawdata of cm_customer end')
        
#cust_pop
    def cust_pop(self):
        '''
        cte1:篩選個人戶、年紀 ; 類別變數encoding，NA用XXX0做，cust_vintage空值補平均數
        cte2:處理duplicates:若有一樣的cust_no留下所有欄位最大值 ; cust_vintage: Normalization
        '''
        self.logger.info('create_pop start.')
        try:
            sql = """
                with
                    cte1 as(
                        select distinct
                            cust_no,
                            etl_dt as data_dt,
                            age,
                            (case when gender_code = 'M' then 1 else 0 end)::numeric as gender_code1,
                            (case when gender_code = 'F' then 1 else 0 end)::numeric as gender_code2,
                            (case when gender_code is null then 1 else 0 end)::numeric as gender_code0,
                            coalesce((select avg(cust_vintage) from mlaas_rawdata.cm_customer), cust_vintage) as cust_vintage,
                            (case when income_range_code = '1' then 1 else 0 end)::numeric as income_range_code1,
                            (case when income_range_code = '2' then 1 else 0 end)::numeric as income_range_code2,
                            (case when income_range_code = '3' then 1 else 0 end)::numeric as income_range_code3,
                            (case when income_range_code = '4' then 1 else 0 end)::numeric as income_range_code4,
                            (case when income_range_code is null then 1 else 0 end)::numeric as income_range_code0
                        from mlaas_rawdata.cm_customer
                        where (age between 20 and 69)
                        and biz_line_code = 'P'
                        ),
                    cte2 as (
                        select
                            cust_no,
                            data_dt,
                            age,
                            gender_code1,
                            gender_code2,
                            gender_code0,
                            ((cust_vintage
                            - (select min(cust_vintage) from mlaas_rawdata.cm_customer))
                            /((select max(cust_vintage) from mlaas_rawdata.cm_customer)
                            - (select min(cust_vintage) from mlaas_rawdata.cm_customer))) as cust_vintage,
                            income_range_code1,
                            income_range_code2,
                            income_range_code3,
                            income_range_code4,
                            income_range_code0,
                            row_number() over (partition by cust_no order by age desc,
                                                                             cust_vintage desc,
                                                                             income_range_code1 desc,
                                                                             income_range_code2 desc,
                                                                             income_range_code3 desc,
                                                                             income_range_code4 desc,
                                                                             income_range_code0 desc,
                                                                             gender_code1 desc,
                                                                             gender_code2 desc,
                                                                             gender_code0 desc) as rank
                        from cte1
                        )
                    select cust_no,
                            data_dt,
                            age,
                            gender_code1,
                            gender_code2,
                            gender_code0,
                            cust_vintage,
                            income_range_code1,
                            income_range_code2,
                            income_range_code3,
                            income_range_code4,
                            income_range_code0
                    from cte2
                    where rank = 1
                """
            #insert new data
            sql_job = SQLHelper(etl_dt=self.etl_dt, table_name='cust_pop', sql=sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=True, delete=False, update_time=True)
            
            self.logger.info('create_pop was succesfully finished.')
        except:
            self.logger.error('cust_pop data was failed.', exc_info=True)
            raise
        
#cust_ind
    def cust_ind(self):
        '''
        cte1:處理duplicates:若有一樣的cust_no留下所有欄位最大值的那一欄(只要是Y就給1，其餘給0，若有空值也給0)
        '''
        self.logger.info('cm_cust_ind start.')
        try:
            sql = """
                with
                    cte1 as (
                        select distinct
                                cust_no,
                                (case when world_card_ind = 'Y' then 1 else 0 end) as world_card_ind,
                                (case when pure_cc_ind = 'Y' then 1 else 0 end) as pure_cc_ind,
                                (case when payroll_acct_ind = 'Y' then 1 else 0 end) as payroll_acct_ind,
                                (case when insure_ind = 'Y' then 1 else 0 end) as insure_ind,
                                (case when gov_teacher_ind = 'Y' then 1 else 0 end) as gov_teacher_ind,
                                (case when esa_ind = 'Y' then 1 else 0 end) as esa_ind,
                                (case when pure_house_mtg_ln_ind = 'Y' then 1 else 0 end) as pure_house_mtg_ln_ind,
                                (case when employer_ind = 'Y' then 1 else 0 end) as employer_ind,
                                (case when curr_traveler_card_ind = 'Y' then 1 else 0 end) as curr_traveler_card_ind,
                                (case when cc_vip_ind = 'Y' then 1 else 0 end) as cc_vip_ind,
                                row_number() over (partition by cust_no order by
                                                                             world_card_ind desc,
                                                                             pure_cc_ind desc,
                                                                             payroll_acct_ind desc,
                                                                             insure_ind desc,
                                                                             gov_teacher_ind desc,
                                                                             esa_ind desc,
                                                                             pure_house_mtg_ln_ind desc,
                                                                             employer_ind desc,
                                                                             curr_traveler_card_ind desc,
                                                                             cc_vip_ind desc) as rank
                            from mlaas_rawdata.cm_cust_ind)
                    select
                        cust_no,
                        world_card_ind,
                        pure_cc_ind,
                        payroll_acct_ind,
                        insure_ind,
                        gov_teacher_ind,
                        esa_ind,
                        pure_house_mtg_ln_ind,
                        employer_ind,
                        curr_traveler_card_ind,
                        cc_vip_ind
                    from cte1
                    where rank = 1
                """
            sql_job = SQLHelper(etl_dt=self.etl_dt, table_name='cust_ind', sql=sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=True, delete=False, update_time=True)
            self.logger.info('cust_ind was succesfully finished.')
        except:
            self.logger.error('cust_ind data was failed.', exc_info=True)
            raise
#acct
    def acct(self):
        '''cte1:處理duplicates:若有一樣的cust_no留下所有欄位最大值的那一欄(產品註記給1、0)'''
        self.logger.info('acct start.')
        try:
            sql = """
                with
                    cte1 as (
                        select distinct
                            cust_no,
                            (case when acct_type_code = 'MD' then 1 else 0 end) as md_ind,
                            (case when acct_type_code = 'MY' then 1 else 0 end) as my_ind,
                            (case when acct_type_code = 'FD' then 1 else 0 end) as fd_ind,
                            (case when acct_type_code = 'FY' then 1 else 0 end) as fy_ind,
                            (case when acct_type_code = 'CC' then 1 else 0 end) as cc_ind,
                            (case when acct_type_code = 'LN' then 1 else 0 end) as ln_ind
                        from mlaas_rawdata.cm_account
                        where acct_status_code='A'),
                    cte2 as (
                        select
                            cte1.*,
                            row_number() over (partition by cust_no order by
                                                                     md_ind desc,
                                                                     my_ind desc,
                                                                     fd_ind desc,
                                                                     fy_ind desc,
                                                                     cc_ind desc,
                                                                     ln_ind desc) as rank
                        from cte1)
                    select
                        cust_no,
                        md_ind,
                        my_ind,
                        fd_ind,
                        fy_ind,
                        cc_ind,
                        ln_ind
                    from cte2
                    where rank = 1
                """
            sql_job = SQLHelper(etl_dt=self.etl_dt, table_name='acct', sql=sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=True, delete=False, update_time=True)
            self.logger.info('acct was succesfully finished.')
        except:
            self.logger.error('acct data was failed.', exc_info=True)
            raise

#aum
    def aum(self):
        '''
        cte1:處理duplicates:若有一樣的cust_no留下所有欄位最大值的那一欄
        cte2:算最大值、最小值
        cte3:aum var: Normalization (如果分母有0，最後直接給0)
        '''
        self.logger.info('aum start.')
        try:
            sql = """
                with
                    cte1 as (
                        select
                                cust_no,
                                aum_twd_new,
                                aum_twd_mf1,
                                aum_twd_mf2,
                                aum_twd_in,
                                aum_twd_gd,
                                aum_twd_etf,
                                aum_twd_bo,
                                etl_dt,
                                row_number() over (partition by cust_no order by
                                                                        aum_twd_new desc,
                                                                        aum_twd_mf1 desc,
                                                                        aum_twd_mf2 desc,
                                                                        aum_twd_in desc,
                                                                        aum_twd_gd desc,
                                                                        aum_twd_etf desc,
                                                                        aum_twd_bo desc) as rank
                             from mlaas_rawdata.cm_cust_aum),
                    cte2 as (
                        select
                            cte1.*,
                            max(aum_twd_new) over (partition by etl_dt) as max_new,
                            max(aum_twd_mf1) over (partition by etl_dt) as max_mf1,
                            max(aum_twd_mf2) over (partition by etl_dt) as max_mf2,
                            max(aum_twd_in) over (partition by etl_dt) as max_in,
                            max(aum_twd_gd) over (partition by etl_dt) as max_gd,
                            max(aum_twd_etf) over (partition by etl_dt) as max_etf,
                            max(aum_twd_bo) over (partition by etl_dt) as max_bo,
                            min(aum_twd_new) over (partition by etl_dt) as min_new,
                            min(aum_twd_mf1) over (partition by etl_dt) as min_mf1,
                            min(aum_twd_mf2) over (partition by etl_dt) as min_mf2,
                            min(aum_twd_in) over (partition by etl_dt) as min_in,
                            min(aum_twd_gd) over (partition by etl_dt) as min_gd,
                            min(aum_twd_etf) over (partition by etl_dt) as min_etf,
                            min(aum_twd_bo) over (partition by etl_dt) as min_bo
                        from cte1
                        where rank = 1)
                    select
                        cust_no,
                        (case when (max_new - min_new) = 0 then 0 else (aum_twd_new - min_new)/(max_new - min_new) end)as aum_twd_new,
                        (case when (max_mf1 - min_mf1) = 0 then 0 else (aum_twd_new - min_mf1)/(max_mf1 - min_mf1) end)as aum_twd_mf1,
                        (case when (max_mf2 - max_mf2) = 0 then 0 else (aum_twd_new - min_mf2)/(max_mf2 - max_mf2) end)as aum_twd_mf2,
                        (case when (max_in - min_in) = 0 then 0 else (aum_twd_new - min_in)/(max_in - min_in) end)as aum_twd_in,
                        (case when (max_gd - min_gd) = 0 then 0 else (aum_twd_new - min_gd)/(max_gd - min_gd) end)as aum_twd_gd,
                        (case when (max_etf - min_etf) = 0 then 0 else (aum_twd_new - min_etf)/(max_etf - min_etf) end)as aum_twd_etf,
                        (case when (max_bo - min_bo) = 0 then 0 else (aum_twd_new - min_bo)/(max_bo - min_bo) end)as aum_twd_bo
                    from cte2
                """
            sql_job = SQLHelper(etl_dt=self.etl_dt, table_name='aum', sql=sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=True, delete=False, update_time=True)
            self.logger.info('aum was succesfully finished.')
        except:
            self.logger.error('aum data was failed.', exc_info=True)
            raise

#w103
    def w103(self):
        '''
        算近六個月"第一次"在數位通路上交易的註記:
        cte1:條件:境內及境外基金、數位通路、交易方式:行、網、企網銀、排掉理財快易通、定期定額為自主購買，判斷交易日是否在近六個月內
        '''
        self.logger.info('w103 start.')
        try:
            sql = """
                    select 
                        cust_no,
                        1 as last_6m_new_ind
                    from eb_ofsn_wm_fund.cust_fund_rating
                """
            sql_job = SQLHelper(etl_dt=self.etl_dt, table_name='w103', sql=sql, source_db='feature', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=True, delete=False, update_time=True)
            self.logger.info('w103 was succesfully finished.')
        except:
            self.logger.error('w103 data was failed.', exc_info=True)
            raise

#aum_m
    def aum_m(self):
        '''
        cte1:篩選近4個月的資料
        '''
        self.logger.info('aum_m start.')
        try:
            sql = """
                with
                    cte1 as (
                        select
                            cust_no,
                            data_ym,
                            aum_twd_val,
                            (aum_twd_mf1_val + aum_twd_mf2_val) as aum_twd_mf,
                            aum_twd_bo,
                            aum_twd_in,
                            aum_twd_gd,
                            aum_twd_etf,
                            row_number() over (partition by cust_no, data_ym order by
                                                                                aum_twd_val desc,
                                                                                (aum_twd_mf1_val + aum_twd_mf2_val) desc,
                                                                                aum_twd_bo desc,
                                                                                aum_twd_in desc,
                                                                                aum_twd_gd desc,
                                                                                aum_twd_etf desc) as rank
                        from mlaas_rawdata.cm_cust_aum_m
                        where data_ym between
                                        date_trunc('month', (select max(data_ym) from mlaas_rawdata.cm_cust_aum_m))::date - interval '4 month'
                                    and (select max(data_ym) from mlaas_rawdata.cm_cust_aum_m)
                    )
                select cust_no,
                        data_ym,
                        aum_twd_val,
                        aum_twd_mf,
                        aum_twd_bo,
                        aum_twd_in,
                        aum_twd_gd,
                        aum_twd_etf
                from cte1
                where rank = 1
                """
            sql_job = SQLHelper(etl_dt=self.etl_dt, table_name='aum_m', sql=sql, source_db='rawdata', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=True, delete=False, update_time=True)
            self.logger.info('aum_m was succesfully finished.')
        except:
            self.logger.error('aum_m data was failed.', exc_info=True)
            raise

#etl_all
    def etl_all(self):
        '''
        cte1:篩選last_6m_ind = 1的人
        cte2:併前1個月、前2個月、前3個月的欄位、且日期都篩選那個月的資料
        cte3:計算前1個月與上個月變化率、計算前2個月與上個月變化率、計算前3個月與上個月變化率(若上個月值為0，那該欄位直接給1)
        cte4:計算累加3個月變化率
        cte5:併到pop，沒有併到資料的人給0
        '''
        self.logger.info('etl_all start.')
        try:
            sql="""
                    with
                        cte1 as (
                            select
                                A.cust_no,
                                A.data_dt,
                                current_date as etl_dt,
                                A.age,
                                A.gender_code1,
                                A.gender_code2,
                                A.gender_code0,
                                A.cust_vintage,
                                A.income_range_code1,
                                A.income_range_code2,
                                A.income_range_code3,
                                A.income_range_code4,
                                A.income_range_code0,
                                coalesce(world_card_ind, 0) as world_card_ind,
                                coalesce(pure_cc_ind, 0) as pure_cc_ind,
                                coalesce(payroll_acct_ind, 0) as payroll_acct_ind,
                                coalesce(insure_ind, 0) as insure_ind,
                                coalesce(gov_teacher_ind, 0) as gov_teacher_ind,
                                coalesce(esa_ind, 0) as esa_ind,
                                coalesce(pure_house_mtg_ln_ind, 0) as pure_house_mtg_ln_ind,
                                coalesce(employer_ind, 0) as employer_ind,
                                coalesce(curr_traveler_card_ind, 0) as curr_traveler_card_ind,
                                coalesce(cc_vip_ind, 0) as cc_vip_ind,
                                coalesce(md_ind, 0) as md_ind,
                                coalesce(my_ind, 0) as my_ind,
                                coalesce(fd_ind, 0) as fd_ind,
                                coalesce(fy_ind, 0) as fy_ind,
                                coalesce(cc_ind, 0) as cc_ind,
                                coalesce(ln_ind, 0) as ln_ind,
                                coalesce(aum_twd_new, 0) as aum_twd_new,
                                coalesce(aum_twd_mf1, 0) aum_twd_mf1,
                                coalesce(aum_twd_mf2, 0) as aum_twd_mf2,
                                coalesce(aum_twd_in, 0) as aum_twd_in,
                                coalesce(aum_twd_gd, 0) as aum_twd_gd,
                                coalesce(aum_twd_etf, 0) as aum_twd_etf,
                                coalesce(aum_twd_bo, 0) as aum_twd_bo,
                                coalesce(last_6m_new_ind, 0) as last_6m_new_ind
                            from eb_ofsn_wm_fund.cust_pop as A
                            left join eb_ofsn_wm_fund.cust_ind as B
                            on A.cust_no = B.cust_no
                            left join eb_ofsn_wm_fund.acct as C
                            on A.cust_no = C.cust_no
                            left join eb_ofsn_wm_fund.aum as D
                            on A.cust_no = D.cust_no
                            left join eb_ofsn_wm_fund.w103 as E
                            on A.cust_no = E.cust_no
                        ),
                        cte2 as (
                            select
                                cust_no,
                                data_ym,
                                aum_twd_val,
                                aum_twd_mf,
                                aum_twd_bo,
                                aum_twd_in,
                                aum_twd_gd,
                                aum_twd_etf
                            from eb_ofsn_wm_fund.aum_m as t
                            where exists (
                                    select pop.cust_no
                                    from eb_ofsn_wm_fund.cust_pop as pop
                                    where pop.cust_no=t.cust_no)
                            and data_ym = (select max(data_ym) from eb_ofsn_wm_fund.aum_m)
                        ),
                        cte3 as (
                            select
                                A.*,
                                B.data_ym as data_ym2,
                                B.aum_twd_val as aum_twd_val2,
                                B.aum_twd_mf as aum_twd_mf2,
                                B.aum_twd_bo as aum_twd_bo2,
                                B.aum_twd_in as aum_twd_in2,
                                B.aum_twd_gd as aum_twd_gd2,
                                B.aum_twd_etf as aum_twd_etf2,
                                C.data_ym as data_ym3,
                                C.aum_twd_val as aum_twd_val3,
                                C.aum_twd_mf as aum_twd_mf3,
                                C.aum_twd_bo as aum_twd_bo3,
                                C.aum_twd_in as aum_twd_in3,
                                C.aum_twd_gd as aum_twd_gd3,
                                C.aum_twd_etf as aum_twd_etf3,
                                D.data_ym as data_ym4,
                                D.aum_twd_val as aum_twd_val4,
                                D.aum_twd_mf as aum_twd_mf4,
                                D.aum_twd_bo as aum_twd_bo4,
                                D.aum_twd_in as aum_twd_in4,
                                D.aum_twd_gd as aum_twd_gd4,
                                D.aum_twd_etf as aum_twd_etf4
                            from cte2 as A
                            left join eb_ofsn_wm_fund.aum_m as B
                            on A.cust_no = B.cust_no
                            and date_trunc('month', A.data_ym)::date - interval '1 month' = B.data_ym
                            left join eb_ofsn_wm_fund.aum_m as C
                            on A.cust_no = C.cust_no
                            and date_trunc('month', A.data_ym)::date - interval '2 month' = C.data_ym
                            left join eb_ofsn_wm_fund.aum_m as D
                            on A.cust_no = D.cust_no
                            and date_trunc('month', A.data_ym)::date - interval '3 month' = D.data_ym
                            where B.data_ym = date_trunc('month', A.data_ym)::date - interval '1 month'
                            and C.data_ym = date_trunc('month', A.data_ym)::date - interval '2 month'
                            and D.data_ym = date_trunc('month', A.data_ym)::date - interval '3 month'
                        ),
                        cte4 as (
                            select
                                cust_no,
                                data_ym,
                                (case when aum_twd_val2 = 0 then 1
                                 else (aum_twd_val - aum_twd_val2)/aum_twd_val2 end) as aum_twd_val_ratio1,
                                (case when aum_twd_mf2 = 0 then 1
                                 else (aum_twd_mf - aum_twd_mf2)/aum_twd_mf2 end) as aum_twd_mf_ratio1,
                                (case when aum_twd_bo2 = 0 then 1
                                 else (aum_twd_bo - aum_twd_bo2)/aum_twd_bo2 end) as aum_twd_bo_ratio1,
                                (case when aum_twd_in2 = 0 then 1
                                 else (aum_twd_in - aum_twd_in2)/aum_twd_in2 end) as aum_twd_in_ratio1,
                                (case when aum_twd_gd2 = 0 then 1
                                 else (aum_twd_gd - aum_twd_gd2)/aum_twd_gd2 end) as aum_twd_gd_ratio1,
                                (case when aum_twd_etf2 = 0 then 1
                                 else (aum_twd_etf - aum_twd_etf2)/aum_twd_etf2 end) as aum_twd_etf_ratio1,
                                (case when aum_twd_val3 = 0 then 1
                                 else (aum_twd_val2 - aum_twd_val3)/aum_twd_val3 end) as aum_twd_val_ratio2,
                                (case when aum_twd_mf3 = 0 then 1
                                 else (aum_twd_mf2 - aum_twd_mf3)/aum_twd_mf3 end) as aum_twd_mf_ratio2,
                                (case when aum_twd_bo3 = 0 then 1
                                 else (aum_twd_bo2 - aum_twd_bo3)/aum_twd_bo3 end) as aum_twd_bo_ratio2,
                                (case when aum_twd_in3 = 0 then 1
                                 else (aum_twd_in2 - aum_twd_in3)/aum_twd_in3 end) as aum_twd_in_ratio2,
                                (case when aum_twd_gd3 = 0 then 1
                                 else (aum_twd_gd2 - aum_twd_gd3)/aum_twd_gd3 end) as aum_twd_gd_ratio2,
                                (case when aum_twd_etf3 = 0 then 1
                                 else (aum_twd_etf2 - aum_twd_etf3)/aum_twd_etf3 end) as aum_twd_etf_ratio2,
                                (case when aum_twd_val4 = 0 then 1
                                 else (aum_twd_val3 - aum_twd_val4)/aum_twd_val4 end) as aum_twd_val_ratio3,
                                (case when aum_twd_mf4 = 0 then 1
                                 else (aum_twd_mf3 - aum_twd_mf4)/aum_twd_mf4 end) as aum_twd_mf_ratio3,
                                (case when aum_twd_bo4 = 0 then 1
                                 else (aum_twd_bo3 - aum_twd_bo4)/aum_twd_bo4 end) as aum_twd_bo_ratio3,
                                (case when aum_twd_in4 = 0 then 1
                                 else (aum_twd_in3 - aum_twd_in4)/aum_twd_in4 end) as aum_twd_in_ratio3,
                                (case when aum_twd_gd4 = 0 then 1
                                 else (aum_twd_gd3 - aum_twd_gd4)/aum_twd_gd4 end) as aum_twd_gd_ratio3,
                                (case when aum_twd_etf4 = 0 then 1
                                 else (aum_twd_etf3 - aum_twd_etf4)/aum_twd_etf4 end) as aum_twd_etf_ratio3
                            from cte3
                        ),
                        cte5 as (
                            select
                                cust_no,
                                aum_twd_val_ratio1 as aum_m1,
                                aum_twd_mf_ratio1 as fund_m1,
                                aum_twd_bo_ratio1 as bond_m1,
                                aum_twd_in_ratio1 as insurance_m1,
                                aum_twd_gd_ratio1 as gold_m1,
                                aum_twd_etf_ratio1 as etf_m1,
                                (aum_twd_val_ratio1 + aum_twd_val_ratio2 + aum_twd_val_ratio3) as aum_m3,
                                (aum_twd_mf_ratio1 + aum_twd_mf_ratio2 + aum_twd_mf_ratio3) as fund_m3,
                                (aum_twd_bo_ratio1 + aum_twd_bo_ratio2 + aum_twd_bo_ratio3) as bond_m3,
                                (aum_twd_in_ratio1 + aum_twd_in_ratio2 + aum_twd_in_ratio2) as insurance_m3,
                                (aum_twd_gd_ratio1 + aum_twd_gd_ratio2 + aum_twd_gd_ratio3) as gold_m3,
                                (aum_twd_etf_ratio1 + aum_twd_etf_ratio2 + aum_twd_etf_ratio3) as etf_m3
                            from cte4)
                        select
                            A.*,
                            coalesce(aum_m1, 0) as aum_m1,
                            coalesce(fund_m1, 0) as fund_m1,
                            coalesce(bond_m1, 0) as bond_m1,
                            coalesce(insurance_m1, 0) as insurance_m1,
                            coalesce(gold_m1, 0) as gold_m1,
                            coalesce(etf_m1, 0) as etf_m1,
                            coalesce(aum_m3, 0) as aum_m3,
                            coalesce(fund_m3, 0) as fund_m3,
                            coalesce(bond_m3, 0) as bond_m3,
                            coalesce(insurance_m3, 0) as insurance_m3,
                            coalesce(gold_m3, 0) as gold_m3,
                            coalesce(etf_m3, 0) as etf_m3
                        from cte1 as A
                        left join cte5 as B
                        on A.cust_no = B.cust_no
                    """
            sql_job = SQLHelper(etl_dt=self.etl_dt, table_name='cust_similarity_var', sql=sql, source_db='feature', target_db='feature')
            sql_job.to_target_db(df=None, page_size=100000, truncate=True, delete=False, update_time=True)
        except:
            self.logger.error('etl_all was failed.', exc_info=True)
            raise
        else:
            self.logger.info('etl_all was succesfully finished.')

    def run(self):
        try:
            self.logger.info('run start!')
            # 1 insert rawdata to feature_db
            # 1-1 cust_pop
            self.cust_pop()
            # 1-2 cust_ind
            self.cust_ind()
            # 1-3 acct
            self.acct()
            # 1-4 aum
            self.aum()
            # 1-5 w103
            self.w103()
            # 1-6 aum_m
            self.aum_m()
            #2 insert final table in feature_db
            self.etl_all()
        except:
            self.logger.error('ETL was failed.', exc_info=True)
            raise
        else:
            self.logger.info('ETL was Finish!.')

if __name__ == '__main__':
    etl = FeatureETL(date.today())
    if etl.check():
        etl.run()