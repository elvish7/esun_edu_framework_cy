import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from common.ETLBase import ProcessBase
from db_connection.utils import get_conn

class Evaluation(ProcessBase):
    
    def __init__(self, date, pred, duration):
        self.today = date
        self.duration = duration
        self.conn = get_conn('edu')
        self.pred = pred
        self.ans = self.answer(self.today, self.conn, self.duration)

    def show(self):
        print(f"Date: {self.today}\n")
        print(f"Ans: {self.ans}\n")
    
    def coverage(self):
        covered = len(set(self.pred.keys()) & set(self.ans.keys()))
        uncovered = len(set(self.ans.keys())) - covered
        return covered, uncovered
    
    def answer(self, date, conn, duration):
        df = self.read(date, conn, duration)
        return df.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

    def read(self, date, conn, duration):
        after_1d_dt, after_1m_dt, after_7d_dt = self.get_data_dt(date, 1)

        if duration == '7d':
            d_end = after_7d_dt
        else:
            d_end = after_1m_dt

        sql = """
            with 
                cte1 as (select cust_id as cust_no, 
                                replace(wm_prod_code, ' ', '') as wm_prod_code, 
                                txn_dt, 
                                (case when wms_txn_amt_twd is null then 1 else wms_txn_amt_twd end) as txn_amt,
                                dta_src,
                                deduct_cnt,
                                etl_dt
                        from sinica.witwo103_hist 
                        where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}' and deduct_cnt <=1), 
                cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                        from sinica.witwo106
                        where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')) 
            select cte1.cust_no, cte1.wm_prod_code from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code order by cust_no
            """.format(d_start=after_1d_dt, d_end=d_end)
        return pd.read_sql(sql, conn)  

    def get_data_dt(self, etl_time_string, backward_months):
        today = pd.to_datetime(etl_time_string, format='%Y-%m-%d')
        data_start_dt = today + relativedelta(days=1)
        data_1m_end_dt = today + relativedelta(months=backward_months)
        data_7d_end_dt = today + relativedelta(days=7)
        data_start = data_start_dt.strftime('%Y-%m-%d')
        data_1m_end = data_1m_end_dt.strftime('%Y-%m-%d')
        data_7d_end = data_7d_end_dt.strftime('%Y-%m-%d')
        return data_start, data_1m_end, data_7d_end

    def results(self):
        p, upper = 0, 0
        count_all = len(self.ans)
        count_warm = len(set(self.pred.keys()) & set(self.ans.keys()))
        for u, pred in tqdm(self.pred.items(), total=count_all):
            p5, up5 = self.precision_at_5(u, pred)
            p += p5
            upper += up5

        up = 0
        for u, list_ in self.ans.items():
            up += min(len(set(list_)),5)/5

        return p/count_all, p/count_warm, upper/count_all, up/count_all, upper/count_warm

    
    def precision_at_5(self, user, pred):
        try:
            y_true = self.ans[user]
            tp = len(set(y_true) & set(pred))
            return tp/5, min(len(set(y_true)),5)/5
        except:
            return 0, 0
