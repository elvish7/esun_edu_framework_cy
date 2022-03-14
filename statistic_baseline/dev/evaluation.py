import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from common.ETLBase import ProcessBase
from experiments.old_cust_baseline.experiment_module import old_cust_CF_baseline
from experiments.old_cust_baseline.dataloader.utils import get_conn
#from crv_tools import db_tool
#conn = db_tool.get_conn() 

class Evaluation(ProcessBase):
    
    def __init__(self, date, pred):
        self.today = date
        self.conn = get_conn('edu')
        self.pred = pred
        self.ans = self.answer(self.today, self.conn)

    def show(self):
        print(f"Date: {self.today}\n")
        print(f"Ans: {self.ans}\n")
        
    def answer(self, date, conn):
        df = self.read(date, conn)
        return df.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

    def read(self, date, conn):
        after_1d_dt, after_1m_dt = self.get_data_dt(date, 1)
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
                        where wm_txn_code='1'and txn_dt>='{d_start}' and txn_dt<='{d_end}'), 
                cte2 as (select distinct replace(wm_prod_code, ' ', '') as wm_prod_code
                        from sinica.witwo106
                        where replace(prod_detail_type_code, ' ','') in ('FNDF','FNDD')) 
            select cte1.cust_no, cte1.wm_prod_code from cte1 inner join cte2 on cte1.wm_prod_code=cte2.wm_prod_code order by cust_no limit 2742
            """.format(d_start=after_1d_dt, d_end=after_1m_dt)
        return pd.read_sql(sql, conn)  

    def get_data_dt(self, etl_time_string, backward_months):
        today = pd.to_datetime(etl_time_string, format='%Y-%m-%d')
        data_start_dt = today + relativedelta(days=1)
        data_end_dt = today + relativedelta(months=backward_months)
        data_start = data_start_dt.strftime('%Y-%m-%d')
        data_end = data_end_dt.strftime('%Y-%m-%d')
        return data_start, data_end

    def results(self):
        p = 0
        count = len(self.pred)
        for u, pred in tqdm(self.pred.items(), total=count):
            p += self.precision_at_5(u, pred)
        return p/count         
    
    def precision_at_5(self, user, pred):
        y_true = self.ans[user]
        tp = len(set(y_true) & set(pred))
        return tp/5
