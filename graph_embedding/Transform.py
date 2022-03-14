import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import relativedelta

class Transform:
    def __init__(self, df, end_dt, decay_factor, decay_window):
        self.end = datetime(*[int(item) for item in end_dt.split('-')])
        self.factor = decay_factor
        self.window = decay_window
        self.w103 = df

    def diff_months(self, start, end):
        return (end.year - start.year) * 12 + (end.month  - start.month )
    
    def scale(self, start_dt):
        diff = self.diff_months(start_dt, self.end)
        return self.factor ** (diff//self.window)
    
    def weight_amount(self, txn_dt, txn_amt):
        return pd.DataFrame([amt*self.scale(dt) for dt, amt in zip(txn_dt, txn_amt)])
        
     
    def transformation(self):
        w103_df = self.w103
        w103_df = w103_df.assign(weighted_txn_amt = lambda x: self.weight_amount(pd.to_datetime(x['txn_dt']), x['txn_amt']))
        #w103_df.groupby["cust_no","wm_prod_code"].apply(lambda y: y["weighted_txn_amt"].sum(), axis=0)#lambda? sort by time filer by window add decay
        return w103_df
