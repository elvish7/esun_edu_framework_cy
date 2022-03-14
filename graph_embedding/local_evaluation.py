import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

class Evaluation:
    
    def __init__(self, date, path, pred, purchase_hist):
        self.today = date
        self.path = path
        self.pred = pred
        self.purchase_hist = purchase_hist
        self.ans = self.answer(self.path)
    
    def show(self):
        print(f"Date: {self.today}\n") 
        coverage = len(set(self.pred.keys()) & set(self.ans.keys()))       
        print(f"Uppper-Bound: {coverage}\n")
 
    def answer(self, path):
        df = self.read(path)
        return df.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

    def read(self, path):
        return pd.read_csv(path, usecols=['cust_no', 'wm_prod_code'])  

    def results(self):
        p, upper = 0, 0
        count = len(self.ans)
        for u, pred in tqdm(self.pred.items(), total=count):
            p5, upper5 = self.precision_at_5(u, pred)
            p += p5
            upper += upper5 
        return p/count, upper/count         
    
    def purchase_statistic(self):
        buy_old, buy_new, warm_start_user, cold_start_user = 0, 0, 0, 0
        for u, ans in self.ans.items():
            if u in self.purchase_hist.keys():
                warm_start_user += 1
                if len(set(ans)) - len(set(ans) & set(self.purchase_hist[u])) > 0:
                    buy_new += 1
                else:
                    buy_old += 1
            else:
                cold_start_user += 1
        return buy_old, buy_new, warm_start_user, cold_start_user
     
    def predict_statistic(self):
        pred_old, pred_new, outlier = 0, 0, 0
        for u, ans in self.ans.items():
            if u in self.purchase_hist.keys():
                if len(set(self.pred[u])) - len(set(self.pred[u]) & set(self.purchase_hist[u])) > 0:
                    pred_new += 1
                else:
                    pred_old += 1
            else:
                outlier += 1
        return pred_old, pred_new, outlier
     
    def precision_at_5(self, user, pred):
        try:
            y_true = self.ans[user]
            tp = len(set(y_true) & set(pred))
            return min(1,tp/5), min(len(set(y_true)),5)/5
        except:
            return 0, 0

