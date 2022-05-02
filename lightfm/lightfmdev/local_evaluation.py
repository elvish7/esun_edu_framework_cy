import pandas as pd
from tqdm import tqdm

class Evaluation:
    
    def __init__(self, date, pred, purchase_hist, eva_df):
        self.today = date
        self.pred = pred
        self.purchase_hist = purchase_hist
        self.ans = self.answer(eva_df)

    def show(self):
        print(f"Date: {self.today}\n")
        print(f"Ans: {self.ans}\n")
        
    def answer(self, df):
        return df.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

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
    
    def warm_cold_list(self):
        warm_user = list(self.purchase_hist.keys() & self.ans.keys())
        cold_user = list(self.ans.keys() - self.purchase_hist.keys())
        return warm_user, cold_user

    def results(self, prediction):
        p, upper = 0, 0
        count_all = len(prediction)
        for u, pred in tqdm(prediction.items(), total=count_all):
            p5, up5 = self.precision_at_5(u, pred)
            p += p5
            upper += up5

        up = 0
        for u, list_ in self.ans.items():
            up += min(len(set(list_)),5)/5
            
        return p/count_all, upper/count_all

    def precision_at_5(self, user, pred):
        try:
            y_true = self.ans[user]
            tp = len(set(y_true) & set(pred))
            return tp/5, min(len(set(y_true)),5)/5
        except:
            return 0, 0
