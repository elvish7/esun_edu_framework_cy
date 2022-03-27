import math, json, fileinput, os, itertools, subprocess, io, sys, collections, fileinput, multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm

class TPR:
    def __init__(self, data, features):
        self.data = self.read(data)
        self.input_ui_path = 'input_ui.txt'
        self.input_iw_path = 'input_iw.txt'
        self.users = data['cust_no'].unique().tolist()
        self.items = data['wm_prod_code'].unique().tolist()
        self.user_dict = self.id2user()
        self.item_dict = self.id2item()
        self.interactions()
        self.build_feature_tuples(features)
        
    def read(self, record_df):
        return [(u, i) for u, i in zip(record_df['cust_no'], record_df['wm_prod_code'])]

    def interactions(self):
        data = self.data
        temp = open(f'{self.input_ui_path}', 'w')
        for record, count in collections.Counter(data).items():
            print(f'{record[0]}\t{record[1]}\t{count}', file=temp)
        
    def concat_feature_colon_value(self, header, my_list):
        """
        Takes as input a list and prepends the columns names to respective values in the list.
        For example: if my_list = [1,1,0,'del'],
        resultant output = ['f1:1', 'f2:1', 'f3:0', 'loc:del']
       
        """
        result = []
        for x,y in zip(header,my_list):
            res = str(x) +""+ str(y)
            result.append(res)
        return result
    
    def build_feature_tuples(self, features):
        """
        One user/item tuple: (item id, {feature name: feature weight})
        Returns a list of tuples
        """
        feature_subset = features.iloc[: , 1:] #drop the first column
        header = [f+':' for f in feature_subset.columns.tolist()]
        feature_list = [list(f) for f in feature_subset.values]
        feature_colon_value_list = []
        for ft in feature_list:
            #feature_colon_value_list.append(self.concat_feature_colon_value(header, ft))
            feature_colon_value_list.append(ft) # no concat
        feature_tuples = list(zip(features.iloc[:,0], feature_colon_value_list))     
        temp = open(f'{self.input_iw_path}', 'w')
        for item_id, fts in feature_tuples:
            for ft in fts:
                print(f'{item_id}\t{ft}\t{1.0}', file=temp)
    
    def fit(self, d=128, ns=5, lr=0.025, update_times=100):
        CMD = f'./Graph_module/codes.tpr.rec/tpr -train_ui {self.input_ui_path} -train_iw {self.input_iw_path} -save /dev/stderr -sample_times {update_times} -dimensions {d} -negative_samples {ns} -init_alpha {lr} -threads {multiprocessing.cpu_count()}'
        proc = subprocess.run(CMD.split(), stderr=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8')
        emb = pd.read_csv(io.StringIO(proc.stderr), header=None, skiprows=1, sep=' ').set_index(0)
        users, items = self.users, self.items
        users_emb, items_emb = emb.reindex(users), emb.reindex(items)
        return users_emb, items_emb

    def id2user(self):
        user_dict = {}
        for i , user in enumerate(self.users):
            user_dict[i] = user
        return user_dict

    def id2item(self):
        item_dict = {}
        for i , item in enumerate(self.items):
            item_dict[i] = item
        return item_dict
