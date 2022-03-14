import math, json, fileinput, os, itertools, subprocess, io, sys, collections, fileinput, multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm

class SMORe:
    def __init__(self, data):
        self.data = self.read(data)
        self.input_data = self.interactions()
        self.users = data['cust_no'].unique().tolist()
        self.items = data['wm_prod_code'].unique().tolist()
        self.user_dict = self.id2user()
        self.item_dict = self.id2item()
        
    def read(self, record_df):
        return [(u, i) for u, i in zip(record_df['cust_no'], record_df['wm_prod_code'])]

    def interactions(self):
        data = self.data
        input_data  = io.StringIO() 
        for record, count in collections.Counter(data).items():
            print(*record, count, file=input_data)
        return input_data
        
    def fit(self, mode='warp', d=128, ns=5, lr=0.025, update_times=100):
        CMD = f'./smore/cli/{mode} -train /dev/stdin -save /dev/stderr -sample_times {update_times} -dimensions {d} -negative_samples {ns} -alpha {lr} -threads {multiprocessing.cpu_count()}'
        proc = subprocess.run(CMD.split(), input=self.input_data.getvalue(), stderr=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8')
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
