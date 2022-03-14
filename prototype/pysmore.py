import io, os, sys
import math, json
import fileinput, collections
import subprocess, multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class SMORe:
    def __init__(self, 
                 dimension_size=16,
                 loss='warp',
                 negative_samples=5,
                 n_user=None,
                 n_item=None,
                 input_file='temp/networks.txt',
                 output_file='temp/reps.txt'):

        self.user_embeddings = None
        self.item_embeddings = None
        self.dimension_size=dimension_size
        self.mode = loss
        self.negative_samples=negative_samples
        self.networks = input_file
        self.representations = output_file
        self.n_user = n_user
        self.n_item = n_item

    def fit(self, update_times=10, cold_start='mean'):
        CMD = f'./smore/cli/{self.mode}'
        CMD += f' -train {self.networks}'
        CMD += f' -save {self.representations}'
        CMD += f' -sample_times {update_times}'
        CMD += f' -dimensions {self.dimension_size}'
        CMD += f' -negative_samples {self.negative_samples}'
        CMD += f' -alpha 0.025'
        CMD += f' -threads {multiprocessing.cpu_count()}'

        proc = subprocess.run(
                CMD.split(), 
                stderr=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                encoding='utf-8'
        )

        df_embeddings = pd.read_csv(
                self.representations,
                header=None, 
                skiprows=1, 
                sep=' '
        ).set_index(0)

        user_embeddings = df_embeddings.reindex(
                [f"USER_{i}" for i in range(self.n_user)]
        ).values
        item_embeddings = df_embeddings.reindex(
                [f"ITEM_{i}" for i in range(self.n_item)]
        ).values

        cs_user = (np.isnan(user_embeddings).sum(axis=1) == self.dimension_size)
        cs_item = (np.isnan(item_embeddings).sum(axis=1) == self.dimension_size)

        print(f"- cold-start users :{sum(cs_user)}")
        print(f"- cold-start items :{sum(cs_item)}")

        # FillNA post-processing
        if cold_start == 'mean':
            mean_user_embedding = np.nanmean(user_embeddings, axis=0)
            mean_item_embedding = np.nanmean(item_embeddings, axis=0)
            user_embeddings[cs_user, :] = 0
            item_embeddings[cs_item, :] = 0
            # user_embeddings[cs_user, :] = np.tile(mean_user_embedding, 
            #                                       sum(cs_user)).reshape(-1, self.dimension_size)
            # item_embeddings[cs_item, :] = np.tile(mean_item_embedding, 
            #                                       sum(cs_item)).reshape(-1, self.dimension_size)

        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

        return 0

    def predict(self):

        scores = cosine_similarity(
                self.user_embeddings, 
                self.item_embeddings
        )

        return scores


