import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import heapq
import numpy as np

# --- dataset --#

random_seed = 0

def get_data(ratings_data, batch_size=256):
    data = ratings_data
    data = data.values # convert to numpy array
    inps = data[:, 0:2].astype(int) # get user, item inputs
    tgts = data[:, 2].astype(int) # get rating targets
    loaders = []
    # split and convert to tensors
    sidx = np.arange(len(inps), dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(len(inps) * (1. - 0.1)))
    train_index, test_index = sidx[:n_train], sidx[n_train:]

    inps_train = torch.tensor(inps[train_index], dtype=torch.long)
    inps_test = torch.tensor(inps[test_index], dtype=torch.long)
    tgts_train = torch.tensor(tgts[train_index], dtype=torch.long)
    tgts_test = torch.tensor(tgts[test_index], dtype=torch.long)
    # convert to TensorDataset type
    train_set = TensorDataset(inps_train, tgts_train)
    test_set = TensorDataset(inps_test, tgts_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    loaders.append([train_loader, test_loader])
    
    # return all loaders for cross validation
    ## no cross validation
    return loaders   

# --- item_to_genre --#

def item_to_genre(item, data):
    funds = data.iloc[:, 1:]
    genre = funds.loc[item]
    return genre

def get_genre(data_size, data):
    funds = data 
    items = funds.iloc[:, 0].values
    genres = funds.iloc[:, 1:].values
    return (items, genres)

# --- topK --#

def getListMaxNumIndex(num_list,topk=3):
    max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    return set(list(max_num_index)), set(list(min_num_index))

#top bot k match
def topK(a, b, k=5, m=3, num_user=63619):
    results_max = np.zeros(num_user) # 943, 6040
    results_min = np.zeros(num_user)
    for i in range(num_user): # 943
        Max1,Min1 = getListMaxNumIndex(list(a[i]),m)
        Max2,Min2 = getListMaxNumIndex(list(b[i]),k)
        results_max[i] = len(Max1&Max2)/m
        results_min[i] = len(Min1&Min2)/m
    return results_max.mean(), results_min.mean()

# hit ratio @k
def hrK(a, b, k=5, num_user=63619):
    # a = pred40
    # b = pref
    results_max = np.zeros(num_user)
    results_min = np.zeros(num_user)
    for i in range(num_user):
        Max1,Min1 = getListMaxNumIndex(list(a[i]),k)
        Max2,Min2 = getListMaxNumIndex(list(b[i]),1)
        results_max[i] = len(Max1&Max2)
        results_min[i] = len(Min1&Min2)
    return results_max.mean()

def load_emb_weights(user_dict, fund_dict, base_model_data):
    user_bias, item_bias = [], []
    user_weights, item_weights = [], []
    item_repts, user_repts, user_id_map, item_id_map = base_model_data
    # get user pretrained embedding with aligned index
    for i in range(len(user_dict)):
        corr_uid = user_id_map[user_dict[i]]
        user_weights.append(user_repts[1][corr_uid])
        user_bias.append(user_repts[0][corr_uid])

    # get item pretrained embedding with aligned index
    for j in range(len(fund_dict)):
        corr_iid = item_id_map[fund_dict[j]]
        item_weights.append(item_repts[1][corr_iid])
        item_bias.append(item_repts[0][corr_iid])
    
    user_weights = torch.FloatTensor(np.array(user_weights))
    item_weights = torch.FloatTensor(np.array(item_weights))
    user_bias, item_bias = torch.FloatTensor(np.array(user_bias)), torch.FloatTensor(np.array(item_bias))


    return [user_weights, item_weights, user_bias, item_bias]
