import pandas as pd
import scipy
import time
import random
import numpy as np
import os
from sklearn import preprocessing

def convert_data(w103, w106, aspect_col):
    """
    convert original dataset to AMCF format
    Input: 
        w103_df, w106_df
    Output:
        rating, item aspect features(fund)
    """

    total_amt = w103.groupby('cust_no').sum()['txn_amt'].to_dict()
    ratings = w103[['cust_no', 'wm_prod_code', 'txn_amt', 'txn_dt']].dropna()
    ratings['txn_dt'] = pd.to_datetime(
        ratings['txn_dt'], format="%Y-%m-%d").astype(int) / 10**9
    # deal with duplicate funds brought by the same user
    ratings = ratings.groupby(['cust_no', 'wm_prod_code'], as_index=False).agg(
        {'txn_amt': 'sum', 'txn_dt': 'mean'})
    # calculate txn_amt/total txn_amt
    ratings['txn_amt'] = [amt/total_amt[i] for i, amt in zip(ratings['cust_no'], ratings['txn_amt'])]
    ratings['txn_amt'] = pd.cut(ratings.txn_amt, bins=5, labels=np.arange(1, 6), right=False).astype(int)

    # encode to index
    le1 = preprocessing.LabelEncoder()
    ratings['cust_no'] = le1.fit_transform(ratings['cust_no'])
    user_dict = dict(zip(le1.transform(le1.classes_), le1.classes_))

    le2 = preprocessing.LabelEncoder()
    ratings['wm_prod_code'] = le2.fit_transform(ratings['wm_prod_code'])
    fund_dict = dict(zip(le2.transform(le2.classes_), le2.classes_))
    fund_label_id = dict(zip(le2.classes_, le2.transform(le2.classes_)))

    ratings.rename({'cust_no': 'uid', 'wm_prod_code': 'fid',
                   'txn_amt': 'rating', 'txn_dt': 'timestamp'}, axis=1, inplace=True)
    ratings = ratings.sort_values(by=['uid'], axis=0).reset_index(drop=True)

    # negative sampling
    neg_samples = neg_sampling(ratings)
    neg_samples.rename({'interact': 'rating'}, axis=1, inplace=True)
    # ratings = neg_samples # interactioin (0 or 1)
    ratings = pd.concat([ratings, neg_samples])
    # print(ratings[ratings['uid']==1])

    fund = w106.join(pd.get_dummies(w106[aspect_col])).drop(aspect_col, axis=1)
    fund['wm_prod_code'] = [fund_label_id[i] for i in fund['wm_prod_code']]
    fund.rename({'wm_prod_code': 'fid'}, axis=1, inplace=True)
    fund = fund.sort_values(by=['fid'], axis=0).reset_index(drop=True)

    user_n, item_n = len(user_dict), len(fund_dict)

    return ratings, fund, user_n, item_n, user_dict, fund_dict


def neg_sampling(ratings_df, n_neg=1, neg_val=0, pos_val=1, percent_print=50):
    """version 1.2: 1 positive 1 neg (2 times bigger than the original dataset by default)
        Parameters:
        input rating data as pandas dataframe: uid|fid|rating
        n_neg: take n_negative / 1 positive
        Returns:
        negative sampled set as pandas dataframe
                uid|fid|interact (implicit)
    """
    ratings_df.uid = ratings_df.uid.astype('category').cat.codes.values
    ratings_df.fid = ratings_df.fid.astype('category').cat.codes.values
    sparse_mat = scipy.sparse.coo_matrix(
        (ratings_df.rating, (ratings_df.uid, ratings_df.fid)))
    dense_mat = np.asarray(sparse_mat.todense())
    print(dense_mat.shape)
    nsamples = ratings_df[['uid', 'fid']].copy()
    nsamples['interact'] = nsamples.apply(lambda row: 1, axis=1)
    length = dense_mat.shape[0]
    printpc = int(length * percent_print/100)
    nTempData = []
    i = 0
    start_time = time.time()
    stop_time = time.time()
    extra_samples = 0
    for row in dense_mat:
        if(i % printpc == 0):
            stop_time = time.time()
            print("processed ... {0:0.2f}% ...{1:0.2f}secs".format(
                float(i)*100 / length, stop_time - start_time))
            start_time = stop_time
            n_non_0 = len(np.nonzero(row)[0])
            zero_indices = np.where(row == 0)[0]
        if(n_non_0 * n_neg + extra_samples > len(zero_indices)):
            print(i, "non 0:", n_non_0, ": len ", len(zero_indices))
            neg_indices = zero_indices.tolist()
            extra_samples = n_non_0 * n_neg + extra_samples - len(zero_indices)
        else:
            neg_indices = random.sample(
                zero_indices.tolist(), n_non_0 * n_neg + extra_samples)
            extra_samples = 0
            nTempData.extend([(uu, ii, rr) for (uu, ii, rr) in zip(np.repeat(
                i, len(neg_indices)), neg_indices, np.repeat(neg_val, len(neg_indices)))])
            i += 1
    nsamples = nsamples.append(pd.DataFrame(
        nTempData, columns=["uid", "fid", "interact"]), ignore_index=True)

    # return the generated negative samples
    return nsamples[nsamples['interact']==0]
