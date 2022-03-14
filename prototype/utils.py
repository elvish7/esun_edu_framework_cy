"""
This program is using lightfm dataset package, and designed for ligthgm recommendation.
Author: Jia-Huei Ju (jhjoo@citi.sinica.edu.tw)
"""
import pandas as pd
import numpy as np
import datetime
from collections import defaultdict

# Preprocessing function for user/item columns
# (1) filling NA preprocessing
def fill_mean(series):
    return series.mean()

def fill_mode(series):
    return series.mode()[0]

def fill_median(series):
    return series.median(numeric_only=True)

# (2) meta-to-feature preprocessing
def median_binarization(series):
    median = series.median()
    series = series.apply(lambda x: 1 if x>= median else 0)
    return series

def binning(series, bin_size):
    """Binnging function for categorization the numeric data, two types:

    [int] bined into n bins, defined by bin_size
    [float] bined the equal sample size of percentage, defined by quantile.
    """
    if isinstance(bin_size, int):
        value_min, value_max = series.min(), series.max()
        bins = np.linspace(value_min, value_max, num=bin_size+1)

    if isinstance(bin_size, float):
        bins = series.quantile(np.linspace(0, 1, int(1/bin_size) + 1))

    labels = list()
    for i, v in enumerate(bins, 1):
        labels += [str(i-1) + '-' + str(i)]
    return pd.cut(series, bins, labels=labels)

binning_10 = lambda x: binning(x, 10)
quantile_binning_25 = lambda x: binning(x, 0.25)

def identity(series):
    return series

def prod_ccy_categorization(series):
    """ [CONCERN] make it more general or more systematic way.  """
    series = series.apply(lambda x: "OTHERS" if x not in ("USD", "TWD", "EUR") else x)
    return series

def discard_categorization(series, threshold=0):
    preserved_values = [value for value, tf in list((series.value_counts() >= threshold).items()) if tf is True]
    series = series.apply(lambda x: "OTHERS" if x not in preserved_values else x)
    return series

discard_categorization_100 = lambda x: discard_categorization(x, 100)

# others
def concat_feature(df, off_cols):
    """Transform the raw meta and its value to the new feature by concatenating with value, 
        (1) Preprocess the field with feature name.
        (2) Remove the variable without variance
    Args:
        df: dataframe with user or item features
        off_cols: the columns of dataframe, which 
    """
    columns = [col for col in df.columns if col not in off_cols]
    for col in columns:
        df[col] = col + ":" + df[col].astype(str)
        if len(df[col].unique()) == 1:
            df.drop(columns=[col], inplace=True)
    return df


def aggregate_fn(criteria):
    """Function for aggregating re-subscribe items (rating function)
        (1) 'txn_amt:sum-deduct:sum': amount summation - deduct summation
        (2) 'regular_n_counts': add subscribing type (solo or regular) and the corresponding counts.
    """
    if criteria == 'txn_amt:sum-deduct:sum':
        def f(x):
            d = {}
            d['prod_rating'] = x['txn_amt'].sum() - x['deduct_cnt'].sum()
        return pd.Series(d, index=['prod_rating'])
    elif criteria == 'regular-n-counts':
        def f(x):
            d = {}
            d['regular'] = x['deduct_cnt'] >= 0
            d['counts'] = len(x)
        return pd.Series(d, index=['regular', 'counts'])
    else:
        exit(0)
    return f
    
def npmapping(arr, mapping, reverse=False):
    """ Faster mapping function worked with numpy array

    Args:
        arr: the ready-to-transforme object array.
        mapping: the mapping dictionary with {key: value}
        reverse: the reversed mapping dictionary with {value: key} from original mapping key and value.
    Returns:
        tranformed_arr: the mapped value of arr, which preserved the same shape.
    """
    if reverse:
        mapping = {v:k for (k, v) in mapping.items()}
    if ~isinstance(arr, np.ndarray):
        arr = np.array(arr)
    u, inv = np.unique(arr, return_inverse=True)
    arr_mapped = np.array([mapping[x] for x in u])[inv].reshape(arr.shape)
    return arr_mapped
