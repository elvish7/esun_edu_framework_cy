import numpy as np
import pandas as pd 
def plus_a_b(a,b):
    return a+b

def repeat(a,b=3):
    return np.repeat(a,b)

def to_dataframe(seq):
    return pd.DataFrame(seq)