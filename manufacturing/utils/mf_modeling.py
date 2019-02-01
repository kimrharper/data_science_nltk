# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn import preprocessing
from scipy import stats
import dask.dataframe as dd
from dask.distributed import Client

# Import Files
mf_num_data = dd.read_csv('bosch_big_data/train_numeric.csv',low_memory=False)
mf_date_data = dd.read_csv('bosch_big_data/train_date.csv',low_memory=False)
mf_num_test_data = dd.read_csv('bosch_big_data/test_numeric.csv',low_memory=False)
mf_date_Test_data = dd.read_csv('bosch_big_data/test_date.csv',low_memory=False)

mf_merged_train = mf_num_data.append(mf_date_data)
mf_merged_test = mf_num_test_data.append(mf_date_Test_data)

def process_data(dd):
    dd = dd.fillna(dd.mean())
    dd = dd.dropna(how='all')
    y = dd.iloc[:,-1]
    X = dd.iloc[:,:-1]
    Xtr,ytr, = X, y
    
    test_data = mf_merged_test.fillna(mf_merged_test.mean())
    test_data = test_data.dropna(how='all')
    Xte = test_data.iloc[:,:-1]

    Xtr, Xte = Xtr.fillna(Xtr.mean()), Xte.fillna(Xte.mean()) 

        
    return Xtr.values, Xte.values, ytr.values