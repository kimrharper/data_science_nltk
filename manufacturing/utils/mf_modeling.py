# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn import preprocessing
from scipy import stats

# Import Files
mf_num_data = pd.read_csv('bosch_big_data/train_numeric.csv',low_memory=False)
mf_date_data = pd.read_csv('bosch_big_data/train_date.csv',low_memory=False)
mf_num_test = pd.read_csv('bosch_big_data/test_numeric.csv',low_memory=False)
mf_date_test = pd.read_csv('bosch_big_data/test_date.csv',low_memory=False)

mf_test_merged = mf_num_test.append(mf_date_test)


# process X values with imputation if needed
def process_data(df, imbalance):
    df = df.fillna(df.mean())
    df = df.dropna(axis=1, how='all')
    y = df.iloc[:,-1]
    X = df.iloc[:,:-1]
    Xtr,ytr, = X, y
    
    test_data = mf_test_merged.fillna(df.mean())
    test_data = test_data.dropna(axis=1, how='all')
    yte = test_data.iloc[:,-1]
    Xte = test_data.iloc[:,:-1]

    Xtr, Xte = Xtr.fillna(Xtr.mean()), Xte.fillna(Xte.mean()) 
    
    if imbalance:
        Xtr['Response'] = ytr
        Xtr = fix_imbalance(Xtr, 2)
        ytr = Xtr['Response']
        del Xtr['Response']
        Xtr = Xtr
        
    return Xtr.values, Xte.values, ytr.values, yte.values