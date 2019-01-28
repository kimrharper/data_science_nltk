# imports
import pandas as pd; import missingno as msno; import matplotlib.pyplot as plt; import seaborn as sns;import numpy as np;import warnings
from sklearn import preprocessing; from scipy import stats
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest
from scipy.stats import ttest_ind, ks_2samp
from scipy.stats import describe

# settings
np.seterr(divide='warn', invalid='warn'); sns.set_style("whitegrid");warnings.filterwarnings('ignore')

# Import Files
mf_num_data = pd.read_csv('bosch_small_data/train_numeric.csv',low_memory=False)
mf_cat_data = pd.read_csv('bosch_small_data/train_cat.csv',low_memory=False)
mf_date_data = pd.read_csv('bosch_small_data/train_date.csv',low_memory=False)

# Function for determining params of sample distribution
def distribution_assignment(sample):
    k2, p = normaltest(sample, nan_policy='omit')
    alpha = 0.00001  # null hypothesis: Sample comes from a normal distribution
    dist = 'Normally Distributed' if p > alpha else 'Not Normally Distributed'
    return dist

def sample_test(df,c, test, p_threshold):
    c = df.columns[c]
    a,b = df[c][df['Response']==0].dropna(), df[c][df['Response']==1].dropna()
    d,p = test(a,b)
    s = 'Different' if p < p_threshold else 'Same'
    return c,s,p,d

# Function for different viz of distributions
def plot_dist(df, col_index, transformed):
    c = df.columns[col_index]
    normed = distribution_assignment(df[c])
    print(describe(df[c]))
    s_3 = df[c][np.abs(df[c]-df[c].mean()) <= (3*df[c].std())] # Keep inner 99.7 % of the Data
    s_1 = df[c][np.abs(df[c]-df[c].mean()) <= (1*df[c].std())] # Keep inner 68% of the Data
    transformed = transformed[~np.isnan(transformed)]          # Show the complete transformation of mf_num_data
    
    plt.figure(figsize=(20,1))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    
    plt.subplot(1, 2, 1)
    plt.title('Feature: {}\nDistribution: {}\n\nOriginal\nSample Count: {}'.format(c,normed,len(df[c].dropna())))
    sns.distplot(df[c].dropna(),color='blue')
    plt.xlabel('')

    s_success = df[c][df['Response']==0].dropna()
    s_failure = df[c][df['Response']==1].dropna()
    sampling_results = sample_test(df,col_index, ks_2samp,.1)
    
    plt.subplot(1, 2, 2)
    sns.distplot(s_success)
    plt.xlabel('')
    
    plt.subplot(1, 2, 2)
    plt.title('Success/Failure\n\n{}\nSuccess Count: {}\nFailure Count: {}'.format(sampling_results,len(s_success),len(s_failure)))
    sns.distplot(s_failure,color='purple')
    plt.xlabel('')
    plt.legend(['Success','Failure'])
    
pt = PowerTransformer()