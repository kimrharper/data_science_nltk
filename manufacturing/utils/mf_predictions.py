# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn import preprocessing
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn import tree

from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA, NMF
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVR
from sklearn.metrics import make_scorer

# Import Files
mf_num_data = pd.read_csv('bosch_small_data/train_numeric.csv',low_memory=False)
print('train_numeric.csv loaded...')
mf_date_data = pd.read_csv('bosch_small_data/train_date.csv',low_memory=False)
print('train_date.csv loaded...')
mf_num_data_test = pd.read_csv('bosch_small_data/test_numeric.csv',low_memory=False)
print('test_numeric.csv loaded...')
mf_date_data_test = pd.read_csv('bosch_small_data/test_date.csv',low_memory=False)
print('test_date.csv loaded...')

# settings
np.seterr(divide='warn', invalid='warn'); sns.set_style("whitegrid");warnings.filterwarnings('ignore')

# Declarations for functions
ms_mcc = make_scorer(matthews_corrcoef)
TEST_PER = .3

# Duplicate imbalanced values (ideally in just the training set)
def fix_imbalance(df, dup_count):
    df = df.append([df[df['Response'] == 1]]*dup_count,ignore_index=True)
    return df

# merge_dfs on response and id
def merge_dfs(a,b,target):
    if target:
        merged = pd.merge(a, b, on=['Id','Response'])
        merged = merged[[c for c in merged if c not in ['Response']] + ['Response']]
    else:
        merged = pd.merge(a, b, on='Id')
    return merged


# visualize data by components
def visualize_data(pipeline,dimred):
    feature_plot = list(zip(features, pipeline.named_steps[dimred].components_[0]))
    feature_plot = pd.DataFrame(data=feature_plot)
    feature_plot = pd.DataFrame(feature_plot.sort_values(1, ascending=False).iloc[0:10])
    plt.figure(figsize=(20,5))
    plt.title('Ordered by variance')
    sns.barplot(x=0, y=1, data=feature_plot, palette=sns.color_palette("cool"))
    plt.ylim(feature_plot[1].min(),feature_plot[1].max())
    plt.savefig(dimred+'-top10.png')
    plt.show()
     
    plt.figure(figsize=(20,8))
    plt.title('Component Variance')
    plt.plot(pipeline.named_steps[dimred].explained_variance_ratio_)

# Get the last recorded time
def final_time(df,row_ind):
    time = df.iloc[row_ind,1:].dropna().iloc[-2]
    response = df.iloc[row_ind,-1]
    return time,response

def display_results(gridsearch, param_list, file_name):
    grid_results = pd.DataFrame(gridsearch.cv_results_)
    columns = param_list
    grid_results = grid_results[columns]
    grid_results['param_model']=grid_results['param_model'].apply(lambda val: str(val).split('(')[0])
    grid_results = grid_results.sort_values('rank_test_score')
    grid_results.to_html(file_name)
    return grid_results