# imports
import pandas as pd; import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn import preprocessing
from scipy import stats
import networkx as nx
from IPython.display import Image
from networkx.drawing.nx_agraph import graphviz_layout,from_agraph, to_agraph; import graphviz
from sklearn.preprocessing import Normalizer
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable


# settings
np.seterr(divide='warn', invalid='warn')
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

# debug
import_fail=False
exec("try: import pygraphviz as pgv\nexcept: problem=import_fail = True")
# exec("try: print('Modules Loaded - Settings Loaded')\nexcept: problem=import_fail = True")


# Import Files
mf_num_data = pd.read_csv('bosch_small_data/train_numeric.csv',low_memory=False)
print('train_numeric.csv loaded...')
mf_date_data = pd.read_csv('bosch_small_data/train_date.csv',low_memory=False)
print('train_date.csv loaded...')
mf_cat_data = pd.read_csv('bosch_small_data/train_cat.csv',low_memory=False)
print('train_cat.csv loaded...')


# Numerical Data Functions
def get_ratio(column):
    all_response = mf_num_data[[column,'Response']].dropna()['Response'] # Get response values as series
    return [column, all_response.sum()/len(all_response)]                # (failure values sum) / (total)

# Graph Data Functions 
# [source: https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order]
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def create_adjacency_matrix(df):
    fail_count_station = []      # success fail count
    row_count = len(df)          # get length of the total number of rows
    network_structure = []       # build the network by using column names

    for i in range(row_count):
        feature_group = remove_duplicates(['_'.join(feature_member.split('_')[0:2]) for feature_member in list(df.iloc[i,1:-1].dropna().index)])
        network_structure.append(feature_group)

    # create adjacency.txt for easy networkx import
    file = open('utils/mf_exploratory_adjaceny.txt','w') 
    for l in network_structure:
        if len(l) > 1:
            file.write(' '.join(l)+'\n')
    file.close()
    
def network_failure_rate(df):
    station_fail_ratio = {}
    station_name, station_count=[],[]
    station_feature_group = {}
    row_count = len(df)
    new_sets = [set(['_'.join(feature_member.split('_')[0:2]) for feature_member in list(df.iloc[i,1:-1].dropna().index)]) for i in range(row_count)]
    stations_ordered = frozenset().union(*new_sets) # method for keeping the order in place for nxviz

    for i in stations_ordered:
        total_count = 0
        fail_count = 0
        station_subset = []
        for m in df.columns[1:-1]:
            if i in m:
                df_temp = df[np.isfinite(df[m])]
                fail_count+=df_temp['Response'][df_temp['Response']==1].sum()
                total_count+=len(df_temp['Response'])
                station_subset.append(m)
        station_feature_group[i] = station_subset
        station_count.append(fail_count/total_count)
        station_name.append(i)
    station_count = np.array(station_count)
    
    return station_count, station_feature_group, station_name

def remove_outlier(arr): # function for removing a single max value outlier (special case)
    l = list(arr)
    l.remove(max(l))
    return min(l),max(l)

# Time Series Functions
def final_time(df,row_ind):
    time = df.iloc[row_ind,1:].dropna().iloc[-2]
    response = df.iloc[row_ind,-1]
    return time,response

def plot_time_series(kde):
    # Iterate through the columns to get the last time value
    last_time =[]
    for i in range(len(mf_date_data.iloc[:,1:-1])):
        try:
            lt, sc = final_time(mf_date_data,i)
            last_time.append(lt)
        except:
            last_time.append(0)
    mf_date_data['final_time'] = last_time
    
    time_series_plot = mf_date_data[['final_time','Response']] # Append the 'last time' series as a new column


    fig = plt.figure(figsize=(20,7))
    g = sns.distplot(time_series_plot['final_time'][time_series_plot['Response']==0], kde=kde)
    g = sns.distplot(time_series_plot['final_time'][time_series_plot['Response']==1], kde=kde ,color='purple')
    fig = plt.legend(['Success','Failure'])