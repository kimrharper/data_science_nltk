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
mf_date_data = pd.read_csv('bosch_small_data/train_date.csv',low_memory=False)
# mf_cat_data = pd.read_csv('bosch_small_data/train_cat.csv',low_memory=False)

# settings
np.seterr(divide='warn', invalid='warn'); sns.set_style("whitegrid");warnings.filterwarnings('ignore')