#! python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
random.seed(1029)
np.random.seed(1029)
import os
import copy
import matplotlib.pyplot as plt
# %matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15
from collections import defaultdict
import lightgbm as lgb
# import xgboost as xgb
import catboost as cat
import time
from collections import Counter
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model
import gc
import seaborn as sns
import warnings
# warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization
import eli5
import shap
from IPython.display import HTML
import json
import altair as alt
from category_encoders.ordinal import OrdinalEncoder
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

import os, time, datetime, json, gc, scipy as sp, seaborn as sns
from numba import jit
from functools import partial
from tqdm import tqdm_notebook

import lightgbm as lgb
# import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from typing import Any
from itertools import product
pd.set_option('max_rows', 500)
import re
from tqdm import tqdm
from joblib import Parallel, delayed


import sys

sys.path.append('/Users/junpei.takubo/Downloads/kaggle_dsb/')

from commons import *
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prepared', type=bool, nargs='?', default=False,
                   help='if get_data() was executed, we can skip this process by setting this var True')
    args = parser.parse_args()
    
    # read data

    train, test, train_labels, specs, sample_submission = read_data()
    # get usefull dict with maping encode
    train, test, train_labels, maps_data = encode_title(train, test, train_labels)


    constants = Constants(maps_data)
    classification_cls = eda_event_data(specs)
    [classification_cls.label_event_id(st) for st in constants.eventIdCategorizer]
    # tranform function to get the train and test set
    categoricals = ['session_title']
    reduce_path = '../data-science-bowl-2019/features/'
    # import pdb;pdb.set_trace()
    base_path = '/Users/junpei.takubo/Downloads/kaggle_dsb/'
    if not args.prepared:
        reduce_train, reduce_test = get_train_and_test(train, test, constants)
        reduce_train.to_csv(base_path + "data-science-bowl-2019/features/reduced_train.csv", index=False)
        reduce_test.to_csv(base_path + "data-science-bowl-2019/features/reduced_test.csv", index=False)
    reduce_train = pd.read_csv(base_path + "data-science-bowl-2019/features/reduced_train.csv")
    reduce_test = pd.read_csv(base_path + "data-science-bowl-2019/features/reduced_test.csv")
    reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test, constants)

    # titles_dict_indexed / 
    y = reduce_train['accuracy_group']

    cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
    params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'seed': 42,
    'early_stopping_rounds': 100,
    'n_estimators': 1000000,
    'learning_rate': 0.04,
    'num_leaves':2**8,
    'max_depth': 15,
    'lambda_l1': 0.7,
    'lambda_l2': 0.5,
    'bagging_fraction': 0.9,
    'verbose': 100
    }
    params =  {'num_leaves': 61,  # 当前base 61
           'min_child_weight': 0.03454472573214212,
           'feature_fraction': 0.3797454081646243,
           'bagging_fraction': 0.4181193142567742,
           'min_data_in_leaf': 96,  # 当前base 106
           'objective': 'regression',
           "metric": 'rmse',
           'max_depth': -1,
           'learning_rate': 0.1,   # 快速验证
    #      'learning_rate': 0.006883242363721497,
           "boosting_type": "gbdt",
           "bagging_seed": 11,
           "verbosity": -1,
           'reg_alpha': 0.3899927210061127,
           'reg_lambda': 0.6485237330340494,
           'random_state': 47,
           'num_threads': 16,
           'lambda_l1': 1,  
           'lambda_l2': 1
    #      'is_unbalance':True
             }
    category_cals = ['session_title']
    y_pred, oof_pred, importances = run_lgb(reduce_train, reduce_test, cols_to_drop, category_cals, params)
    # reduce_train['1st_stacking'] = oof_pred
    # reduce_test['1st_stacking'] = y_pred
    # import pdb; pdb.set_trace()
    run_lgb_wrapper(reduce_train, reduce_test, cols_to_drop, category_cals, params)
    # y_pred, oof_pred = run_lgb(reduce_train, reduce_test, cols_to_drop, category_cals, params)
    # 1st 0.5613486386645519
    # seed averagingすごい聞くことがわかった。 12/25








    




