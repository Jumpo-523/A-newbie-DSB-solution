"""
DSB 2019の
前処理→訓練→推論の（手続き型）実行プログラム


"""

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
import yaml

import sys

sys.path.append('/Users/junpei.takubo/Downloads/kaggle_dsb/')

from commons import *
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prepared', action='store_true')
#     parser.add_argument('--prepared', type=bool, nargs='?', default=False,
#                    help='if get_data() was executed, we can skip this process by setting this var True')
    args = parser.parse_args()
    with open(config_path, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    # read data
    train, test, train_labels, specs, sample_submission = read_data()
    
    # Label-Encoding preprocessing
    train, test, train_labels, maps_data = encode_title(train, test, train_labels)
    
    t_levels = Title_levels()
    t_levels.set_titlesLevel()
    
    exit_completely_list = specs.loc[specs["info"].str.contains("The exit game event") , "event_id"].to_list()

    maps_data.update({"exit_completely_list":exit_completely_list})
    constants = Constants(maps_data, train_labels, t_levels)
    classification_cls = eda_event_data(specs)
    [classification_cls.label_event_id(st) for st in constants.eventIdCategorizer]
    

    # tranform function to get the train and test set
    categoricals = ['session_title','last_Clip', 'last_Activity', 'last_Assessment', 'last_Game']
    reduce_path = '../data-science-bowl-2019/features/'

    base_path = '/Users/junpei.takubo/Downloads/kaggle_dsb/'

    if not args.prepared:
        reduce_train, reduce_test = get_train_and_test(train, test, constants)
        reduce_train.to_csv(base_path + "data-science-bowl-2019/features/reduced_train.csv", index=False)
        reduce_test.to_csv(base_path + "data-science-bowl-2019/features/reduced_test.csv", index=False)
    reduce_train = pd.read_csv(base_path + "data-science-bowl-2019/features/reduced_train.csv")
    reduce_test = pd.read_csv(base_path + "data-science-bowl-2019/features/reduced_test.csv")
    reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test, constants)


    cols_to_drop = ['game_session', 'accuracy', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate', 'num_incorrect','num_correct']
    features = [col for col in reduce_train.columns if col not in cols_to_drop and col not in categoricals]

    target_variable = 'accuracy_group'
    
    # set target variable.
    y = reduce_train[target_variable]

    useful_columns = []

    # XXX: 特徴量選択：correlationの高い変数のうち片方を除外すると言う悪手をしています。
    to_exclude, ajusted_test = exclude(reduce_train, reduce_test, features)
    features = [col for col in features if col not in to_exclude]

    for c in categoricals:
        for df in reduce_train, reduce_test:
            df[c] = df[c].astype('category')
    
    y_pred, oof_pred, importances, scores = run_lgb_wrapper(reduce_train,
                             reduce_test, cols_to_drop,
                            categoricals, config["params"], 
                               n_seeds=10)

    pd.concat(importances, axis=0).to_csv(f'importance_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)








    



