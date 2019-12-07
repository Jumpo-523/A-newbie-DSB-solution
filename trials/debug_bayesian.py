
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

import os
import time
import datetime
import json
import gc
from numba import jit



from functools import partial
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
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

# import pdb; pdb.set_trace()
# from commons.devtools import say_notify
# say_notify("hogehoge")

titles_dict = {'Dino Drink':{"good":'"correct":true',"bad":'"correct":false'},
'Watering Hole (Activity)':{"good":'"filled":true', "bad":'"filled":false'},
'All Star Sorting':{"good":'"correct":true',"bad":'"correct":false'},
'Air Show':{"good":'"correct":true',"bad":'"correct":false'},
'Crystals Rule':{"good":'"correct":true',"bad":'"correct":false'},
'Bubble Bath':{"good":'"correct":true',"bad":'"correct":false'},
'Bottle Filler (Activity)':{"good":["wowSoCool","niceJob","ohWow"]},
'Dino Dive':{"good":'"correct":true',"bad":'"correct":false'},
'Happy Camel':{"good":'"correct":true',"bad":'"correct":false'},
'Pan Balance':{"good":'"correct":true',"bad":'"correct":false'},
'Egg Dropper (Activity)':{"bad":"Buddy_EggsWentToOtherNest", "good":"Buddy_Incoming"},
'Leaf Leader':{"good":'"correct":true',"bad":'"correct":false'},
'Sandcastle Builder (Activity)':{"good":["So cool!", 'Great job! You did it!'],"bad":'need'},
'Scrub-A-Dub':{"good":'"correct":true',"bad":'"correct":false'},
'Chow Time':{"good":'"correct":true',"bad":'"correct":false'},
# not yet to be doneつか完全に遊ぶやつ
'Fireworks (Activity)':4000,
'Flower Waterer (Activity)':4000,
'Bug Measurer (Activity)':4000,
 'Chicken Balancer (Activity)':4000}# measure使ったかいなかで測ろう．

def each_game_and_activity_score(titles_dict, session):
    session_title = session['title'].iloc[0]
    if isinstance(titles_dict[session_title], dict):
        # find str
        result = dict(good=0, bad=0)
        good_and_bad_dict = titles_dict[session_title]
        for key in ["good", "bad"]:
            if good_and_bad_dict.get(key) and isinstance(good_and_bad_dict[key], str):
                result[key] = session.event_data.str.contains(good_and_bad_dict[key]).sum()
            elif good_and_bad_dict.get(key):
                search_word = r"|".join(good_and_bad_dict[key])
                counts = session.event_data.str.contains(search_word).sum()
                # print(counts)
                result[key] = counts    
        summand = result["good"] + result["bad"]
        return result["good"]/summand if summand >0 else -1 
    elif titles_dict[session_title] == 4000:
        # just a pure activity so that I want to count how many users tap or enjoy the one. 
        return session.event_data.str.contains(r'"event_code":4\d{3}').sum()

def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('../data-science-bowl-2019/originals/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('../data-science-bowl-2019/originals/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('../data-science-bowl-2019/originals/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('../data-science-bowl-2019/originals/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('../data-science-bowl-2019/originals/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels, titiles_dict):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # indexEncoding to titles_dict
    origin_GameAndAct_titles = list(titles_dict.keys()).copy()
    for key in origin_GameAndAct_titles:
        titles_dict[activities_map[key]] = titles_dict.pop(key)

    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, \
                list_of_event_code, activities_labels, assess_titles,\
                list_of_event_id, all_title_event_code, titles_dict



def get_data(user_sample, titles_dict, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    
    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()}
    
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
    # calculate the last score of each activity
    gameActivityScores = {'score_title_' + str(ga_title): 0 for ga_title in titles_dict.keys()}
    


    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
        # import pdb;pdb.set_trace()
        if gameActivityScores.get('score_title_' + str(session_title))==0:
            # import pdb;pdb.set_trace()
            score_ = each_game_and_activity_score(titles_dict, session)
            # print(score_)
            gameActivityScores['score_title_' + str(session_title)] = score_
            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            # features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())
            
            features.update(gameActivityScores.copy())
            
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
            
        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        # title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type 
                        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def get_train_and_test(train, test, titles_dict):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample, titles_dict)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        compiled_train += get_data(user_sample, titles_dict)
        test_data = get_data(user_sample, titles_dict,test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    return reduce_train, reduce_test


def preprocess(reduce_train, reduce_test):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        event_codes = [str(ec) for ec in [2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
                                        2040, 4090, 4220, 4095]]
        # df['sum_event_code_count'] = df[[str(ec) for ec in event_codes ]].sum(axis = 1)
        
        # df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')
        df.drop(columns=event_codes, inplace=True)
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]
    return reduce_train, reduce_test, features

def LGB_bayesian(max_depth,
                 lambda_l1,
                 lambda_l2,
                 bagging_fraction,
                 bagging_freq,
                 colsample_bytree,
                 learning_rate):
    
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'eval_metric': 'cappa',
        'n_jobs': -1,
        'seed': 42,
        'early_stopping_rounds': 100,
        'n_estimators': 2000,
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': int(bagging_freq),
        'colsample_bytree': colsample_bytree,
        'verbose': 0
    }
    
    mt = MainTransformer()
    ft = FeatureTransformer()
    transformers = {'ft': ft}
    model = RegressorModel(model_wrapper=LGBWrapper_regr())
    model.fit(X=reduce_train, 
              y=y, 
              folds=folds, 
              params=params, 
              preprocesser=mt, 
              transformers=transformers,
              eval_metric='cappa', 
              cols_to_drop=cols_to_drop,
              plot=False)
    
    return model.scores['valid']

if __name__ == "__main__":
    # read data

    train, test, train_labels, specs, sample_submission = read_data()
    # get usefull dict with maping encode
    train, test, train_labels, win_code, list_of_user_activities,\
                list_of_event_code, activities_labels, assess_titles,\
                list_of_event_id, all_title_event_code, titles_dict = encode_title(train, test, train_labels, titles_dict)
    # tranform function to get the train and test set
    categoricals = ['session_title']
    reduce_path = '../data-science-bowl-2019/features/'
    # import pdb;pdb.set_trace()
    if 'reduce_train.csv' in os.listdir(reduce_path):
        
        reduce_train = pd.read_csv(reduce_path + 'reduce_train.csv')
        reduce_test =  pd.read_csv(reduce_path + 'reduce_test.csv')
    else:
        reduce_train, reduce_test = get_train_and_test(train, test, titles_dict)    
        reduce_train.to_csv(reduce_path+'reduce_train.csv', index=False)
        reduce_test.to_csv(reduce_path+'reduce_test.csv', index=False)

    from pipelines import *
    # reduce_train, reduce_test = get_train_and_test(train, test, titles_dict)    
    # reduce_train.to_csv(reduce_path+'reduce_train.csv', index=False)
    # reduce_test.to_csv(reduce_path+'reduce_test.csv', index=False)

    # import pdb; pdb.set_trace()
    # call feature engineering function
    reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)
    y = reduce_train['accuracy_group']

    cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
            
    n_fold = 5
    folds = GroupKFold(n_splits=n_fold)
    gc.collect()
    init_points = 16
    n_iter = 16
    bounds_LGB = {
    'max_depth': (8, 11),
    'lambda_l1': (0, 5),
    'lambda_l2': (0, 5),
    'bagging_fraction': (0.4, 0.6),
    'bagging_freq': (1, 10),
    'colsample_bytree': (0.4, 0.6),
    'learning_rate': (0.05, 0.1)
    }
    #2^max_depth > num_leaves

    LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1029)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
    
    params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'eval_metric': 'cappa',
    'n_jobs': -1,
    'seed': 42,
    'early_stopping_rounds': 100,
    'n_estimators': 2000,
    'learning_rate': LGB_BO.max['params']['learning_rate'],
    'max_depth': int(LGB_BO.max['params']['max_depth']),
    'lambda_l1': LGB_BO.max['params']['lambda_l1'],
    'lambda_l2': LGB_BO.max['params']['lambda_l2'],
    'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
    'bagging_freq': int(LGB_BO.max['params']['bagging_freq']),
    'colsample_bytree': LGB_BO.max['params']['colsample_bytree'],
    'verbose': 100
    }

    mt = MainTransformer()
    ft = FeatureTransformer()
    transformers = {'ft': ft}
    regressor_model = RegressorModel(model_wrapper=LGBWrapper_regr())
    regressor_model.fit(X=reduce_train, 
                        y=y, 
                        folds=folds, 
                        params=params, 
                        preprocesser=mt, 
                        transformers=transformers,
                        eval_metric='cappa', 
                        cols_to_drop=cols_to_drop)

    preds_train_1 = regressor_model.predict(reduce_train)
    preds_1 = regressor_model.predict(reduce_test)
    w_1 = LGB_BO.max['target']
    del bounds_LGB, LGB_BO, params, mt, ft, transformers, regressor_model
    gc.collect()
    bounds_LGB = {
    'max_depth': (11, 14),
    'lambda_l1': (0, 10),
    'lambda_l2': (0, 10),
    'bagging_fraction': (0.7, 1),
    'bagging_freq': (1, 10),
    'colsample_bytree': (0.7, 1),
    'learning_rate': (0.08, 0.2)
    }

    LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1030)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
    params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'eval_metric': 'cappa',
    'n_jobs': -1,
    'seed': 42,
    'early_stopping_rounds': 100,
    'n_estimators': 2000,
    'learning_rate': LGB_BO.max['params']['learning_rate'],
    'max_depth': int(LGB_BO.max['params']['max_depth']),
    'lambda_l1': LGB_BO.max['params']['lambda_l1'],
    'lambda_l2': LGB_BO.max['params']['lambda_l2'],
    'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
    'bagging_freq': int(LGB_BO.max['params']['bagging_freq']),
    'colsample_bytree': LGB_BO.max['params']['colsample_bytree'],
    'verbose': 100
}

    mt = MainTransformer()
    ft = FeatureTransformer()
    transformers = {'ft': ft}
    regressor_model = RegressorModel(model_wrapper=LGBWrapper_regr())
    regressor_model.fit(X=reduce_train, 
                        y=y, 
                        folds=folds, 
                        params=params, 
                        preprocesser=mt, 
                        transformers=transformers,
                        eval_metric='cappa', 
                        cols_to_drop=cols_to_drop)

    preds_train_2 = regressor_model.predict(reduce_train)
    preds_2 = regressor_model.predict(reduce_test)
    w_2 = LGB_BO.max['target']
    del bounds_LGB, LGB_BO, params, mt, ft, transformers, regressor_model
    gc.collect()
    preds = (w_1/(w_1+w_2)) * preds_1 + (w_2/(w_1+w_2)) * preds_2

    del preds_1, preds_2
    gc.collect()
    coefficients = [1.12232214, 1.73925866, 2.22506454]
    preds[preds <= coefficients[0]] = 0
    preds[np.where(np.logical_and(preds > coefficients[0], preds <= coefficients[1]))] = 1
    preds[np.where(np.logical_and(preds > coefficients[1], preds <= coefficients[2]))] = 2
    preds[preds > coefficients[2]] = 3
    sample_submission['accuracy_group'] = preds.astype(int)
    sample_submission.to_csv('submission.csv', index=False)
    sample_submission['accuracy_group'].value_counts(normalize=True)