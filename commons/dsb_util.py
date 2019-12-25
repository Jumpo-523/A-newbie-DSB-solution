# -*- coding: utf-8 -*-


from typing import List, Any, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook, tqdm
import re


class Constants(object):
    eventIdCategorizer = ["end of system-initiated instruction", "system-initiated instruction event", "help", "when an intro or outro movie starts to play","The movie ended event","when a video starts playing", "Correct",
    "Incorrect", "interactive", "hover", "place","drag", "click", "when the player finishes a round", "clicks on the play again", "the start of a round","The start game event", "tutorial","The exit game event"]
    game_category = {'Dino Drink':"size", 'Watering Hole (Activity)':"size", 'All Star Sorting':"size", 'Air Show':"size", 'Crystals Rule':"length", 'Bubble Bath':"size", 'Bottle Filler (Activity)':"size", 'Dino Dive':"length",'Happy Camel':"weight",
    'Pan Balance':"weight", 'Egg Dropper (Activity)':"weight", 'Leaf Leader':"weight", 'Sandcastle Builder (Activity)':"size", 'Scrub-A-Dub':"size",'Chow Time':"weight",
    'Fireworks (Activity)':"length",
    'Flower Waterer (Activity)':"length",
    'Bug Measurer (Activity)':"length",
    'Chicken Balancer (Activity)':"weight"}
    Assessment_category = {'Mushroom Sorter (Assessment)':"size", 'Bird Measurer (Assessment)':"length",
        'Cauldron Filler (Assessment)':"size", 'Cart Balancer (Assessment)':"weight",
        'Chest Sorter (Assessment)':"size"}
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
    'Chicken Balancer (Activity)':4000}
    def __init__(self, maps:dict):
        # import pdb; pdb.set_trace()
        self._set_attr(maps_data=maps)
        self.index_to_name(maps['activities_labels'])
        fields = ["titles_dict", "Assessment_category","game_category"]
        for f_ in fields:
            dict_ = getattr(Constants, f_)
            indexed_dict = {}
            for name, value in dict_.items():
                indexed_dict[Constants.name2idx[name]] = value 
            setattr(Constants, f_+"_indexed" ,indexed_dict)
        pass
    @classmethod
    def index_to_name(self, activities_labels):
        name2idx = {}
        for k, v in activities_labels.items():
            name2idx[v] = k
        self.name2idx = name2idx        
    @classmethod
    def _set_attr(self, maps_data:dict):
        for m, dic in maps_data.items():
            setattr(Constants, m ,dic)

        
class eda_event_data():
    def __init__(self, specs):
        self.specs = specs
        self.dict_event_ids = {k:set() for k in self.specs.event_id}
    def label_event_id(self, kw):
        events_with_kw = self.specs.loc[self.specs["info"].str.contains(kw), "event_id"].to_list()
        for e_id in events_with_kw:
            self.dict_event_ids[e_id].add(kw)
    def find_isolated_events(self):
        output = []
        for k, v in self.dict_event_ids.items():
            if not v:
                output.append(k)
        return output
from collections import Counter
class Classified_event_id():

    def __init__(self, specs, classification_cls):
        total = set()
        self.classification_cls = classification_cls
        for i in  classification_cls.dict_event_ids.values():
            total = total.union(i)
        self.dict_cls_category = {cls_cat:0 for cls_cat in total}
    def count_cls_cat(self, session):
        """set()のたいしょも入れないといけない。"""
        num_of_session_count = Counter(session["event_id"])
        for event_id in num_of_session_count:
            if self.classification_cls.dict_event_ids[event_id] == set():
                pass
            else:
                for e_category in self.classification_cls.dict_event_ids[event_id]:
                    self.dict_cls_category[e_category] += num_of_session_count[event_id]

    def sub_space_underscore(self):
        keys = self.dict_cls_category.keys()
        for k in keys:
            self.dict_cls_category[re.sub(string=k, pattern=" ", repl="_")] = \
                        self.dict_cls_category.pop(k)

def read_data():
    base_path = '/Users/junpei.takubo/Downloads/kaggle_dsb/'
    print('Reading train.csv file....')
    train = pd.read_csv(base_path+'data-science-bowl-2019/originals/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv(base_path+'data-science-bowl-2019/originals/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv(base_path+'data-science-bowl-2019/originals/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv(base_path + 'data-science-bowl-2019/originals/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(base_path + 'data-science-bowl-2019/originals/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
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
    activities_world_labels = dict(zip(np.arange(len(list_of_worlds)), list_of_worlds))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))


    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(list_of_user_activities, (4100*np.ones(len(list_of_user_activities))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code['Bird Measurer (Assessment)'] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    maps_data = {"activities_map": activities_map, 'assess_titles': assess_titles,"list_of_user_activities":list_of_user_activities,
                'list_of_event_code':list_of_event_code, 'list_of_event_id':list_of_event_id, 'all_title_event_code': all_title_event_code,
                'activities_labels':activities_labels, 'activities_world':activities_world, "win_code":win_code}
    return train, test, train_labels, maps_data 




# constants導入につき、titles_dict消す
def get_data(user_sample, constants:Constants, test_set=False):
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
    # last_accuracy_title入れたっけ？
    last_accuracy_title = {'acc_' + title: -10 for title in constants.assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in constants.list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in constants.list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in constants.list_of_user_activities}
    
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in constants.all_title_event_code}
    # calculate the last score of each activity
    gameActivityScores = {'score_title_' + str(ga_title): 0 for ga_title in constants.titles_dict.keys()}

    # length, size, weightに関連するactivity数を形状する。
    # activity数じゃないもっと目的に沿った指標を生成したい。
    activity_type = {'length':0, 'size':0, 'weight':0}
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session

        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = constants.activities_labels[session_title]         
   
        # if gameActivityScores.get('score_title_' + session_title) is not None:
        if constants.game_category.get(session_title_text) is not None:
            # import pdb;pdb.set_trace()
            # score_ = each_game_and_activity_score(titles_dict, session)
            # print(score_)
            # gameActivityScores['score_title_' + session_title] = score_
            # とりあえずイベントカウントで。scoreでも、duration_timeでもいいな。
            activity_type[constants.game_category[session_title_text]] += session.game_time.max()

        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):            
            # import pdb; pdb.set_trace()
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {constants.win_code[constants.activities_labels[session_title]]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            # features.update(event_code_count.copy())
            # features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            
            features.update(gameActivityScores.copy())
            features.update(activity_type.copy())

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
            last_accuracy_title['acc_' + str(session_title)] = accuracy
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
                        x = constants.activities_labels[k]
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

def get_train_and_test(train, test, constants):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample, constants)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        compiled_train += get_data(user_sample, constants)
        test_data = get_data(user_sample, constants, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    return reduce_train, reduce_test



def preprocess(reduce_train, reduce_test, constants:Constants):
    for df in [reduce_train, reduce_test]:
        # df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        # df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        event_codes = [2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 2040, 4090, 4220, 4095]
        event_codes_str = [str(ec) for ec in event_codes]
        # df['sum_event_code_count'] = df[[str(ec) for ec in event_codes ]].sum(axis = 1)
        
        # df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')
        # try:
        #     df.drop(columns=event_codes, inplace=True)
        # except KeyError:
        #     df.drop(columns=event_codes_str, inplace=True)
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in constants.assess_titles]
    return reduce_train, reduce_test, features


