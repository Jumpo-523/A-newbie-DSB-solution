# -*- coding: utf-8 -*-


from typing import List, Any, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook, tqdm
import re
from sklearn import linear_model
from .new_features import learningRoute, Title_levels, countSession_eachlevels
import json

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
    clip_times = {'Welcome to Lost Lagoon!':19, 'Tree Top City - Level 1':17, 'Ordering Spheres':61,
                        'Costume Box':61, '12 Monkeys':109, 'Tree Top City - Level 2':25,
                        "Pirate's Tale":80, 'Treasure Map':156, 'Tree Top City - Level 3':26, 'Rulers':126,
                        'Magma Peak - Level 1':20, 'Slop Problem':60, 'Magma Peak - Level 2':22,
                          'Crystal Caves - Level 1':18, 'Balancing Act':72,'Lifting Heavy Things':118,
              'Crystal Caves - Level 2':24,'Honey Cake':142,'Crystal Caves - Level 3':19,
                'Heavy, Heavier, Heaviest':61}


    categorical_features = ['session_title']
    target_variable = 'accuracy_group'

    def __init__(self, maps:dict, train_labels, Title_levels, categoricals=['session_title']):
        # import pdb; pdb.set_trace()
        self._set_attr(maps_data=maps)
        self.train_labels = train_labels
        self.Title_levels = Title_levels
        self.categoricals = categoricals
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

###########################################################################################
###########################################################################################
###########################################################################################
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
    assess_indices = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code['Bird Measurer (Assessment)'] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    maps_data = {"activities_map": activities_map, 'assess_titles': assess_titles,"list_of_user_activities":list_of_user_activities,
                'list_of_event_code':list_of_event_code, 'list_of_event_id':list_of_event_id, 'all_title_event_code': all_title_event_code,
                'activities_labels':activities_labels, 'activities_world':activities_world, 
                'activities_world_labels':activities_world_labels, "win_code":win_code}
    return train, test, train_labels, maps_data 


def count_to_share(count_dict, denominator=None):
    """2ndprocessinにてtotal_count by installation_id　で割る"""
    if denominator is None:
        # we have to set denominator as total of count_dict
        denominator = sum(count_dict.values())
    for k, v in count_dict.items():
        count_dict[k] = v/denominator if denominator !=0 else 0
    return count_dict

class Features:
    pass
class Games(Features):
    def __init__(self):
        self.mean_game_duration = 0
        self.mean_game_round = 0
        self.mean_game_level = 0
        self.Game_mean_event_count = 0
        # which to use
        self.accumulated_game_miss = 0
        self.game_miss_mean_count = 0
        self.n = 1
    def to_dict(self):
        dic = self.__dict__.copy()
        dic.pop("n")
        return dic 
    

def cnt_miss(df):
    cnt = 0
    for e in range(len(df)):
        x = df['event_data'].iloc[e]
        y = json.loads(x)['misses']
        cnt += y
    return cnt

def game_features(games:Games, session, constants:Constants):

    games.Game_mean_event_count = (games.Game_mean_event_count + session['event_count'].iloc[-1])/2.0
    game_s = session[session.event_code == 2030]
    # import pdb; pdb.set_trace()
    # exit_completely = sum(session.event_id.map(lambda x: x in constants.exit_completely_list))
    # print(exit_completely, games.n)
    misses_cnt = cnt_miss(game_s)
    games.accumulated_game_miss += misses_cnt
    games.game_miss_mean_count += (misses_cnt + games.game_miss_mean_count)/2.0
    try:
        game_round = json.loads(session['event_data'].iloc[-1])["round"]
        games.mean_game_round =  (games.mean_game_round + game_round)/2.0
    except:
        pass
    try:
        game_duration = json.loads(session['event_data'].iloc[-1])["duration"]
        games.mean_game_duration = (games.mean_game_duration + game_duration) /2.0
    except:
        pass
    try:
        game_level = json.loads(session['event_data'].iloc[-1])["level"]
        games.mean_game_level = (games.mean_game_level + game_level) /2.0
    except:
        pass
    games.n += 1
    return games

# constants導入につき、titles_dict消す
def get_data(installation_id, user_sample, constants:Constants, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    user_last_activities = {'last_Clip':999, 'last_Activity': 999,
                            'last_Assessment': 999, 'last_Game':999}
    # new features: time spent in each activity
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_last_activity = user_sample['timestamp'].values[0]
    durations = []
    # last_accuracy_title入れたっけ？
    # last_accuracy_title = {'acc_' + title:0 for title in constants.assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in constants.list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in constants.list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in constants.list_of_user_activities}
    
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in constants.all_title_event_code}
    # calculate the last score of each activity
    gameActivityScores = {'score_title_' + str(ga_title): 0 for ga_title in constants.titles_dict.keys()}

    level_counts = {'title_treetopcity_level_1':0, 'title_treetopcity_level_2':0, 'title_treetopcity_level_3':0,
                    'title_magmapeak_level_1':0, 'title_magmapeak_level_2':0, 'title_crystalcaves_level_1':0,
                    'title_crystalcaves_level_2':0, 'title_crystalcaves_level_3':0, 
                    }
    # game related
    games = Games()
    # Activity related
    Activity_mean_event_count = 0

    # length, size, weightに関連するactivity数を形状する。
    # activity数じゃないもっと目的に沿った指標を生成したい。
    activity_type = {'length':0, 'size':0, 'weight':0}
    check_routes = learningRoute()._set_dict()

    clip_durations = 0; clip_n = 1
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = constants.activities_labels[session_title]         
        session_world = session['world'].iloc[0]
        check_routes.record_experiences(title_text=session_title_text,
                           world_text=constants.activities_world_labels[session_world])
        level_counts = countSession_eachlevels(level_counts, constants.Title_levels, session_title_text)
        # if gameActivityScores.get('score_title_' + session_title) is not None:
        if constants.game_category.get(session_title_text) is not None:
            # import pdb;pdb.set_trace()
            # score_ = each_game_and_activity_score(titles_dict, session)
            # print(score_)
            # gameActivityScores['score_title_' + session_title] = score_
            # とりあえずイベントカウントで。scoreでも、duration_timeでもいいな。
            activity_type[constants.game_category[session_title_text]] += session.game_time.max()
        
        if session_type=="Activity":
            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1])/2.0
        if session_type=="Game":
            games = game_features(games, session, constants)
        if session_type=="Clip":
            # import pdb; pdb.set_trace()
            if not constants.clip_times.get(session_title_text):
                print("There are other clips...")
            # clip times inserted
            # ここでsession.index + 1がアウトオフバウンズ？
            try:
                _clip_duration = user_sample['timestamp'][session.index + 1] - session['timestamp'].iloc[0]
                clip_duration = _clip_duration.dt.seconds - constants.clip_times[session_title_text]
                clip_durations = clip_durations + (clip_duration.iloc[0] - clip_durations)/clip_n
            except KeyError:
                pass
        user_last_activities['last_' + session_type] = session_title
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # on_route_values = check_routes.is_on_appropriate_route(world_text=constants.activities_world_labels[session_world],
            #                                      Asessment_text=session_title_text)
            
            # import pdb; pdb.set_trace()
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {constants.win_code[constants.activities_labels[session_title]]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = count_to_share(user_activities_count).copy()
            
            features.update(count_to_share(title_count).copy())
            features.update(user_last_activities.copy())
            
            # title_event_code_count
            features.update(count_to_share(title_event_code_count).copy())
            
            features.update(gameActivityScores.copy())
            
            features.update({"clip_durations":clip_durations}.copy())

            features.update(activity_type.copy())
            # features.update({"on_route":on_route_values})
            # features.update(last_accuracy_title.copy())
            features.update(count_to_share(event_code_count).copy())
            # features.update(event_id_count.copy())

            features.update(count_to_share(level_counts.copy()))
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]

            # added an activity feature
            features['Activity_mean_event_count'] = Activity_mean_event_count
            # added game features
            features.update(games.to_dict().copy())

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
            # last_accuracy_title['acc_' + str(session_title)] = accuracy
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
                features['num_correct'] = true_attempts
                features['num_incorrect'] = false_attempts
                # print(false_attempts)
        # constants.train_labels.query(f'installation_id == "{installation_id}" and game_session == "{i}"')['num_incorrect'].iloc[0]
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
        # event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        time_last_activity = session['timestamp'].max()
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
        compiled_train += get_data(ins_id, user_sample, constants)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        compiled_train += get_data(ins_id, user_sample, constants)
        test_data = get_data(ins_id, user_sample, constants, test_set = True)
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
        
        # reduce_train = target_encoder(reduce_train, constants.target_variable, constants.categorical_features)
    # reduce_train, reduce_test = onehot_encoder(reduce_train, reduce_test, constants.categoricals)
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in constants.assess_titles]
    return reduce_train, reduce_test, features


def target_encoder(reduce_train, target_variable, categorical_features):
    for cat in categorical_features:
            dic_ = reduce_train.groupby(by=cat).agg({target_variable:"mean"}).to_dict()[target_variable]
            print(dic_)
            # int 64にfloatブッ込めないよね。
            reduce_train[cat].replace(dic_, inplace=True)
    return  reduce_train


def onehot_encoder(reduce_train, reduce_test, category_cols):
    for col in category_cols:
        one_hot_vectors = pd.get_dummies(pd.concat([reduce_train[col],reduce_test[col]], axis=0))
        one_hot_vectors.columns = [col+'_'+str(i) for i in one_hot_vectors.columns]
        reduce_train.drop(columns=col, inplace=True);reduce_test.drop(columns=col, inplace=True)
        reduce_train = pd.concat([reduce_train, one_hot_vectors[:len(reduce_train)]], axis=1)
        reduce_test = pd.concat([reduce_test, one_hot_vectors[len(reduce_train):]], axis=1)
    return reduce_train, reduce_test


class Feature_selection:
    '''feature selection with Lasso'''
    def __init__(self, target, cols_to_drop=[], category_cols=[]):
        self.cols_to_drop = cols_to_drop
        self.category_cols = category_cols
        self.target = target
        self.l1_reg=linear_model.Lasso(alpha=0.001, max_iter=100000)
        self.useful_cols = None
        self.trained = False
        
    def onehot_encoder(self, reduce_train_):
        reduce_train = reduce_train_.copy()
        for col in self.category_cols :
            one_hot_vectors = pd.get_dummies(reduce_train[col])
            one_hot_vectors.columns = [col+'_'+str(i) for i in one_hot_vectors.columns]
            reduce_train.drop(columns=col, inplace=True)
            reduce_train = pd.concat([reduce_train, one_hot_vectors], axis=1)
        return reduce_train

    def fit_transform(self, reduce_train, y, na_cols_drop=True):
        self.useful_cols = [c for c in reduce_train.columns if c not in self.cols_to_drop]
        X = reduce_train[self.useful_cols]
        if na_cols_drop:
            X = X.dropna(axis=1)
        self.l1_reg.fit(X, y)
        self.cols_to_regress = X.columns.to_list()
        print("learning was done. Please use 'get_useful_features' method.")
        self.trained = True
        return self.l1_reg.predict(X)
    def predict(self, X_test):
        if not self.trained:
            raise NotImplementedError("You have to train beforehand")
        return self.l1_reg.predict(X_test[self.cols_to_regress])
    def get_useful_features(self):
        res = pd.DataFrame({"coefficient": self.l1_reg.coef_}, index=self.cols_to_regress)
        output = res.index[res.coefficient.map(lambda x: abs(x)!=0)].to_list()
        # one hot encodingしたカラムをどう扱う？
        for cat_col in self.category_cols:
            not_used_in_lgb = [col for col in output if cat_col in col]
            if not_used_in_lgb:
                [output.remove(el) for el in not_used_in_lgb]
                output.append(cat_col)
        return output

# These methods are from
# https://www.kaggle.com/ragnar123/truncated-val/
# That is the good chance to know why truncation is important and how to do that.

def remove_correlated_features(reduce_train, features):
    counter = 0
    to_remove = []
    for feat_a in features:
        for feat_b in features:
            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
                c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]
                if c > 0.995:
                    counter += 1
                    to_remove.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    return to_remove


# function to exclude columns from the train and test set if the mean is different, also adjust test column by a factor to simulate the same distribution
def exclude(reduce_train, reduce_test, features):
    to_exclude = [] 
    ajusted_test = reduce_test.copy()
    for feature in features:
        if feature not in ['accuracy_group', 'installation_id', 'session_title']:
            data = reduce_train[feature]
            train_mean = data.mean()
            data = ajusted_test[feature] 
            test_mean = data.mean()
            try:
                ajust_factor = train_mean / test_mean
                if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:
                    to_exclude.append(feature)
                    print(feature)
                else:
                    ajusted_test[feature] *= ajust_factor
            except:
                to_exclude.append(feature)
                print(feature)
    return to_exclude, ajusted_test






