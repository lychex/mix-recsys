from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def retrieve_files(path):
    _, _, filenames = next(os.walk(path))
    uids = [os.path.splitext(f)[0] for f in filenames]
    return zip(filenames, uids)


def move_last_col_first(df):
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df_new = df[cols]
    return df_new


def combine_data(path):
    df_list = []
    for filename, uid in retrieve_files(path):
        with open(path+filename, encoding='utf-8') as f:
            json_data = json.load(f)
            df = transform_data(json_data)
            df['uid'] = uid
            df = move_last_col_first(df)
            df_list.append(df)
    user_df = pd.concat(df_list, axis=0, ignore_index=True)
    return user_df


def remove(col, df):
    df.drop(columns=col, axis=1, inplace=True)


def duplicate_session_count(df, num_col_name):
    new_df = df.loc[df.index.repeat(df[num_col_name])]
    new_df.drop(columns=num_col_name, axis=1, inplace=True)
    return new_df


def flatten_list(nested_list):
    """Converts a nested list to a flat list"""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def sort_tuple(tup):
    return(sorted(tup, key=lambda x: x[1]))


def sort_dict(dic, reverse=True):
    return sorted(dic.items(), key=lambda x: x[1], reverse=reverse)


# Function for sorting tf_idf in descending order
def sort_matrix(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def transform_data(json_data, *args):
    dic_1 = {
        'date': [],
        'hasGoal': [],
        'hasRevenue': [],
        'sessionCount': []}

    dic_2 = {
        'duration': [],
        'deviceCategory': [],
        'channel': [],
        'activitySummary': []}

    dic_3 = {
        'pageTitle': [],
        'eventAction': [],
        'eventLabel': []}

    # Create basic info for the first dictionary
    for i in range(len(json_data['dates'])):
        info = json_data['dates'][i]
        dic_1['date'].append(info['date'])
        dic_1['hasGoal'].append(info['hasGoal'])
        dic_1['hasRevenue'].append(info['hasRevenue'])
        dic_1['sessionCount'].append(info['sessionCount'])

        # Break down sessions
        sessions = info['sessions']
        for j in range(len(sessions)):
            dic_2['duration'].append(sessions[j]['duration'])
            dic_2['deviceCategory'].append(sessions[j]['deviceCategory'])
            dic_2['channel'].append(sessions[j]['channel'])
            dic_2['activitySummary'].append(sessions[j]['activitySummary'])

            # Break down activities
            activities = sessions[j]['activities']
            for m in range(len(activities)):
                dic_3['pageTitle'].append(
                    activities[m]['details'][0].get('Page title', ''))
                dic_3['eventAction'].append(
                    activities[m]['details'][0].get('Event action', ''))
                dic_3['eventLabel'].append(
                    activities[m]['details'][0].get('Event label', ''))

    info_df = pd.DataFrame(dic_1)
    info_df = duplicate_session_count(
        info_df, 'sessionCount').reset_index(drop=True)
    sess_df = pd.DataFrame(dic_2)
    break_down = pd.json_normalize(
        sess_df['activitySummary']).replace(np.nan, 0)
    break_down.columns = break_down.columns.map(lambda x: x.lower())
    remove('activitySummary', sess_df)
    sess_df = pd.concat([sess_df, break_down], axis=1)
    user_df = pd.concat([info_df, sess_df], axis=1)
    act_df = pd.DataFrame(dic_3).apply(lambda x: flatten_list(x))
    act_df['event'] = act_df['eventAction'] + " " + act_df['eventLabel']
    # Extract unique values from column "pageTitle" and "event"
    interested_page = act_df['pageTitle'].unique()
    interested_page = interested_page[interested_page != '']
    interested_action = act_df['event'].unique()
    interested_action = interested_action[interested_action != '']
    user_df['page'] = "|".join(interested_page)
    user_df['action'] = "|".join(interested_action)
    return user_df


def clean_page_col(df, col_name='page'):
    content = df[['uid', col_name]]
    content = content.drop_duplicates().reset_index()
    pages = list()
    category = list()
    for i in range(len(content)):
        # Split content for each users
        eles = [eles for eles in content[col_name].iloc[i].split('|')]
        # Unify the dash symbol and use it as delimiter to split the category
        user_content = [ele.replace('—', '-').strip().split('-')
                        for ele in eles]
        # Extract content user interested in
        top = [ele[:-1] for ele in user_content]
        # Join strings that are unnecessarily split
        top = [['-'.join(ele)] if len(ele) > 1 else ele for ele in top]
        # Extract the category
        cat = [ele[-1] for ele in user_content]

        # Flatten the list and strip whitespace
        top = flatten_list(top)
        top = [ele.strip() for ele in top]
        cat = [ele.strip() for ele in cat]

        # User id
        uid = content['uid'].iloc[i]

        # Append individual user's content to the group
        pages.extend(zip([uid] * len(top), top))
        category.extend(cat)

    page_df = pd.DataFrame(pages, columns=['uid', 'page'])
    return page_df


def convert_to_secs(duration):
    m, s = map(int, duration.split(':'))
    return m*60+s+1


def cal_active_days(date_col):
    return (list(date_col)[0] - list(date_col)[-1]).days + 1


def count_cat(col_list):
    cat_set = set()
    for item in col_list:
        cat_set.add(item)
    return [cat_set]


def clean_multi_cat_col(df, col_name, index_col='uid'):
    new_df = df.groupby(index_col, as_index=False)[col_name].apply(count_cat)
    new_df[col_name] = new_df[col_name].apply(frozenset)
    new_df = new_df.set_index(index_col)
    for cat in frozenset.union(*new_df['deviceCategory']):
        new_df[cat] = new_df.apply(lambda _: int(cat in _[col_name]), axis=1)
    new_df.drop(columns=col_name, inplace=True)
    return new_df


def find_triage_info(user_action):
    text = user_action
    ele_list = []
    e_1 = re.findall(r"(age)(?:\s)(\d+)", text)
    e_2 = re.findall(r'(location)(?:\s)(?:")(\w+\s-\s\w+)(?:")', text)
    e_3 = re.findall(r'(search)(?:\s")(\w+)(?:")', text)
    e_4 = re.findall(r'(mood)(?:\s)(\d+)', text)
    ele_list.extend(e_1)
    ele_list.extend(e_2)
    ele_list.extend(e_3)
    ele_list.extend(e_4)
    return ele_list


def get_pageview(json_data):
    dic_1 = {
        'date': [],
        'hasGoal': [],
        'hasRevenue': [],
        'sessionCount': [],
    }

    dic_2 = {
        'duration': [],
        'deviceCategory': [],
        'channel': [],
        'activitySummary': [],
    }

    dic_3 = {
        'time': [],
        'page': [],
        'pageURL': [],
    }

    # Create basic info for the first dictionary
    for i in range(len(json_data['dates'])):
        info = json_data['dates'][i]
        dic_1['date'].append(info['date'])
        dic_1['hasGoal'].append(info['hasGoal'])
        dic_1['hasRevenue'].append(info['hasRevenue'])
        dic_1['sessionCount'].append(info['sessionCount'])

        # Break down sessions
        sessions = info['sessions']
        for j in range(len(sessions)):
            dic_2['duration'].append(sessions[j]['duration'])
            dic_2['deviceCategory'].append(sessions[j]['deviceCategory'])
            dic_2['channel'].append(sessions[j]['channel'])
            dic_2['activitySummary'].append(sessions[j]['activitySummary'])

            # Break down activities
            activities = sessions[j]['activities']
            for m in range(len(activities)):
                dic_3['time'].append(activities[m].get('time', ''))
                dic_3['page'].append(
                    activities[m]['details'][0].get('Page title', ''))
                dic_3['pageURL'].append(
                    activities[m]['details'][0].get('Page URL', ''))

    info_df = pd.DataFrame(dic_1)
    info_df = duplicate_session_count(
        info_df, 'sessionCount').reset_index(drop=True)
    sess_df = pd.DataFrame(dic_2)
    break_down = pd.json_normalize(
        sess_df['activitySummary']).replace(np.nan, 0)
    break_down.columns = break_down.columns.map(lambda x: x.lower())
    break_down['pageview'] = break_down['pageview'].map(lambda x: int(x))
    try:
        break_down['event'] = break_down['event'].map(lambda x: int(x))
        break_down['count'] = break_down['pageview'] + break_down['event']
    except:
        break_down['count'] = break_down['pageview']
        break_down['event'] = ''
    sess_df = pd.concat([sess_df, break_down], axis=1)
    user_df = pd.concat([info_df, sess_df], axis=1)
    user_df = user_df[['date', 'count']]
    user_df = duplicate_session_count(user_df, 'count').reset_index(drop=True)
    act_df = pd.DataFrame(dic_3).apply(flatten_list)
    fin_df = pd.concat([user_df, act_df], axis=1)
    return fin_df


def combine_timestamp_data(path):
    df_list = []
    for filename, uid in retrieve_files(path):
        with open(path+filename, encoding='utf-8') as f:
            json_data = json.load(f)
            df = get_pageview(json_data)
            df['uid'] = uid
            df = move_last_col_first(df)
            df_list.append(df)
    user_df = pd.concat(df_list, axis=0, ignore_index=True)
    return user_df


def get_user_journey(json_data):
    dic_1 = {
        'date': [],
        'hasGoal': [],
        'hasRevenue': [],
        'sessionCount': [],
    }

    dic_2 = {
        'duration': [],
        'deviceCategory': [],
        'channel': [],
        'activitySummary': [],
    }

    dic_3 = {
        'time': [],
        'page': [],
        'pageURL': [],
        'eventAction': [],
    }

    # Create basic info for the first dictionary
    for i in range(len(json_data['dates'])):
        info = json_data['dates'][i]
        dic_1['date'].append(info['date'])
        dic_1['hasGoal'].append(info['hasGoal'])
        dic_1['hasRevenue'].append(info['hasRevenue'])
        dic_1['sessionCount'].append(info['sessionCount'])

        # Break down sessions
        sessions = info['sessions']
        for j in range(len(sessions)):
            dic_2['duration'].append(sessions[j]['duration'])
            dic_2['deviceCategory'].append(sessions[j]['deviceCategory'])
            dic_2['channel'].append(sessions[j]['channel'])
            dic_2['activitySummary'].append(sessions[j]['activitySummary'])

            # Break down activities
            activities = sessions[j]['activities']
            for m in range(len(activities)):
                dic_3['time'].append(activities[m].get('time', ''))
                dic_3['page'].append(
                    activities[m]['details'][0].get('Page title', ''))
                dic_3['pageURL'].append(
                    activities[m]['details'][0].get('Page URL', ''))
                dic_3['eventAction'].append(
                    activities[m]['details'][0].get('Event action', ''))

    info_df = pd.DataFrame(dic_1)
    info_df = duplicate_session_count(
        info_df, 'sessionCount').reset_index(drop=True)
    sess_df = pd.DataFrame(dic_2)
    break_down = pd.json_normalize(
        sess_df['activitySummary']).replace(np.nan, 0)
    break_down.columns = break_down.columns.map(lambda x: x.lower())
    break_down['pageview'] = break_down['pageview'].map(lambda x: int(x))
    try:
        break_down['event'] = break_down['event'].map(lambda x: int(x))
        break_down['count'] = break_down['pageview'] + break_down['event']
    except:
        break_down['count'] = break_down['pageview']
        break_down['event'] = ''
    sess_df = pd.concat([sess_df, break_down], axis=1)
    user_df = pd.concat([info_df, sess_df], axis=1)
    user_df = user_df[['date', 'count']]
    user_df = duplicate_session_count(user_df, 'count').reset_index(drop=True)
    act_df = pd.DataFrame(dic_3).apply(flatten_list)
    fin_df = pd.concat([user_df, act_df], axis=1)
    return fin_df


def combine_journey_data(path):
    df_list = []
    for filename, uid in retrieve_files(path):
        with open(path+filename, encoding='utf-8') as f:
            json_data = json.load(f)
            df = get_user_journey(json_data)
            df['uid'] = uid
            df = move_last_col_first(df)
            df_list.append(df)
    user_df = pd.concat(df_list, axis=0, ignore_index=True)
    return user_df


#############################################################################################
#                                      Delete Later
#############################################################################################
# Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
# Creating a list of custom stopwords
cust_words = ['page', 'mix', 'searched', 'find', 'general', 'read', 'first', 'please',
              'op', 'greentea', 'speak', 'thread', 'post', 'posting',
              'another', 'ambassador', 'voice', 'tw', 'th', ]
stop_words = stop_words.union(cust_words)


def build_corpus(df, text_col, stop_words):

    corpus = []
    for i in range(len(df)):
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', df[text_col][i])
        # Convert to lowercase
        text = text.lower()
        # Remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
        # Remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)
        # Convert to list from string
        text = text.split()
        # Stemming
        ps = PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        text = " ".join(text)
        corpus.append(text)
    return corpus


def extract_keyword_from_vector(corpus, corpus_idx, n=1):

    vec = CountVectorizer(min_df=4, stop_words=stop_words,
                          max_features=3000, ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(X)

    doc = corpus[corpus_idx]
    doc_vec = tfidf_transformer.transform(vec.transform([doc]))

    feature_names = vec.get_feature_names()
    sorted_items = sort_matrix(doc_vec.tocoo())

    # Use only top n items from vector
    sorted_items = sorted_items[:n]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    results = list(zip(feature_vals, score_vals))

    return results


def sort_page_df(page_df):
    page_df['page'] = tidy_text(page_df, 'page')
    page_sorted = page_df.groupby('uid')['page'].apply(list)
    return page_sorted


def tidy_text(df, text_col, filter_num=0):
    new_text = []
    for i in range(len(df)):
        text = re.sub('[^a-zA-Z]', ' ', df[text_col][i])
        # Convert to lowercase
        text = text.lower()
        # Remove whitespace at the beginning or end of the text
        text = text.strip()
        if len(re.findall(r'\w+', text)) >= filter_num:
            text = text
        else:
            text = np.nan
        new_text.append(text)
    return new_text


def get_user_content_df(page_df, content_dict):

    page_sorted = sort_page_df(page_df)

    df_list = []
    for i in range(len(page_sorted)):
        user_id = page_sorted.index[i]
        user_content = user_content = set(page_sorted.iloc[i])

        user_list = []
        for j, page in enumerate(list(content_dict.keys())):
            search = user_content.intersection(set(content_dict[page]))
            if search != set():
                user_list.extend(list(search))
            num_of_article = len(user_list)
            df = pd.DataFrame(
                {'uid': [user_id] * num_of_article, 'content': user_list})

        df_list.append(df)

    user_content_df = pd.concat(df_list, axis=0)
    user_content_df.drop_duplicates(inplace=True)
    user_content_df = user_content_df.reset_index(drop=True)
    return user_content_df


def count_article_by_cat(user_article):
    article_cat = ['sex_relation', 'body', 'mental', 'drink_drug', 'housing',
                   'money', 'work_study', 'crime_safety', 'travel_lifestyle']
    # the first 16 columns containing sex_relation subtopics
    count_sex_relation = user_article.apply(
        lambda x: (x[:][:16]).sum(), axis=1)
    # column 16-23 containing body subtopics
    count_body = user_article.apply(lambda x: (x[:][16:24]).sum(), axis=1)
    # column 24-36 containing mental health subtopics
    count_mental = user_article.apply(lambda x: (x[:][24:37]).sum(), axis=1)
    # column 37-44 containing drink and drug subtopics
    count_drink_drug = user_article.apply(
        lambda x: (x[:][37:45]).sum(), axis=1)
    # column 45-50 containing housing subtopics
    count_housing = user_article.apply(lambda x: (x[:][45:51]).sum(), axis=1)
    # column 51-59 containing money subtopics
    count_money = user_article.apply(lambda x: (x[:][51:60]).sum(), axis=1)
    # column 60-72 containing work and study subtopics
    count_work = user_article.apply(lambda x: (x[:][60:73]).sum(), axis=1)
    # column 73-78 containing crime and safety subtopics
    count_crime = user_article.apply(lambda x: (x[:][73:78]).sum(), axis=1)
    # column 78-83 containing travel and lifestyle subtopics
    count_travel = user_article.apply(lambda x: (x[:][78:]).sum(), axis=1)

    count_df = pd.concat([count_sex_relation, count_body, count_mental, count_drink_drug,
                          count_housing, count_money, count_work, count_crime, count_travel], axis=1)
    count_df.columns = article_cat
    return count_df


def get_article_list(catalog_dict, catalog_keys):
    items = [d[k] for k in catalog_keys for d in [catalog_dict]]
    items = flatten_list(items)
    return [x for x in items if x]


def normalize(df, col_name, a, b):
    x_max = df[col_name].max()
    x_min = df[col_name].min()
    x_normalized = (b-a) * (df[col_name] - x_min)/(x_max - x_min) + a
    return x_normalized
