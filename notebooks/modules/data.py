import json
import os
import re
import sys

import numpy as np
import pandas as pd
import itertools as it

from utils import *


class GAData():
    def __init__(self, path):
        self.path = path

    def retrieve_files(self):
        _, _, filenames = next(os.walk(self.path))
        uids = [os.path.splitext(f)[0] for f in filenames]
        return zip(filenames, uids)

    def get_basic_info(self, json_data, *args):
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
        info_df = duplicate_num_col_count(
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

    @property
    def basic_data(self):
        df_list = []
        for filename, uid in self.retrieve_files():
            with open(self.path+filename, encoding='utf-8') as f:
                json_data = json.load(f)
                df = self.get_basic_info(json_data)
                df['uid'] = uid
                df = move_last_col_first(df)
                df_list.append(df)
        user_df = pd.concat(df_list, axis=0, ignore_index=True)
        return user_df

    def get_user_journey(self, json_data):
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
        info_df = duplicate_num_col_count(
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
        user_df = duplicate_num_col_count(
            user_df, 'count').reset_index(drop=True)
        act_df = pd.DataFrame(dic_3).apply(flatten_list)
        fin_df = pd.concat([user_df, act_df], axis=1)
        return fin_df

    @property
    def journey_data(self):
        df_list = []
        for filename, uid in self.retrieve_files():
            with open(self.path+filename, encoding='utf-8') as f:
                json_data = json.load(f)
                df = self.get_user_journey(json_data)
                df['uid'] = uid
                df = move_last_col_first(df)
                df_list.append(df)
        user_df = pd.concat(df_list, axis=0, ignore_index=True)
        return user_df


class PageData():
    def __init__(self, basic_data):
        self.data = basic_data
        self.groupOne = self._clean_page_col()
        self.mask = self.data[self.data['uid'].isin(self.groupOne[self.groupOne['page'] 
                                                                == 'NaN']['uid'])]
        self.groupTwo = self._adjust_page()

    def _clean_page_col(self):
        content = self.data[['uid', 'page']]
        content = content.drop_duplicates().reset_index()
        pages = list()
        category = list()
        for i in range(len(content)):
            # Split content for each users
            eles = [eles for eles in content['page'].iloc[i].split('|')]
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
            
            if top == []:
                top.append('NaN')
            else:
                top=top

            # User id
            uid = content['uid'].iloc[i]

            # Append individual user's content to the group
            pages.extend(zip([uid] * len(top), top))
            category.extend(cat)

        page_df = pd.DataFrame(pages, columns=['uid', 'page'])
        return page_df

    def _adjust_page(self):
        """Clean pages without hyphen"""
        content = self.mask[['uid', 'page']]
        content = content.drop_duplicates().reset_index()
        pages = list()

        for i in range(len(content)):
            # Split content for each users
            eles = [eles for eles in content['page'].iloc[i].split('|')]

            # Flatten the list and strip whitespace
            top = flatten_list(eles)

            if top == []:
                top.append('NaN')
            else:
                top=top

            # User id
            uid = content['uid'].iloc[i]

            # Append individual user's content to the group
            pages.extend(zip([uid] * len(top), top))

        page_df = pd.DataFrame(pages, columns=['uid', 'page'])
        return page_df

    @property
    def cleaned(self):
        page_df = pd.concat([self.groupOne, self.groupTwo], axis=0)
        page_df['page'].replace('NaN', np.nan, inplace=True)
        page_df.dropna(subset=['page'], inplace=True)
        page_df = page_df.reset_index()
        return page_df

    def _find_page_pairs(self, x):
        pairs = pd.DataFrame(list(it.permutations(x.values, 2)), columns=['page', 'related page'])
        return pairs


    def get_page_combinations(self):
        page_combinations = self.cleaned.groupby('uid')['page'].apply(self._find_page_pairs)
        combination_counts = page_combinations.groupby(['page', 'related page']).size()
        combination_counts_df = combination_counts.to_frame(name='size').reset_index()
        combination_counts_df.sort_values('size', ascending=False, inplace=True)
        return combination_counts_df




class ActivityData():
    # Import the catalog to add tags on each category
    with open('D:\\Github\\mix-recsys\\databases\\static\\article.txt', 'r') as reader:
        article_dict = json.loads(reader.read())

    with open('D:\\Github\\mix-recsys\\databases\\static\\support.txt', 'r') as reader:
        support_dict = json.loads(reader.read())
        
    with open('D:\\Github\\mix-recsys\\databases\\static\\story.txt', 'r') as reader:
        story_dict = json.loads(reader.read())
        
    with open('D:\\Github\\mix-recsys\\databases\\static\\app.txt', 'r') as reader:
        app_dict = json.loads(reader.read())
        
    with open('D:\\Github\\mix-recsys\\databases\\static\\skill.txt', 'r') as reader:
        skill_dict = json.loads(reader.read())
        
    with open('D:\\Github\\mix-recsys\\databases\\static\\volunteer.txt', 'r') as reader:
        volunteer_dict = json.loads(reader.read())
        
    with open('D:\\Github\\mix-recsys\\databases\\static\\news.txt', 'r') as reader:
        news_dict = json.loads(reader.read())

    ARTICLE_DICT = article_dict
    MERGED_DICT = {**support_dict, **story_dict, **app_dict, **skill_dict, 
                    **volunteer_dict, **news_dict}

    def __init__(self, page_df):
        self.data = page_df
        self.sorted_data = self._sort_page_df()
        self.user_article = self._get_article_df()
        self.article = self._count_article_by_cat()
        self.other = self._get_activity_df()

    

    def _tidy_text(self):
        new_text = []
        for i in range(len(self.data)):
            text = re.sub('[^a-zA-Z]', ' ', self.data['page'][i])
            # Convert to lowercase
            text = text.lower()
            # Remove whitespace at the beginning or end of the text
            text = text.strip()
            if len(re.findall(r'\w+', text)) >= 0:
                text = text
            else:
                text = np.nan
            new_text.append(text)
        return new_text


    def _sort_page_df(self):
        self.data['page'] = self._tidy_text()
        page_sorted = self.data.groupby('uid')['page'].apply(list)
        return page_sorted

    
    def _get_article_df(self):
        user_df = pd.DataFrame(index = list(ActivityData.ARTICLE_DICT.keys()))
        
        for i in range(len(self.sorted_data)):
            user_id = self.sorted_data.index[i]
            user_content = set(self.sorted_data.iloc[i])

            user_list = []
            for _, page in enumerate(list(ActivityData.ARTICLE_DICT.keys())):
                search = user_content.intersection(set(ActivityData.ARTICLE_DICT[page]))
                user_cat = len(search)
                user_list.append(user_cat)
            user_df[user_id] = user_list
            user_article = user_df.T
    #         user_article.columns = user_article.columns.map(lambda x: ("_").join(x.split()))
        return user_article

    def _count_article_by_cat(self):
        article_cat = ['sex_relation', 'body', 'mental', 'drink_drug', 'housing',
                    'money', 'work_study', 'crime_safety', 'travel_lifestyle']
        # the first 16 columns containing sex_relation subtopics
        count_sex_relation = self.user_article.apply(
            lambda x: (x[:][:16]).sum(), axis=1)
        # column 16-23 containing body subtopics
        count_body = self.user_article.apply(lambda x: (x[:][16:24]).sum(), axis=1)
        # column 24-36 containing mental health subtopics
        count_mental = self.user_article.apply(lambda x: (x[:][24:37]).sum(), axis=1)
        # column 37-44 containing drink and drug subtopics
        count_drink_drug = self.user_article.apply(
            lambda x: (x[:][37:45]).sum(), axis=1)
        # column 45-50 containing housing subtopics
        count_housing = self.user_article.apply(lambda x: (x[:][45:51]).sum(), axis=1)
        # column 51-59 containing money subtopics
        count_money = self.user_article.apply(lambda x: (x[:][51:60]).sum(), axis=1)
        # column 60-72 containing work and study subtopics
        count_work = self.user_article.apply(lambda x: (x[:][60:73]).sum(), axis=1)
        # column 73-78 containing crime and safety subtopics
        count_crime = self.user_article.apply(lambda x: (x[:][73:78]).sum(), axis=1)
        # column 78-83 containing travel and lifestyle subtopics
        count_travel = self.user_article.apply(lambda x: (x[:][78:]).sum(), axis=1)

        count_df = pd.concat([count_sex_relation, count_body, count_mental, count_drink_drug,
                            count_housing, count_money, count_work, count_crime, count_travel], axis=1)
        count_df.columns = article_cat
        return count_df.sum(axis=1).to_frame().rename(columns={0:'expert article'})

 
    def _count_keywords(self, kwords, user_content): 
        count_list = []
        for kword in kwords:
            for doc in user_content:
                if doc.find(kword) != -1:
                    count_list.append(doc)
        return len(count_list)  


    def _get_activity_df(self):
        user_df = pd.DataFrame(index=list(ActivityData.MERGED_DICT.keys()))
        for i in range(len(self.sorted_data)):
            user_id = self.sorted_data.index[i]
            user_content = set(self.sorted_data.iloc[i])

            user_list = []
            for cat in list(ActivityData.MERGED_DICT.keys()):
                kwords = ActivityData.MERGED_DICT[cat]
                kword_num = self._count_keywords(kwords, user_content)
                user_list.append(kword_num)

            user_df[user_id] = user_list
    
        return user_df.T

    @property
    def cleaned(self):
        self.other.insert(0, 'expert article', self.article)
        return self.other


