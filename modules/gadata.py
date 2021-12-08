import json
import os

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
        page_df = page_df.reset_index(drop=True)
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

