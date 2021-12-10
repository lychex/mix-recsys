import pandas as pd
import numpy as np
import sys
import json
from scipy import stats
import functools

sys.path.append("./scripts/2.dataPrep/modules")
from helper import day_gap, clean_url_col, create_feature, get_pairs, clean_text, tag_mapper


#=====================================================================================
# Clean journey_data because we want to use pageview with timestamps
#=====================================================================================

data = pd.read_csv('./databases/csv-data/journey_data.csv')

# Convert time featture into datetime 
data['datetime'] = data['date'] + ' ' + data['time']
data['datetime'] = pd.to_datetime(data['datetime'])

# Calculate the time difference between each page
data['duration'] = - data.groupby('uid')['datetime'].diff(periods=1)


# Convert time difference into minutes
data['dur_mins'] = data['duration'].map(day_gap)
# Preprocess pageURL column
data['pageURL'] = data['pageURL'].replace('', np.nan)
# Backfill missing data
data['pageURL'] = data['pageURL'].fillna(method='bfill')


# Apply text cleaning to pageURL
data['pageURL'] = data['pageURL'].apply(clean_url_col)


# Remove user-item duplicates entries
# Sum up the total time a user spend on a page
data_cleaned = data.groupby(['uid', 'pageURL'], 
                            as_index=False).agg({'datetime':'first', 'page': 'last', 'dur_mins': 'sum'})
# Cap the maximum duration to 30 minutes
data_cleaned['dur_mins'] = np.clip(data_cleaned['dur_mins'], a_max=30, a_min=0.5)



dur_mins = data_cleaned['dur_mins'].values

score = [stats.percentileofscore(dur_mins, a, 'rank') for a in dur_mins]
# Make ratings between 0 and 1
score = [ele/100 for ele in score]
data_cleaned['interest'] = score


# Remove blank entries
data_cleaned = data_cleaned[['uid', 'page', 'pageURL', 'datetime', 'interest']]
data_cleaned = data_cleaned[data_cleaned['page'] != 'Page Not Found - The Mix']
data_cleaned = data_cleaned[data_cleaned.pageURL != '']
data_cleaned = data_cleaned[data_cleaned.page != '']


print(f'[INFO]: Cleaned Dataframe NA number:\n{data_cleaned.isna().sum()}')

data_cleaned = data_cleaned[data_cleaned.page != '']

print('[INFO] Cleaned Dataframe shape\n', data_cleaned.shape)

# Create item features
data_cleaned['category'] = data_cleaned['pageURL'].apply(create_feature)

# Check if all pages are categorized
print('[INFO]: Unique category of pages: \n', data_cleaned['category'].unique())


data_cleaned.to_csv('./databases/csv-data/cleaned.csv', index=False)


#=====================================================================================
# Create pageview feature by adding individual pages tags
#=====================================================================================
data_cleaned = pd.read_csv('./databases/csv-data/cleaned.csv')

with open('./databases/static/article.txt', 'r') as reader:
    article_dict = json.loads(reader.read()) 


pairs = get_pairs(article_dict)

article_tag = data_cleaned.copy()[['pageURL', 'page']]


article_tag['page'] = article_tag.page.map(clean_text)
article_tag['tag'] = article_tag.page.map(functools.partial(tag_mapper, pairs=pairs))

article_tag.to_csv('./databases/csv-data/article_tag.csv', index=False)

print(data_cleaned.shape)