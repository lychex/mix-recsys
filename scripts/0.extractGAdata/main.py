import sys

sys.path.append('./scripts/0.extractGAdata/modules')
from data import GAData


#=====================================================================================
# Extract two major dataframe - basic_data, journery_data from 6106 json files 
#=====================================================================================

DATA_PATH = './databases/json-data/'

journey_data = GAData(DATA_PATH).journey_data
journey_data.to_csv('./databases/csv-data/journey_data.csv', index=False)

basic_data = GAData(DATA_PATH).basic_data
basic_data.to_csv('./databases/csv-data/basic_data.csv', index=False)

