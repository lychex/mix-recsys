from gadata import *

DATA_PATH = './databases/json-data/'

data = GAData(DATA_PATH).journey_data
print(data.head())
