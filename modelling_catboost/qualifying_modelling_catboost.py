
##!pip install requests

##!pip install CatBoost

import requests
import pandas as pd
import time
import datetime
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from catboost import Pool

"""get qualifying info"""

qualifying = {'season': [],
        'round': [],
        'circuit_id': [],
        'lat': [],
        'long': [],
        'country': [],
        'city':[],
        'q_date': []}

for year in list(range(2014,2023)):
    
    url = 'https://ergast.com/api/f1/{}.json'
    r = requests.get(url.format(year))
    json = r.json()

    for item in json['MRData']['RaceTable']['Races']:
        try:
            qualifying['season'].append(int(item['season']))
        except:
            qualifying['season'].append(None)

        try:
            qualifying['round'].append(int(item['round']))
        except:
            qualifying['round'].append(None)

        try:
            qualifying['circuit_id'].append(item['Circuit']['circuitId'])
        except:
            qualifying['circuit_id'].append(None)

        try:
            qualifying['lat'].append(float(item['Circuit']['Location']['lat']))
        except:
            qualifying['lat'].append(None)

        try:
            qualifying['long'].append(float(item['Circuit']['Location']['long']))
        except:
            qualifying['long'].append(None)

        try:
            qualifying['country'].append(item['Circuit']['Location']['country'])
        except:
            qualifying['country'].append(None)

        try:
            qualifying['city'].append(item['Circuit']['Location']['locality'])
        except:
            qualifying['city'].append(None)

        try:
            qualifying['q_date'].append(item['Qualifying']['date'])
        except:
            qualifying['q_date'].append((datetime.datetime.strptime(item['date'], "%Y-%m-%d")-datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        
qualifying = pd.DataFrame(qualifying)
print(qualifying.shape)

"""get qualifying results"""

for index, row in qualifying.iterrows():
    url = 'https://ergast.com/api/f1/{}/{}/qualifying/1.json'.format(row['season'],row['round'])
    r = requests.get(url)
    json = r.json()

    for item in json['MRData']['RaceTable']['Races']:
      for race in item['QualifyingResults']:
        if race['Driver']['givenName']:
          qualifying.loc[index, 'pole_position'] = race['Driver']['givenName']+' '+race['Driver']['familyName']
        else:
          qualifying.loc[index, 'pole_position'] = None

"""get driver table"""

drivers = {'driver_id': [],
            'driver' : []}

url = 'https://ergast.com/api/f1/2022/drivers.json'
r = requests.get(url)
json = r.json()
num = 0

for item in json['MRData']['DriverTable']['Drivers']:
    try:
      drivers['driver'].append(item['givenName']+' '+item['familyName'])
      drivers['driver_id'].append(num)
      num += 1
    except:
      drivers['driver'].append(None)
      drivers['driver_id'].append(None)

drivers = pd.DataFrame(drivers)

drivers.to_json('drivers.json')

qualifying = pd.merge(qualifying, drivers, right_on=['driver'], left_on=['pole_position'], how='left')

"""get weather for qualifying date"""

for index, row in qualifying.iterrows():
  url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{}%2C{}/{}?unitGroup=metric&elements=temp%2Cwindspeed%2Cconditions&include=days&key={YOUR_KEY}&contentType=json'.format(row['lat'],row['long'],row['q_date'])
  r = requests.get(url)
  json = r.json()
  for item in json['days']:
        try:
            qualifying.loc[index, 'temp'] = item['temp']
        except:
            qualifying.loc[index, 'temp'] = None

        try:
            qualifying.loc[index, 'windspeed'] = item['windspeed']
        except:
            qualifying.loc[index, 'windspeed'] = None

        try:
            qualifying.loc[index, 'conditions'] = item['conditions']
        except:
            qualifying.loc[index, 'conditions'] = None

for index, row  in qualifying.iterrows():
  if row['conditions'].split(', ')[0] != 'Rain':
    qualifying.loc[index, 'rain'] = 'no_rain'
    qualifying.loc[index, 'weather'] = row['conditions'].split(', ')[0]
  else:
    qualifying.loc[index, 'rain'] = 'rain'
    try:
      qualifying.loc[index, 'weather'] = row['conditions'].split(', ')[1]
    except:
      qualifying.loc[index, 'weather'] = 'Precipitation'

"""transform data"""

qualifying = qualifying.sort_values(['season', 'round'])

qualifying = qualifying[:173]

qualifying = qualifying.dropna()

qualifying = qualifying.drop(['driver', 'pole_position', 'conditions', 'lat', 'long', 'country', 'city', 'q_date'], axis=1)

qualifying

qualifying.to_csv('qualifying_for_catboost.csv')

"""fitting"""

X = qualifying.drop("driver_id", axis=1)
y = qualifying["driver_id"]

label = drivers['driver_id'],
cat_features=[0,1,2,5,6]

X_train,X_test,y_train,y_test = train_test_split(X, 
                                                 y, 
                                                 test_size=0.3, 
                                                 random_state=43)

model = CatBoostClassifier(
    iterations=150,
    random_seed=43,
    loss_function='MultiClass'
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    verbose=False)

fast_model = CatBoostClassifier(
    random_seed=63,
    iterations=150,
    learning_rate=0.01,
    boosting_type='Plain',
    bootstrap_type='Bernoulli',
    subsample=0.5,
    one_hot_max_size=20,
    rsm=0.5,
    leaf_estimation_iterations=5,
    max_ctr_complexity=1,
    loss_function='MultiClass'
)

fast_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    verbose=False)

tunned_model = CatBoostClassifier(
    random_seed=63,
    iterations=1000,
    learning_rate=0.03,
    l2_leaf_reg=3,
    bagging_temperature=1,
    random_strength=1,
    one_hot_max_size=2,
    leaf_estimation_method='Newton',
    loss_function='MultiClass'
)

tunned_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    verbose=False)

best_model = CatBoostClassifier(
    random_seed=63,
    iterations=int(tunned_model.tree_count_ * 1.2),
    loss_function='MultiClass')

best_model.fit(
    X, y,
    cat_features=cat_features)

best_model.save_model('catboost_model')

import pickle

pickle.dump(list(X.columns), open('q_columns.sav', 'wb'))