
##!pip install requests

##!pip install CatBoost

import requests
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from catboost import Pool

"""get races info"""

races = {'season': [],
        'round': [],
        'circuit_id': [],
        'lat': [],
        'long': [],
        'country': [],
        'city':[],
        'date': [],
        'sprint' : []}

for year in list(range(2014,2023)):
    
    url = 'https://ergast.com/api/f1/{}.json'
    r = requests.get(url.format(year))
    json = r.json()

    for item in json['MRData']['RaceTable']['Races']:
        try:
            races['season'].append(int(item['season']))
        except:
            races['season'].append(None)

        try:
            races['round'].append(int(item['round']))
        except:
            races['round'].append(None)

        try:
            races['circuit_id'].append(item['Circuit']['circuitId'])
        except:
            races['circuit_id'].append(None)

        try:
            races['lat'].append(float(item['Circuit']['Location']['lat']))
        except:
            races['lat'].append(None)

        try:
            races['long'].append(float(item['Circuit']['Location']['long']))
        except:
            races['long'].append(None)

        try:
            races['country'].append(item['Circuit']['Location']['country'])
        except:
            races['country'].append(None)

        try:
            races['city'].append(item['Circuit']['Location']['locality'])
        except:
            races['city'].append(None)

        try:
            races['date'].append(item['date'])
        except:
            races['date'].append(None)

        try:
            item['Sprint']['date'] is not None
        except:
            races['sprint'].append(0)
        else:
            races['sprint'].append(1)
        
races = pd.DataFrame(races)
print(races.shape)

"""get races results"""

for index, row in races.iterrows():
    url = 'https://ergast.com/api/f1/{}/{}/results/1.json'.format(row['season'],row['round'])
    r = requests.get(url)
    json = r.json()

    for race in json['MRData']['RaceTable']['Races']:
      if race['Results'][0]['Driver']['givenName']:
        races.loc[index, 'driver'] = race['Results'][0]['Driver']['givenName']+' '+race['Results'][0]['Driver']['familyName']
      else:
        races.loc[index, 'driver'] = None

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

drivers.to_csv('drivers.csv')

races = pd.merge(races, drivers, on=['driver'], how='left')

races = races.drop(['driver'], axis=1)

races = races.dropna()

races = races.rename({'driver_id':'position_1'}, axis='columns')

"""get qualifying results"""

for index, row in races.iterrows():
    url = 'https://ergast.com/api/f1/{}/{}/qualifying/1.json'.format(row['season'],row['round'])
    r = requests.get(url)
    json = r.json()

    for item in json['MRData']['RaceTable']['Races']:
      for race in item['QualifyingResults']:
        if race['Driver']['givenName']:
          races.loc[index, 'driver'] = race['Driver']['givenName']+' '+race['Driver']['familyName']

        else:
          races.loc[index, 'driver'] = None

races = pd.merge(races, drivers, on=['driver'], how='left')

races = races.drop(['driver'], axis=1)

races = races.rename({'driver_id':'q_p1'}, axis='columns')

races = races.fillna(0)

for index, row in races.iterrows():
    url = 'https://ergast.com/api/f1/{}/{}/qualifying/2.json'.format(row['season'],row['round'])
    r = requests.get(url)
    json = r.json()

    for item in json['MRData']['RaceTable']['Races']:
      for race in item['QualifyingResults']:
        if race['Driver']['givenName']:
          races.loc[index, 'driver'] = race['Driver']['givenName']+' '+race['Driver']['familyName']

        else:
          races.loc[index, 'driver'] = None

races = pd.merge(races, drivers, on=['driver'], how='left')

races = races.drop(['driver'], axis=1)

races = races.rename({'driver_id':'q_p2'}, axis='columns')

races = races.fillna(0)



for index, row in races.iterrows():
    url = 'https://ergast.com/api/f1/{}/{}/qualifying/3.json'.format(row['season'],row['round'])
    r = requests.get(url)
    json = r.json()

    for item in json['MRData']['RaceTable']['Races']:
      for race in item['QualifyingResults']:
        if race['Driver']['givenName']:
          races.loc[index, 'driver'] = race['Driver']['givenName']+' '+race['Driver']['familyName']

        else:
          races.loc[index, 'driver'] = None

races = pd.merge(races, drivers, on=['driver'], how='left')

races = races.drop(['driver'], axis=1)

races = races.rename({'driver_id':'q_p3'}, axis='columns')

races = races.fillna(0)

"""get sprint qualifying results"""

for index, row in races.iterrows():
    if row['sprint'] == 1:
        url = 'https://ergast.com/api/f1/{}/{}/sprint/1.json'.format(row['season'],row['round'])
        r = requests.get(url)
        json = r.json()

        for item in json['MRData']['RaceTable']['Races']:
          for race in item['SprintResults']:
            if race['Driver']['givenName']:
              races.loc[index, 'driver'] = race['Driver']['givenName']+' '+race['Driver']['familyName']
            else:
              races.loc[index, 'driver'] = None

races = pd.merge(races, drivers, on=['driver'], how='left')

races = races.drop(['driver'], axis=1)

races = races.rename({'driver_id':'sprint_p1'}, axis='columns')

races['sprint_p1'] = races['sprint_p1'].fillna(0)

"""get weather for racing date"""

for index, row in races.iterrows():
  url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{}%2C{}/{}?unitGroup=metric&elements=temp%2Cwindspeed%2Cconditions&include=days&key=H6GND588FW74NLWNW6P95HV83&contentType=json'.format(row['lat'],row['long'],row['date'])
  r = requests.get(url)
  json = r.json()
  for item in json['days']:
        try:
            races.loc[index, 'temp'] = item['temp']
        except:
            races.loc[index, 'temp'] = None

        try:
            races.loc[index, 'windspeed'] = item['windspeed']
        except:
            races.loc[index, 'windspeed'] = None

        try:
            races.loc[index, 'conditions'] = item['conditions']
        except:
            races.loc[index, 'conditions'] = None

for index, row  in races.iterrows():
  if row['conditions'].split(', ')[0] != 'Rain':
    races.loc[index, 'rain'] = 'no_rain'
    races.loc[index, 'weather'] = row['conditions'].split(', ')[0]
  else:
    races.loc[index, 'rain'] = 'rain'
    try:
      races.loc[index, 'weather'] = row['conditions'].split(', ')[1]
    except:
      races.loc[index, 'weather'] = 'Precipitation'

"""transform data"""

races = races.sort_values(['season', 'round'])

races = races.astype({"q_p1": "Int64","q_p2": "Int64","q_p3": "Int64","sprint_p1": "Int64"})

races = races.drop(['lat', 'long', 'country', 'city', 'date', 'conditions'], axis=1)

races

races.to_csv('races_for_catboost.csv')

"""fitting"""

X = races.drop("position_1", axis=1)
y = races["position_1"]

label = drivers['driver_id'],
cat_features=[0,1,2,3,4,5,6,7,10,11]

X_train,X_test,y_train,y_test = train_test_split(X, 
                                                 y, 
                                                 test_size=0.3, 
                                                 random_state=43)

model = CatBoostClassifier(
    iterations=150,
    random_seed=43,
    loss_function='MultiClass'
)

X

pool = Pool(X.values, y.values, cat_features=cat_features)

model.fit(
    pool,
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
    pool,
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
    pool,
    verbose=False)

best_model = CatBoostClassifier(
    random_seed=63,
    iterations=int(tunned_model.tree_count_ * 1.2),
    loss_function='MultiClass')

best_model.fit(
    pool,
    verbose=False)

best_model.save_model('catboost_model_races')

best_model.predict(['2022', '14', 'spa', '0', '7.0', '18.0', '2.0', '0', '14.1',
        '12.2', 'no_rain', 'Partially cloudy'])

import pickle

pickle.dump(list(X.columns), open('r_columns.sav', 'wb'))

