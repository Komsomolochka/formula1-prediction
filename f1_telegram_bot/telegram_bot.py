# -*- coding: utf-8 -*-

import telebot
import requests
import shutil
import subprocess
import os.path
from telebot import types
import pickle
from datetime import datetime
import requests
import json
import collections
import numpy as np
import catboost
from catboost import CatBoostClassifier
from environs import Env


env = Env()
env.read_env()

API_TOKEN = env.str("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(API_TOKEN, parse_mode="Markdown")

datetime.now().strftime('%Y-%m-%d')
year = datetime.now().strftime('%Y')
today = datetime.now().strftime('%Y-%m-%d')

url = 'https://ergast.com/api/f1/{}.json'.format(year)
r = requests.get(url)
races = r.json()['MRData']['RaceTable']['Races']


q_model = CatBoostClassifier()
q_model.load_model('catboost_model', format='cbm')
X_qual = pickle.load(open('q_columns.sav', 'rb'))
X_q = collections.OrderedDict.fromkeys(X_qual)

with open('drivers.json') as file:
    drivers = json.load(file)

r_model = CatBoostClassifier()
r_model.load_model('catboost_model_races', format='cbm')
X_race = pickle.load(open('r_columns.sav', 'rb'))
X_r = collections.OrderedDict.fromkeys(X_race)


def convert_time(z_time):
    z_time = z_time.replace("Z", "")
    str_time = z_time.split(":")
    time = str(int(str_time[0]) + 1)
    return time + ":" + str_time[1]

def get_weather(lat, long, date):
    w_url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{}%2C{}/{}?unitGroup=metric&elements=temp%2Cwindspeed%2Cconditions&include=days&key=H6GND588FW74NLWNW6P95HV83&contentType=json'.format(lat, long, date)
    w_r = requests.get(w_url)
    return w_r.json()

def feel_x(q_weather, X_q, call_data):
    X_q['season'] = int(races[call_data]['season'])
    X_q['round'] = int(races[call_data]['round'])
    X_q['circuit_id'] = races[call_data]['Circuit']['circuitId']
    X_q['temp'] = q_weather['days'][0]['temp']
    X_q['windspeed'] = q_weather['days'][0]['windspeed']

    try:
        q_weather['days'][0]['conditions'].split(', ')[1]
        X_q['weather'] = q_weather['days'][0]['conditions']
        X_q['rain'] = 'rain'
    except:
        if q_weather['days'][0]['conditions'] == 'Rain':
            X_q['weather'] = 'precipitation'
            X_q['rain'] = 'rain'
        else:
            X_q['rain'] = 'no_rain'
            X_q['weather'] = q_weather['days'][0]['conditions']

    for k, v in X_q.items():
        if v is None:
            X_q[k] = 0
            
    X = np.array(list(X_q.values())).reshape(1, -1)
    return X

def feel_x_r(r_weather, X_r, call_data, q_p1, q_p2, q_p3):
    X_r['season'] = int(races[call_data]['season'])
    X_r['round'] = int(races[call_data]['round'])
    X_r['circuit_id'] = races[call_data]['Circuit']['circuitId']
    X_r['temp'] = r_weather['days'][0]['temp']
    X_r['q_p1'] = q_p1
    X_r['q_p2'] = q_p2
    X_r['q_p3'] = q_p3
    X_r['windspeed'] = r_weather['days'][0]['windspeed']

    try:
        r_weather['days'][0]['conditions'].split(', ')[1]
        X_r['weather'] = r_weather['days'][0]['conditions']
        X_r['rain'] = 'rain'
    except:
        if r_weather['days'][0]['conditions'] == 'Rain':
            X_r['weather'] = 'precipitation'
            X_r['rain'] = 'rain'
        else:
            X_r['rain'] = 'no_rain'
            X_r['weather'] = r_weather['days'][0]['conditions']

    for k, v in X_r.items():
        if v is None:
            X_r[k] = 0
            
    X_for_race = np.array(list(X_r.values())).reshape(1, -1)
    return X_for_race


@bot.message_handler(commands=["start"])
def default_test(message):

    keyboard = types.InlineKeyboardMarkup()
    for i, race in enumerate(races):
      if race['date'] >= today:
        keyboard.add(types.InlineKeyboardButton(text=race['raceName'], callback_data=str(i)))
    bot.send_message(message.chat.id, "Choose a race:", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):

    call_data = int(call.data)
    name = races[int(call.data)]['Circuit']['circuitName']
    country = races[int(call.data)]['Circuit']['Location']['country']
    city = races[int(call.data)]['Circuit']['Location']['locality']
    q_date = races[int(call.data)]['Qualifying']['date']
    q_time = races[int(call.data)]['Qualifying']['time']
    r_date = races[int(call.data)]['date']
    r_time = races[int(call.data)]['time']
    lat = races[int(call.data)]['Circuit']['Location']['lat']
    long = races[int(call.data)]['Circuit']['Location']['long']
    q_weather = get_weather(lat, long, q_date)
    r_weather = get_weather(lat, long, r_date)
    q_conditions = q_weather['days'][0]['conditions']
    q_temp = q_weather['days'][0]['temp']
    r_conditions = r_weather['days'][0]['conditions']
    r_temp = r_weather['days'][0]['temp']

    X = feel_x(q_weather, X_q, call_data)
    pred = q_model.predict_proba(X)
    classes = q_model.classes_
    q_dict = dict(zip(classes, pred[0]))
    q_sort_keys = sorted(q_dict, key=q_dict.get)
    q_p1 = q_sort_keys[-1]
    q_p2 = q_sort_keys[-2]
    q_p3 = q_sort_keys[-3]
    q1 = drivers['driver'][str(int(q_p1))]
    q2 = drivers['driver'][str(int(q_p2))]
    q3 = drivers['driver'][str(int(q_p3))]
    q1_res = round(q_dict[q_p1]*100)
    q2_res = round(q_dict[q_p2]*100)
    q3_res = round(q_dict[q_p3]*100)

    X_for_race = feel_x_r(r_weather, X_r, call_data, q_p1, q_p2, q_p3)
    r_pred = r_model.predict_proba(X_for_race)
    r_classes = r_model.classes_
    r_dict = dict(zip(r_classes, r_pred[0]))
    r_sort_keys = sorted(r_dict, key=r_dict.get)
    r_p1 = r_sort_keys[-1]
    r_p2 = r_sort_keys[-2]
    r_p3 = r_sort_keys[-3]
    r1 = drivers['driver'][str(int(r_p1))]
    r2 = drivers['driver'][str(int(r_p2))]
    r3 = drivers['driver'][str(int(r_p3))]
    r1_res = round(r_dict[r_p1]*100)
    r2_res = round(r_dict[r_p2]*100)
    r3_res = round(r_dict[r_p3]*100)

    if 'Sprint' in races[int(call.data)]:
        s_date = races[int(call.data)]['Sprint']['date']
        s_time = races[int(call.data)]['Sprint']['time']
        s_weater = get_weather(lat, long, s_date)
        s_conditions = s_weater['days'][0]['conditions']
        s_temp = s_weater['days'][0]['temp']   
    else:
        s_date = None

    if s_date:
        answer = ("*{}*\n{}, {}\n\n*Qualifying*:\nDate: {}\nTime (UTC +3): {}\nWeather:  {}   {}°C\n\n*Predicted pole position:*\n{}: {}%\n{}: {}%\n{}: {}%\n\n*Sprint*:\nDate: {}\nTime (UTC +3): {}\nWeather:  {}   {}°C\n\n*Race*:\nDate: {}\nTime (UTC +3): {}\nWeather:  {}   {}°C\n\n*Predicted winner:*\n{}: {}%\n{}: {}%\n{}: {}%".format(name, country, city, q_date, convert_time(q_time), q_conditions, q_temp, q1, q1_res, q2, q2_res, q3, q3_res, s_date, convert_time(s_time), s_conditions, s_temp, r_date, convert_time(r_time), r_conditions, r_temp, r1, r1_res, r2, r2_res, r3, r3_res))
    else:  
        answer = ("*{}*\n{}, {}\n\n*Qualifying*:\nDate: {}\nTime (UTC +3): {}\nWeather:  {}   {}°C\n\n*Predicted pole position:*\n{}: {}%\n{}: {}%\n{}: {}%\n\n*Race*:\nDate: {}\nTime (UTC +3): {}\nWeather:  {}   {}°C\n\n*Predicted winner:*\n{}: {}%\n{}: {}%\n{}: {}%".format(name, country, city, q_date, convert_time(q_time), q_conditions, q_temp, q1, q1_res, q2, q2_res, q3, q3_res, r_date, convert_time(r_time), r_conditions, r_temp, r1, r1_res, r2, r2_res, r3, r3_res))
    
    bot.send_message(call.message.chat.id, answer)


if __name__ == "__main__":
    bot.polling()
