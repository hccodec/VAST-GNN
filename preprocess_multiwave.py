#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Social Awareness-Based Graph Neural Network (SAB-GNN)
# This code trains and tests the SAB-GNN model for the COVID-19 infection prediction in Tokyo
# Step 1: read and pack the training and testing data
# Step 2: training epoch, training process, testing
# Step 3: build the model = spatial module + social awareness decay + temporal module
# Step 4: training and testing
# Step 5: evaluation
# Step 6: visualization

from datetime import datetime, timedelta

class datetime(datetime):
    def __add__(self, other):
        if isinstance(other, int): return self + timedelta(days=other)
        else: return super().__add__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, int): return self + timedelta(days=-other)
        else:
            return super().__sub__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

def str2date(f): return f if isinstance(f, datetime) else datetime.strptime(f, '%Y%m%d')
def date2str(f): return f if isinstance(f, str) else datetime.strftime(f, '%Y%m%d')

import os
os.environ['CUDA_AVAILABLE_DEVICES'] = '7,8,9'
import csv
import json
import copy
import time
import random
import string
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torch.nn.functional as F
from tqdm.auto import tqdm

from model.sab_gnn import SpecGCN
from model.sab_gnn import SpecGCN_LSTM
#hyperparameter for the setting
# X_day, Y_day = 21,7
# X_day, Y_day = 21,14
X_day, Y_day = 21,7
# START_DATE, END_DATE = '20200414','20210207' # possible wave 3
START_DATE, END_DATE = '20200720','20210515' # possible wave 4

# START_DATE, END_DATE = '20201210', '20210207' # 3rd wave
# START_DATE, END_DATE = '20210317', '20210515' # 4th wave

WINDOW_SIZE = 7

#hyperparameter for the learning
DROPOUT, ALPHA = 0.50, 0.20
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = 100, 8, 0.0001
HIDDEN_DIM_1, OUT_DIM_1, HIDDEN_DIM_2 = 6,4,3
infection_normalize_ratio = 100.0
web_search_normalize_ratio = 100.0
train_ratio = 0.7
validate_ratio = 0.1
# SEED = 5
# torch.manual_seed(SEED)
r = random.random
# random.seed(5)
zone_indicies, text_indicies = {}, {}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("HCCODEC: Running on ", device)

text_list = list(['痛み', '頭痛', '咳', '下痢', 'ストレス', '不安',
                    '腹痛', 'めまい', '吐き気', '嘔吐', '筋肉痛', '動悸',
                    '副鼻腔炎', '発疹', 'くしゃみ', '倦怠感', '寒気', '脱水',
                    '中咽頭', '関節痛', '不眠症', '睡眠障害', '鼻漏', '片頭痛',
                    '多汗症', 'ほてり', '胸痛', '発汗', '無気力', '呼吸困難',
                    '喘鳴', '目の痛み', '体の痛み', '無嗅覚症', '耳の痛み',
                    '錯乱', '見当識障害', '胸の圧迫感', '鼻の乾燥', '耳感染症',
                    '味覚消失', '上気道感染症', '眼感染症', '食欲減少'])

symptoms_zh = ['疼痛', '头痛', '咳嗽', '腹泻', '压力', '焦虑',
                '腹痛', '头晕', '恶心', '呕吐', '肌肉疼痛', '心悸',
                '鼻窦炎', '皮疹', '打喷嚏', '疲劳', '寒冷', '脱水',
                '咽喉炎', '关节痛', '失眠', '睡眠障碍', '流鼻涕', '偏头痛',
                '多汗', '潮红', '胸痛', '出汗', '无精打采', '呼吸困难',
                '喘鸣', '眼痛', '身体疼痛', '无嗅', '耳痛',
                '混乱', '迷失方向', '胸闷', '鼻干', '耳感染',
                '味觉丧失', '上呼吸道感染', '眼感染', '食欲减退']

symptoms_en = ['Pain', 'Headache', 'Cough', 'Diarrhea', 'Stress', 'Anxiety',
                'Abdominal pain', 'Dizziness', 'Nausea', 'Vomiting', 'Muscle pain', 'Palpitations',
                'Sinusitis', 'Rash', 'Sneezing', 'Fatigue', 'Chills', 'Dehydration',
                'Pharyngitis', 'Joint pain', 'Insomnia', 'Sleep disorder', 'Rhinorrhea', 'Migraine',
                'Hyperhidrosis', 'Flushing', 'Chest pain', 'Sweating', 'Apathy', 'Shortness of breath',
                'Wheezing', 'Eye pain', 'Body pain', 'Anosmia', 'Ear pain',
                'Confusion', 'Disorientation', 'Chest pressure', 'Dry nose', 'Ear infection',
                'Loss of taste', 'Upper respiratory tract infection', 'Eye infection', 'Decreased appetite']

text_list = symptoms_en

# # Step 1: read and pack the training and testing data

# In[2]:


#1.total period (mobility+text): 
#from 20200201 to 20210620: (29+31+30+31+30+31+31+30+31+30+31)+(31+28+31+30+31+20)\
#= 335 + 171 = 506;
#2.number of zones: 23;
#3.infection period:
#20200331 to 20210620: (1+30+31+30+31+31+30+31+30+31)+(31+28+31+30+31+20) = 276 + 171 = 447.

#1. Mobility: functions 1.2 to 1.7
#2. Text:  functions 1.8 to 1.14
#3. InfectionL:  functions 1.15 
#4. Preprocess:  functions 1.16 to 1.24
#5. Learn:  functions 1.25 to 1.26

#function 1.1
#get the central areas of Tokyo (e.g., the Special wards of Tokyo)
#return: a 23 zone shapefile
def read_tokyo_23(path=""):
    # folder = "tokyo_23" 
    # file = "tokyo_23zones.shp"
    # path = os.path.join(folder,file) 
    if path == "": path = "tokyo_shapefile/tokyo.shp"
    data = gpd.read_file(path)   
    return data

##################1.Mobility#####################
#function 1.2
#get the average of two days' mobility (infection) records
def mob_inf_average(data, key1, key2):
    new_record = dict()
    record1, record2 = data[key1], data[key2]
    for i in record1:
        if i in record2:
            new_record[i] = (record1[i] + record2[i]) / 2.
    return new_record

#function 1.3
#get the average of multiple days' mobility (infection) records
def mob_inf_average_multiple(data, keyList):
    new_record = dict()
    num_day = len(keyList)
    for i in range(num_day):
        record = data[keyList[i]]
        for zone_id in record:
            if zone_id not in list(new_record.keys()):
                new_record[zone_id] = record[zone_id]
            else:
                new_record[zone_id] += record[zone_id]
    for new_record_key in new_record:
        new_record[new_record_key] = new_record[new_record_key] * 1. / num_day
    return new_record

#function 1.5
#smooth the mobility (infection) data using the neighborhood average
#under a given window size 
#dateList: [20200101, 20200102, ..., 20211231]
def mob_inf_smooth(data, window_size, dateList, hint):
    data_copy = copy.copy(data)
    data_key_list = list(data_copy.keys())
    qbar1 = progress_indicator(data_key_list, desc=f"Smoothing {hint}")
    for data_key in qbar1:
        if data_key not in dateList: continue
        left = int(max(dateList.index(data_key) - (window_size-1) / 2, 0))
        right = int(min(dateList.index(data_key) + (window_size-1) / 2, len(dateList) - 1))
        potential_neighbor = dateList[left:right+1]
        neighbor_data_key = list(set(data_key_list).intersection(set(potential_neighbor)))
        data_average = mob_inf_average_multiple(data_copy, neighbor_data_key)
        data[data_key] = data_average
    return data

#function 1.6
#set the mobility (infection) of one day as zero
def mob_inf_average_null(data, key1, key2):
    new_record = dict()
    record1, record2 = data[key1], data[key2]
    for i in record1:
        if i in record2:
            new_record[i] = 0
    return new_record

#function 1.7
#read the mobility data from "mobility_feature_20200201.json"...
#return: all_mobility:{"20200201":{('123','123'):12345,...},...}
#20200201 to 20210620: 506 days
def read_mobility_data(jcode23):
    all_mobility = dict()
    mobilityFilePath = "mobility"
    mobilityNameList = os.listdir(mobilityFilePath)

    # print('Reading mobility data...')
    qbar1 = progress_indicator(range(len(mobilityNameList)), desc='Reading mobility data')

    for i in qbar1:
        day_mobility = dict()
        file_name = mobilityNameList[i] 
        if "20" in file_name:
            day = (file_name.split("_")[3]).split(".")[0]  #get the day
            file_path = mobilityFilePath + '/' + file_name
            with open(file_path,) as f:
                df_file = json.load(f)   #read the mobility file
            
            qbar2 = tqdm(df_file, leave=False, position=1)

            for key in qbar2:
                origin, dest = key.split("_")
                if origin in jcode23 and dest in jcode23:
                    _v = 0. if origin == dest else df_file[key] #ignore the inner-zone flow
                    day_mobility[(origin, dest)] = _v

            all_mobility[day] = day_mobility
    #missing data
    # all_mobility["20201128"] = mob_inf_average(all_mobility,"20201127","20201129")
    # all_mobility["20210104"] = mob_inf_average(all_mobility, "20210103","20210105")
    interpolate(all_mobility, mob_inf_average)
    # all_mobility = interpolate(all_mobility, mob_inf_average)
    return all_mobility

##################2.Text#####################
#function 1.8
#get the average of two days' infection records
def text_average(data, key1, key2):
    new_record = dict()
    record1, record2 = data[key1], data[key2]
    for i in record1:
        if i in record2:
            zone_record1, zone_record2 = record1[i], record2[i]
            new_zone_record = dict()
            for j in zone_record1:
                if j in zone_record2:
                    new_zone_record[j] = (zone_record1[j] + zone_record2[j])/2.0
            new_record[i] = new_zone_record
    return new_record

#function 1.9
#get the average of multiple days' text records
def text_average_multiple(data, keyList):
    new_record = dict()
    num_day = len(keyList)
    for i in range(num_day):
        record = data[keyList[i]]
        for zone_id in record:                           #zone_id
            if zone_id not in new_record:
                new_record[zone_id] = dict()   
            for j in record[zone_id]:                    #symptom
                if j not in new_record[zone_id]:
                    new_record[zone_id][j] = record[zone_id][j]
                else: 
                    new_record[zone_id][j] += record[zone_id][j]
    for zone_id in new_record:
        for j in new_record[zone_id]:
            new_record[zone_id][j] = new_record[zone_id][j]*1.0/num_day 
    return new_record

#function 1.10
#smooth the text data using the neighborhood average
#under a given window size 
def text_smooth(data, window_size, dateList, hint):
    data_copy = copy.copy(data)
    data_key_list = list(data_copy.keys())
    qbar1 = progress_indicator(data_key_list, desc=f"Smoothing {hint}")
    for data_key in qbar1:
        left = int(max(dateList.index(data_key)-(window_size-1)/2, 0))
        right = int(min(dateList.index(data_key)+(window_size-1)/2, len(dateList)-1))
        potential_neighbor = dateList[left:right+1]
        neighbor_data_key = list(set(data_key_list).intersection(set(potential_neighbor)))
        data_average = text_average_multiple(data_copy, neighbor_data_key)
        data[data_key] =  data_average
    return data

#function 1.11
#read the number of user points
def read_point_json():
    with open('user_point/mobility_user_point.json') as point1:
        user_point1 = json.load(point1)
    with open('user_point/mobility_user_point_20210812.json') as point2:
        user_point2 = json.load(point2)
    user_point_all = dict()
    for i in user_point1:
        user_point_all[i] = user_point1[i]
    for i in user_point2:
        user_point_all[i] = user_point2[i]
    user_point_all["20201128"] = user_point_all["20201127"]  #data missing
    user_point_all["20210104"] = user_point_all["20210103"]  #data missing
    return user_point_all

#function 1.12
#normalize the text search by the number of user points.
def normalize_text_user(all_text, user_point_all):
    for day in all_text:
        if day in user_point_all:
            num_user = user_point_all[day]["num_user"]
            all_text_day_new = dict()
            all_text_day = all_text[day]
            for zone in all_text_day:
                if zone not in all_text_day_new:
                    all_text_day_new[zone] = dict()
                for sym in all_text_day[zone]:
                    all_text_day_new[zone][sym] = all_text_day[zone][sym]*1.0/num_user
            all_text[day] = all_text_day_new
    return all_text
    
#function 1.13
#read the text data
#20200201 to 20210620: 506 days
#all_text = {"20200211":{"123":{"code":3,"fever":2,...},...},...}
def read_text_data(jcode23):
    all_text = dict()
    textFilePath = "text"
    textNameList = os.listdir(textFilePath)

    # print('Reading text data...')
    qbar1 = progress_indicator(range(len(textNameList)), desc='Reading text data')

    for i in qbar1:
        day_text = dict()
        file_name = textNameList[i]
        if "20" in file_name:
            day = (file_name.split("_")[3]).split(".")[0]
            file_path = textFilePath + "/" + file_name
            with open(file_path,) as f: df_file = json.load(f)   #read the mobility file
            new_dict = dict()
            
            qbar2 = tqdm(df_file, leave=False, position=-1)

            for key in qbar2:
                if key in jcode23:
                    new_dict[key] = {key1:df_file[key][key1]*1.0*web_search_normalize_ratio for key1 in df_file[key]}
                    #new_dict[key] = df_file[key]*WEB_SEARCH_RATIO
            all_text[day] = new_dict
    # all_text["20201030"] = text_average(all_text, "20201029", "20201031") #data missing
    interpolate(all_text, text_average)
    # all_text = interpolate(all_text, text_average)
    return all_text

#function 1.14
#perform the min-max normalization for the text data.
def min_max_text_data(all_text,jcode23):
    #calculate the min_max
    #region_key: sym: [min,max]
    # text_list = list(['痛み', '頭痛', '咳', '下痢', 'ストレス', '不安',
    #                   '腹痛', 'めまい', '吐き気', '嘔吐', '筋肉痛', '動悸',
    #                   '副鼻腔炎', '発疹', 'くしゃみ', '倦怠感', '寒気', '脱水',
    #                   '中咽頭', '関節痛', '不眠症', '睡眠障害', '鼻漏', '片頭痛',
    #                   '多汗症', 'ほてり', '胸痛', '発汗', '無気力', '呼吸困難',
    #                   '喘鳴', '目の痛み', '体の痛み', '無嗅覚症', '耳の痛み',
    #                   '錯乱', '見当識障害', '胸の圧迫感', '鼻の乾燥', '耳感染症',
    #                   '味覚消失', '上気道感染症', '眼感染症', '食欲減少'])
    
    total = len(all_text) * len(jcode23) * len(text_list)
    desc = lambda f: f'Min-max normalizing the text data ({f})'

    qbar = progress_indicator(total=total * 2 + len(text_list) * len(jcode23),
                              show_total=False)
    
    qbar.desc = desc('Initializing')
    region_sym_min_max = dict()
    #initialize
    for area in jcode23:
        region_sym_min_max[area] = dict()
        for sym in text_list:
            region_sym_min_max[area][sym] = [1000000,0]  #min, max
            qbar.update()

    qbar.desc = desc('Updating')
    #update
    for day in all_text:
        for area in jcode23:
            for sym in text_list:
                if sym in all_text[day][area]:
                    count = all_text[day][area][sym]
                    if count < region_sym_min_max[area][sym][0]: region_sym_min_max[area][sym][0] = count
                    if count > region_sym_min_max[area][sym][1]: region_sym_min_max[area][sym][1] = count
                qbar.update()

    qbar.desc = desc('Normalizing')
    #normalize
    #print ("region_sym_min_max",region_sym_min_max)
    for area in jcode23:
        for sym in text_list:
            min_count, max_count = region_sym_min_max[area][sym]
            for day in all_text:
                if sym in all_text[day][area]:
                    all_text[day][area][sym] = 1 \
                        if max_count - min_count == 0 \
                        else (all_text[day][area][sym] - min_count) * 1. / (max_count - min_count)
                        #print("all_text[day][key][sym]",all_text[day][key][sym])
                qbar.update()
    return all_text

##################3.Infection#####################
#function 1.15
#read the infection data
#20200331 to 20210620: (1+30+31+30+31+31+30+31+30+31)+(31+28+31+30+31+20) = 276 + 171 = 447.
#all_infection = {"20200201":{"123":1,"123":2}}
def read_infection_data(jcode23):
    all_infection = dict()
    infection_path = "covid_case/patient.json"
    # infection_path = "patient_20210725.json"

    # print('Reading infection data...')

    with open(infection_path,) as f: df_file = json.load(f)   #read the mobility file

    qbar1 = progress_indicator(df_file, desc='Reading infection data')

    for zone_id in qbar1:

        qbar2 = tqdm(df_file[zone_id], leave=False)

        for one_day in qbar2:
            # year, month, day = one_day.split("/")
            # if len(month) == 1: month = "0" + month
            # if len(day) == 1: day = "0" + day
            # new_date = year + month + day
            # assert new_date == "{:s}{:0>2s}{:0>2s}".format(*one_day.split("/"))
            new_date = "{:s}{:0>2s}{:0>2s}".format(*one_day.split("/"))

            if str(zone_id[0:5]) in jcode23: # 13101* in 13101 can be seen
                area, _res = zone_id[0:5], df_file[zone_id][one_day] * (1. / infection_normalize_ratio)
                if new_date not in all_infection:
                    all_infection[new_date] = {area: _res}
                else:
                    all_infection[new_date][area] = _res
    #missing
    # date_list = [str(20200316+i) for i in range(15)]
    # assert date_list == generateDates('20200316', '20200330')
    date_list = generateDates('20200316', '20200330')
    # for date in date_list:
    #     all_infection[date] = mob_inf_average(all_infection,'20200401','20200401')
    _res = mob_inf_average(all_infection,'20200401','20200401')
    for date in date_list: all_infection[date] = _res
    # all_infection['20200514'] = mob_inf_average(all_infection,'20200513','20200515')
    # all_infection['20200519'] = mob_inf_average(all_infection,'20200518','20200520')
    # all_infection['20200523'] = mob_inf_average(all_infection,'20200522','20200524')
    # all_infection['20200530'] = mob_inf_average(all_infection,'20200529','20200601')
    # all_infection['20200531'] = mob_inf_average(all_infection,'20200529','20200601')
    # all_infection['20201231'] = mob_inf_average(all_infection,'20201230','20210101')
    # all_infection['20210611'] = mob_inf_average(all_infection,'20210610','20210612')
    interpolate(all_infection, mob_inf_average)
    # all_infection = interpolate(all_infection, mob_inf_average)
    #outlier
    all_infection['20200331'] = mob_inf_average(all_infection, '20200401', '20200401')
    all_infection['20200910'] = mob_inf_average(all_infection, '20200909', '20200912')
    all_infection['20200911'] = mob_inf_average(all_infection, '20200909', '20200912')
    all_infection['20200511'] = mob_inf_average(all_infection, '20200510', '20200512')
    all_infection['20201208'] = mob_inf_average(all_infection, '20201207', '20201209')
    all_infection['20210208'] = mob_inf_average(all_infection, '20210207', '20210209')
    all_infection['20210214'] = mob_inf_average(all_infection, '20210213', '20210215')
    #calculate the subtraction
    all_infection_subtraction = dict()
    all_infection_subtraction['20200331'] = all_infection['20200331']
    all_keys = list(all_infection.keys())
    all_keys.sort()
    for i in range(len(all_keys) - 1):
        record = dict()
        for j in all_infection[all_keys[i + 1]]:
            record[j] = all_infection[all_keys[i + 1]][j] - all_infection[all_keys[i]][j]
        all_infection_subtraction[all_keys[i + 1]] = record
    return all_infection_subtraction, all_infection

##################4.Preprocess#####################
#function 1.16
#ensemble the mobility, text, and infection.
#all_mobility = {"20200201":{('123','123'):12345,...},...}
#all_text = {"20200201":{"123":{"cold":3,"fever":2,...},...},...}
#all_infection = {"20200316":{"123":1,"123":2}}
#all_original = {"0":[[mobility_1,text_1, ..., mobility_x_day,text_x_day], [infection_1,...,infection_y_day],[infection_1,...,infection_x_day]],0}
#x_days, y_days: use x_days to predict y_days
def ensemble(all_mobility, all_text, all_infection, x_days, y_days, all_day_list):
    all_original = dict()
    for j in range(len(all_day_list) - x_days - y_days + 1):
        x_sample, y_sample, x_sample_infection = list(), list(), list()                   
        #add the data from all_day_list[0+j] to all_day_list[x_days-1+j]
        for k in range(x_days):
            day = all_day_list[k + j]
            x_sample.append(all_mobility[day])
            x_sample.append(all_text[day])  
            x_sample_infection.append(all_infection[day])             #concatenate with the infection data                       
        #add the data from all_day_list[x_days+j] to all_day_list[x_days+y_day-1+j]
        for k in range(y_days):
            day = all_day_list[x_days + k + j]
            y_sample.append(all_infection[day]) 
        all_original[str(j)] = [x_sample, y_sample, x_sample_infection, j]                          
    return all_original

#function 1.17
#split the data by train/validate/test = train_ratio/validation_ratio/(1-train_ratio-validation_ratio)
def split_dataset_sequential(all_original, train_ratio, validation_ratio):
    n = len(all_original.keys())
    n_train, n_validate = round(n * train_ratio), round(n * validation_ratio)

    train_value = [all_original[str(k)] for k in range(n_train)]
    validate_value = [all_original[str(k)] for k in range(n_train, n_train + n_validate)]
    test_value = [all_original[str(k)] for k in range(n_train + n_validate, n)]

    return train_value, validate_value, test_value

##function 1.18
#the second data split method
#split the data by train/validate/test = train_ratio/validation_ratio/(1-train_ratio-validation_ratio)
def split_dataset_modulo_9(all_original, train_ratio, validation_ratio):
    n = len(all_original.keys())
    n_train, n_validate = round(n * train_ratio), round(n * validation_ratio)
    
    train_validate_indicies, test_indicies = range(n_train + n_validate), range(n_train + n_validate, n)
    train_indicies, validate_indicies = [], []
    for k in train_validate_indicies:
        validate_indicies.append(k) if k % 9 == 8 else train_indicies.append(k)

    train_value = [all_original[str(k)] for k in train_indicies]
    validate_value = [all_original[str(k)] for k in validate_indicies]
    test_value = [all_original[str(k)] for k in test_indicies]

    return train_value, validate_value, test_value, train_indicies, validate_indicies

##function 1.19
#the third data split method
#split the data by train/validate/test = train_ratio/validation_ratio/(1-train_ratio-validation_ratio)
def split_dataset_backwards_even_index(all_original, train_ratio, validation_ratio):
    '''
    从总数据的后 2*len_validate 条，将 index 为偶数的挑选为验证集
    '''
    n = len(all_original.keys())
    n_train, n_validate = round(n * train_ratio), round(n * validation_ratio)
    
    train_validate_indicies, test_indicies = range(n_train + n_validate), range(n_train + n_validate, n)
    train_indicies, validate_indicies = [], []
    for k in train_validate_indicies:
        _k = n_train + n_validate - k
        validate_indicies.append(k) if (_k <= 2 * n_validate and _k % 2 == 0) else train_indicies.append(k)

    train_value = [all_original[str(k)] for k in train_indicies]
    validate_value = [all_original[str(k)] for k in validate_indicies]
    test_value = [all_original[str(k)] for k in test_indicies]

    return train_value, validate_value, test_value, train_indicies, validate_indicies
# ##function 1.20
# #find the mobility data starting from the day, which is x_days before the start_date
# #start_date = "20200331", x_days = 7
# def sort_date(all_mobility, start_date, x_days): 
#     mobility_date_list = list(all_mobility.keys())
#     mobility_date_list.sort()
#     idx = mobility_date_list.index(start_date)
#     mobility_date_cut = mobility_date_list[idx-x_days:] 
#     return mobility_date_cut

# #function 1.21
# #find the mobility data starting from the day, which is x_days before the start_date,
# #ending at the day, which is y_days after the end_date
# #start_date = "20200331", x_days = 7
# def sort_date_2(all_mobility, start_date, x_days, end_date, y_days): 
#     mobility_date_list = list(all_mobility.keys())
#     mobility_date_list.sort()
#     idx = mobility_date_list.index(start_date)
#     idx2 = mobility_date_list.index(end_date)
#     mobility_date_cut = mobility_date_list[idx-x_days:idx2+y_days] 
#     return mobility_date_cut

#function 1.22
#get the mappings from zone id to id, text id to id.
#get zone_text_to_idx 
# def get_zone_text_to_idx(all_infection):
#     zone_list = sorted(set(all_infection["20200401"].keys()))
#     # text_list = list(['痛み', '頭痛', '咳', '下痢', 'ストレス', '不安',                     '腹痛', 'めまい'])
#     zone_dict = {str(zone_list[i]): i for i in range(len(zone_list))}
#     text_dict = {str(text_list[i]): i for i in range(len(text_list))}
#     return zone_dict, text_dict
# def get_zone_text_to_idx():
#     zone_list = sorted(set(all_infection["20200401"].keys()))
#     # text_list = list(['痛み', '頭痛', '咳', '下痢', 'ストレス', '不安',                     '腹痛', 'めまい'])
#     zone_dict = {str(zone_list[i]): i for i in range(len(zone_list))}
#     text_dict = {str(text_list[i]): i for i in range(len(text_list))}
#     return zone_dict, text_dict

def get_zone_text_to_idx(jcode23, text_list):
    zone_dict = {str(jcode23[i]): i for i in range(len(jcode23))}
    text_dict = {str(text_list[i]): i for i in range(len(text_list))}
    return zone_dict, text_dict

#function 1.23
#change the data format to matrix
#zoneid_to_idx = {"13101":0, "13102":1, ..., "13102":22}
#sym_to_idx = {"cough":0}
#mobility: {('13101', '13101'): 709973, ...}
#text: {'13101': {'痛み': 51,...},...}  text
#infection: {'13101': 50, '13102': 137, '13103': 401,...} 
#data_type = {"mobility", "text", "infection"}
def to_matrix(input_data, data_type):
    global zone_indicies, text_indicies
    n_zone, n_text = len(zone_indicies), len(text_indicies)
    if data_type == "mobility":
        result = np.zeros((n_zone, n_zone))
        for zone in input_data:    
            from_idx, to_idx = zone_indicies[zone[0]], zone_indicies[zone[1]]
            result[from_idx][to_idx] += input_data[zone]
    if data_type == "text":
        result = np.zeros((n_zone, n_text))
        for zone in input_data:
            for sym in input_data[zone]:
                if zone in list(zone_indicies.keys()) and sym in list(text_indicies.keys()):
                    zone_idx, text_idx = zone_indicies[zone], text_indicies[sym]
                    result[zone_idx][text_idx] += input_data[zone][sym]
    if data_type == "infection":
        result = np.zeros(n_zone)
        for zone in input_data:
            zone_idx = zone_indicies[zone]
            result[zone_idx] += input_data[zone]
    return result

#function 1.24
#change the data to the matrix format
# def change_to_matrix(data, zoneid_to_idx, sym_to_idx):
#     data_result = list()
#     for i in range(len(data)):
#         combine1, combine2 = list(), list()
#         combine3 = list()                                    #NEW
#         mobility_text = data[i][0]
#         x_infection_all = data[i][2]          #the x_days infection data
#         day_order =  data[i][3] #NEW          the order of the day
#         for j in range(round(len(mobility_text)*1.0/2)):
#             mobility, text = mobility_text[2*j], mobility_text[2*j+1]
#             x_infection =  x_infection_all[j]   #NEW
#             new_mobility = to_matrix(zoneid_to_idx, sym_to_idx, mobility, "mobility")
#             new_text = to_matrix(zoneid_to_idx, sym_to_idx, text, "text")
#             combine1.append(new_mobility)
#             combine1.append(new_text) 
#             new_x_infection = to_matrix(zoneid_to_idx, sym_to_idx, x_infection, "infection") #NEW
#             combine3.append(new_x_infection)   #NEW
#         for j in range(len(data[i][1])):
#             infection = data[i][1][j]                                                          
#             new_infection = to_matrix(zoneid_to_idx, sym_to_idx, infection, "infection")
#             combine2.append(new_infection)                                               
#         data_result.append([combine1,combine2,combine3,day_order])    #mobility/text; infection_y; infection_x; day_order
#     return data_result  
def change_to_matrix(data, hint):
    data_result = list()

    n = len(data)

    qbar = progress_indicator(total=n, desc=f"Transforming {hint} data to matrix")


    for i in range(n):

        combine1, combine2, combine3 = list(), list(), list()
        mobility_text_all, y_infection_all, x_infection_all, day_order = data[i]

        for j in range(round(len(mobility_text_all) * 1. / 2)):

            mobility, text, x_infection = mobility_text_all[2 * j], mobility_text_all[2 * j + 1], x_infection_all[j]   #NEW

            new_mobility = to_matrix(mobility, "mobility")
            new_text = to_matrix(text, "text")
            new_x_infection = to_matrix(x_infection, "infection") #NEW

            combine1.append(new_mobility)
            combine1.append(new_text) 
            combine3.append(new_x_infection)   #NEW
        for j in range(len(y_infection_all)):
            infection = y_infection_all[j]                                                          
            new_infection = to_matrix(infection, "infection")
            combine2.append(new_infection)                                               
        data_result.append([combine1, combine2, combine3, day_order])    #mobility/text; infection_y; infection_x; day_order

        qbar.update()
    return data_result

##################5.learn#####################
#function 1.25
def visual_loss(e_losses, vali_loss, test_loss):
    plt.figure(figsize=(4,3), dpi=300)
    x = range(len(e_losses))
    y1,y2,y3 = copy.copy(e_losses), copy.copy(vali_loss), copy.copy(test_loss)
    plt.plot(x,y1,linewidth=1, label="train",color="r")
    plt.plot(x,y2,linewidth=1, label="validate",color='b')
    plt.plot(x,y3,linewidth=1, label="test",color='g')
    plt.legend()
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    os.makedirs('result/figure_supp/', exist_ok=True)
    plt.savefig('result/figure_supp/learning_curve.pdf',bbox_inches = 'tight')
    plt.show()

#function 1.26
def visual_loss_train(e_losses):
    plt.figure(figsize=(4,3), dpi=300)
    x = range(len(e_losses))
    y1 = copy.copy(e_losses)
    plt.plot(x,y1,linewidth=1, label="train")
    plt.legend()
    plt.title('Loss decline on entire training data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.savefig('final_f6.png',bbox_inches = 'tight')
    plt.show()


# # Step 2: training epoch, training process, testing

# In[3]:


#function 2.1
#normalize each column of the input mobility matrix as one
def normalize_column_one(input_matrix):
    column_sum = np.sum(input_matrix, axis=0)
    row_num, column_num = len(input_matrix), len(input_matrix[0])
    for i in range(row_num):
        for j in range(column_num):
             input_matrix[i][j] = input_matrix[i][j]*1.0/column_sum[j]
    return input_matrix

#function 2.2
#evalute the trained_model on validation or testing data.
def validate_test_process(trained_model, vali_test_data):
    criterion = nn.MSELoss()
    vali_test_y = [vali_test_data[i][1] for i in range(len(vali_test_data))]
    y_real = torch.tensor(vali_test_y)
    
    vali_test_x = [vali_test_data[i] for i in range(len(vali_test_data))]
    vali_test_x = convertAdj(vali_test_x)
    y_hat = trained_model.run_specGCN_lstm(vali_test_x)                                          
    loss = criterion(y_hat.float(), y_real.float())            ###Calculate the loss  
    return loss, y_hat, y_real 

#function 2.3
#convert the mobility matrix in x_batch in a following way
#normalize the flow between zones so that the in-flow of each zone is 1.
def convertAdj(x_batch):
    #x_batch：(n_batch, 0/1, 2*i+1)
    x_batch_new = copy.copy(x_batch)
    n_batch = len(x_batch)
    days = round(len(x_batch[0][0])/2)
    for i in range(n_batch):
        for j in range(days):
            mobility_matrix = x_batch[i][0][2*j]
            x_batch_new[i][0][2*j] = normalize_column_one(mobility_matrix)   #20210818
    return x_batch_new

#function 2.4
#a training epoch
def train_epoch_option(model, opt, criterion, trainX_c, trainY_c, batch_size):  
    model.train()
    losses = []
    batch_num = 0
    for beg_i in range(0, len(trainX_c), batch_size):
        batch_num += 1
        if batch_num % 16 ==0:
            print ("batch_num: ", batch_num, "total batch number: ", int(len(trainX_c)/batch_size))
        x_batch = trainX_c[beg_i:beg_i+batch_size]        
        y_batch = torch.tensor(trainY_c[beg_i:beg_i+batch_size])   
        opt.zero_grad()
        x_batch = convertAdj(x_batch)   #conduct the column normalization
        y_hat = model.run_specGCN_lstm(x_batch)                          ###Attention
        loss = criterion(y_hat.float(), y_batch.float()) #MSE loss
        #opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.data.numpy())
    return sum(losses)/float(len(losses)), model

#function 2.5
#multiple training epoch
def train_process(train_data, lr, num_epochs, net, criterion, bs, vali_data, test_data):
    opt = optim.Adam(net.parameters(), lr, betas = (0.9,0.999), weight_decay=0) 
    train_y = [train_data[i][1] for i in range(len(train_data))]
    e_losses = list()
    e_losses_vali = list()
    e_losses_test = list()
    time00 = time.time()
    for e in range(num_epochs):
        time1 = time.time()
        print ("current epoch: ",e, "total epoch: ", num_epochs)
        number_list = list(range(len(train_data)))       
        random.shuffle(number_list, random = r)
        trainX_sample = [train_data[number_list[j]] for j in range(len(number_list))]
        trainY_sample = [train_y[number_list[j]] for j in range(len(number_list))]
        loss, net =  train_epoch_option(net, opt, criterion, trainX_sample, trainY_sample, bs)  
        print ("train loss", loss*infection_normalize_ratio*infection_normalize_ratio)
        e_losses.append(loss*infection_normalize_ratio*infection_normalize_ratio)
        
        loss_vali, y_hat_vali, y_real_vali = validate_test_process(net, vali_data) 
        loss_test, y_hat_test, y_real_test = validate_test_process(net, test_data)
        e_losses_vali.append(float(loss_vali)*infection_normalize_ratio*infection_normalize_ratio)
        e_losses_test.append(float(loss_test)*infection_normalize_ratio*infection_normalize_ratio)
        
        print ("validate loss", float(loss_vali)*infection_normalize_ratio*infection_normalize_ratio)
        print ("test loss", float(loss_test)*infection_normalize_ratio*infection_normalize_ratio)
        if e>=2 and (e+1)%10 ==0:
            visual_loss(e_losses, e_losses_vali, e_losses_test)     
            visual_loss_train(e_losses) 
        time2 = time.time()
        print ("running time for this epoch:", time2 - time1)
        time01 = time.time()
        print ("---------------------------------------------------------------")
        print ("---------------------------------------------------------------")
    return e_losses, net


# # Step 3: models

# In[4]:


#function 3.1
def read_data():

    global zone_indicies, text_indicies

    ####################################
    cwd = os.getcwd()
    os.chdir("data/multiwave_data")
    jcode23 = list(read_tokyo_23()["JCODE"])                    #1.1 get the tokyo 23 zone shapefile
    jcode23 = jcode23[:23]
    all_mobility = read_mobility_data(jcode23)                  #1.2 read the mobility data
    all_infection, all_infection_cum = read_infection_data(jcode23)                #1.4 read the infection data
    all_text = read_text_data(jcode23)                          #1.3 read the text data
    os.chdir(cwd)
    del cwd

    #smooth the data using 7-days average
    window_size = WINDOW_SIZE                 #20210818
    dateList = generateDates('20200101', '20211231')  #20210818
    all_mobility = mob_inf_smooth(all_mobility, window_size, dateList, "mobility") #20210818
    all_infection = mob_inf_smooth(all_infection, window_size, dateList, "infection")  #20210818
    
    #smooth, user, min-max.
    # point_json = read_point_json()                           #20210821
    # all_text = normalize_text_user(all_text, point_json)       #20210821
    all_text = text_smooth(all_text, window_size, dateList, 'text') #20210818
    all_text = min_max_text_data(all_text,jcode23)                 #20210820

    zone_indicies, text_indicies = get_zone_text_to_idx(jcode23, text_list)                       #get zone_indicies, text_indicies

    ####################################
        
    x_days, y_days =  X_day, Y_day
    # mobility_date_cut = sort_date_2(all_mobility, START_DATE, x_days, END_DATE, y_days)
    date_all = generateDates(str2date(START_DATE) - x_days, str2date(END_DATE) + y_days - 1)
    all_original = ensemble(all_mobility, all_text, all_infection, x_days, y_days, date_all)
    train_original, validate_original, test_original, train_indicies, validate_indicies = \
        split_dataset_backwards_even_index(all_original, train_ratio, validate_ratio)

    # zone_dict, text_dict = get_zone_text_to_idx(jcode23, text_list)                       #get zone_indicies, text_indicies
    train_x_y = change_to_matrix(train_original, "train")                   #get train
    validate_x_y = change_to_matrix(validate_original, "validate")             #get validate
    test_x_y = change_to_matrix(test_original, "test")                     #get test
    
    print ("train_x_y_shape",len(train_x_y),"train_x_y_shape[0]",len(train_x_y[0]))
    print (len(train_x_y),
           ': Length of training set')  #300
    print (len(train_x_y[0][0]),
           'Count of mobility&text data per zone in training set. (the time span for training set is', f'{len(train_x_y[0][0])} / 2 = {len(train_x_y[0][0]) // 2})') #14
    print (np.shape(train_x_y[0][0][0]),
           "Shape of mobility data per day per zone in training set.") #(23,23)
    print (np.shape(train_x_y[0][0][1]),
           "Shape of text data per dday per zone in training set.") #(23,43)
    #print ("---------------------------------finish data reading and preprocessing------------------------------------")

    return train_x_y, validate_x_y, test_x_y, all_mobility, all_infection, \
        train_original, validate_original, test_original, train_indicies, validate_indicies

#function 3.2
#train the model
def model_train(train_x_y, vali_data, test_data, device):
    #3.2.1 define the model
    input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2 = len(train_x_y[0][0][1][1]),    HIDDEN_DIM_1, OUT_DIM_1, HIDDEN_DIM_2 
    dropout_1, alpha_1, N = DROPOUT, ALPHA, len(train_x_y[0][0][1])
    G_L_Model = SpecGCN_LSTM(X_day, Y_day, input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2, dropout_1,N, device)         ###Attention
    #3.2.2 train the model
    num_epochs, batch_size, learning_rate = NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE                                                 #model train
    criterion = nn.MSELoss() 
    e_losses, trained_model = train_process(train_x_y, learning_rate, num_epochs, G_L_Model, criterion, batch_size,                          vali_data, test_data)
    return e_losses, trained_model

#function 3.3
#evaluate the error on validation (or testing) data.
def validate_test_process(trained_model, vali_test_data):
    criterion = nn.MSELoss()
    vali_test_y = [vali_test_data[i][1] for i in range(len(vali_test_data))]
    y_real = torch.tensor(vali_test_y)
    vali_test_x = [vali_test_data[i] for i in range(len(vali_test_data))]
    vali_test_x = convertAdj(vali_test_x)
    y_hat = trained_model.run_specGCN_lstm(vali_test_x)                                  ###Attention              
    loss = criterion(y_hat.float(), y_real.float())
    return loss, y_hat, y_real 

#################################################################################

# l_bar='{desc}...({n_fmt}/{total_fmt} {percentage:3.2f}%)'
# r_bar= '{n_fmt}/{total_fmt}'
# r_bar= '{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
# bar_format = f'{l_bar}|{{bar}}|{r_bar}{"{postfix}"} '
def bar_format(show_total):
    if show_total: return '{desc}...|{bar}|({n_fmt}/{total_fmt} {percentage:3.0f}%){postfix}'
    else: return '{desc}...{percentage:3.2f}% {postfix}'

def progress_indicator(*args, show_total=True, **kwargs):
    return tqdm(*args, **kwargs, bar_format=bar_format(show_total))

def generateDates(start, end):
    '''
    根据形如 20220103 的字符串 生成 起止时间中间的所有日期的字符串
    (包含 start 和 end)
    '''
    start, end = str2date(start), str2date(end) + timedelta(1)
    if not (end - start).days > 1: return None

    res = []
    while (end - start).days:
        res.append(date2str(start))
        start += timedelta(1)
    return res

def checkContinuous(lis: list, _type=None, _print=False):
    '''
    计算列表中连续的值的范围
    '''

    if _type == "date": convert = [str2date, date2str]
    else: convert = [lambda f: int(f)] * 2
    lis = [convert[0](i) for i in lis]
    msg = ('TOTAL', 'RANGE', 'LAST')

    lis.sort()
    # res 是元素为 tuple 的列表，以 (start, end) 的形式储存连续范围的起止
    res = []

    if _print: print(f"{msg[0]:6s} {convert[1](lis[0])} - {convert[1](lis[-1])}\n")
    
    start = lis[0]
    for i, l in enumerate(lis[1:]):
        _delta = l - lis[i]
        if isinstance(_delta, timedelta): _delta = _delta.days
        if _delta == 1: continue
        else:
            _res = (convert[1](start), convert[1](lis[i]))
            res.append(_res)
            start = l

            if _print: print(f"{msg[1]:6s} {_res}")

    if start == lis[0] and _print: print('Continuous'); return res

    _res = (convert[1](start), convert[1](l))
    res.append(_res)
    start = l

    if _print: print(f"{msg[2]:6s} {_res}")

    return res

def interpolate(dic: dict, avg = None, _type: str='date', _print=False):
    '''
    插补数据。数据是字典格式，其中key是日期，value是一个 {jCode: value} 格式的字典
    '''
    assert avg is not None

    # 获得 keys 的连续性信息
    res_continuous = checkContinuous(dic.keys(), "date")

    # 根据连续性信息插补字典对象
    res = []
    count = 0
    for i, (start, end) in enumerate(res_continuous[1:]):
        end_last = res_continuous[i][1]
        
        if _type == 'date':
            value = avg(dic, end_last, start)
            for d in generateDates(end_last, start)[1:-1]:
                if _print: print(f"插补 {d} 为 {end_last}-{start} 均值")
                dic[d] = value
                count += 1
        else:
            _res = list(range(int(end_last) + 1, int(start)))
            _res = [str(i) for i in _res]
            res += _res
    print(f'{count} new data interpreted')
    # return dic