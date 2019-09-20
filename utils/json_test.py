# -*- coding:utf-8 -*-
__author__ = 'SilverQ'

import json
import os
import pickle


raw_path = '../data/data.json'
raw_path = '../data/content.txt'
raw_path = '../data/Test.json'
raw_path = '../data/cpc_idx.json'
raw_path = '../data/patnum_idx.json'


# with open(raw_path, encoding='utf-8') as data_file:
#     for line in data_file:
#         print(line)
#         d = json.loads(line)
#         print(d)
#
# with open(raw_path, encoding='utf-8') as data_file:
#     for i, line in enumerate(data_file):
#         if i < 1:
#             print(line, end='\n')
#             # print(line['testid'], end='\n')
#             d = json.loads(line)
#             print(d)
#             print(d['testid'])
#     with open(os.path.join('../data/Raw_data', file), encoding='utf-8') as data_file:
#         data = []

# with open(raw_path, 'rb') as read_file:
#     dict_a = pickle.load(read_file)
#     print(dict_a[4])

with open('../data/cpc_idx.json', 'rb') as data_file:
    data = pickle.load(data_file)
    num_classes = len(data)        # 680

print(data)
