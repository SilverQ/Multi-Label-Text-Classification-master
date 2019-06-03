# -*- coding:utf-8 -*-
__author__ = 'SilverQ'

import json


raw_path = '../data/data.json'
raw_path = '../data/content.txt'
raw_path = '../data/Test.json'


# with open(raw_path, encoding='utf-8') as data_file:
#     for line in data_file:
#         print(line)
#         d = json.loads(line)
#         print(d)
#
with open(raw_path, encoding='utf-8') as data_file:
    for i, line in enumerate(data_file):
        if i < 1:
            print(line, end='\n')
            # print(line['testid'], end='\n')
            d = json.loads(line)
            print(d)
            print(d['testid'])
