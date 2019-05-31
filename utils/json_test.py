# -*- coding:utf-8 -*-
__author__ = 'SilverQ'

import json


raw_path = '../data/data.json'

with open(raw_path, encoding='utf-8') as data_file:
    for line in data_file:
        print(line)
        d = json.loads(line)
        print(d)
