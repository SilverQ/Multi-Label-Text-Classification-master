import json
import os
import pickle

with open('../data/cpc_idx.json', 'rb') as data_file:
    data = pickle.load(data_file)
    print(len(data), data)
