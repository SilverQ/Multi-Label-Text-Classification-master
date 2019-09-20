# 원래 소스코드에 맞도록 데이터를 만들어보자
# data_sample.json :
#
# {"testid": "3930328", "features_content": ["anchor", "slidably", "adjustably", "carried", "tie-line", "floatable",
# "decoy", "duck", "comprising", "generally", "elongated", "non-rustable", "metal", "body", "preferably", "wider",
# "end", "continuous", "passageway", "tie-line", "extending", "returning", "end", "partially", "body", "passageway",
# "open", "side", "portion", "length", "elastic", "member", "carried", "body", "overlying", "open", "side",
# "passageway", "forcibly", "engaging", "tie-line", "resist", "free", "travel", "tie-line", "required", "useable",
# "length", "tie-line", "anchor", "desirably", "determined", "anchoring", "decoy", "limited", "movement", "floating",
# "unit", "storage", "tie-line", "tension", "decoy", "ready", "re-use"],
# "labels_index": [9, 17, 288],
# "labels_num": 3}


import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from tensorflow.keras import preprocessing
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# testid는 출원번호->id 데이터를 생성해서 매칭 데이터를 생성해두자
# features_content는 p를 단어 단위로 달라서 리스트로
# labels_index는 subclass 단위로 id를 생성해서 리스트로

"""
Original Data
{"id_kipi":"US001899016A_19330228",
 "pd":"19330228",
 "cpc":"C25D1/18|B29B15/1",
 "title":"Dehydrating rubber deposited from aqueous dispersions",
 "p":"299,713. Anode Rubber Co., Ltd., (Assignees of Darby, C. L.). Oct. 28, 1927, [Convention date]. Coagulating.
      -Electric endosmose is employed in the removal of water from layers or articles of rubber or like substance 
      formed from an aqueous dispersion. In one method a metal plate coated with rubber is connected as anode while 
      immersed in mercury or an electrolyte; exposed parts of the plate not coated with rubber are coated with wax or 
      other insulating composition. In another method the coated plate, connected as anode, is placed between two 
      metal plates connected as cathodes. In the apparatus shown, a wire or strip 32 is passed through an insulating 
      seal 33 of soft vulcanized rubber in the base of a tubular tank 30 connected as cathode and is thereby coated 
      with rubber or like substance from a dispersion in the tank. The coated wire then passes through a vessel 35, 
      connected as a cathode, containing a conductive liquid for dehydration, and then to a final drying chamber 39. 
      The wire is connected to the generators B, E by brushes 34. The depositing voltage may be 20-60 and the drying 
      voltage 20-110. In a modification the coated wire is passed for dehydration between pairs of rollers connected 
      as cathodes instead of through a conductive liquid. The invention may be applied to the dehydration of coated 
      fabrics."}

I will convert the original data to below data
[label, sentence]
B Flow rack
B Paper roll storage and handling installation and method for storing and handling paper rolls
B Carrier wheel assembly for sweep auger
B Unitary liftgate
E Hydraulically actuated casing slip lifter with hinged wrap arm assembly
F Horizontal wind generator
B Rotor blade system with reduced blade-vortex interaction noise
F Vane and/or blade for noise control
F Cool air circulation blower for refrigerator
F Formed disk plate heat exchanger
F Spiral-based axial flow devices
F Water torque converter
F Heat dissipation device and its impeller thereof
F Turbochargers
"""

raw_path = '../data/Raw_data'
max_length = 10
file_list = os.listdir(raw_path)
file_list = [file for file in file_list if file.endswith(".txt")]
# file_list = [file for file in file_list if file.endswith("docs.txt")]
print('file_list: ', file_list)

"""
Original Data
{"id_kipi":"US001899016A_19330228",
 "pd":"19330228",
 "cpc":"C25D1/18|B29B15/1",
 "title":"Dehydrating rubber deposited from aqueous dispersions",
 "p":"299,713. Anode Rubber Co., Ltd., (Assignees of Darby, C. L.). Oct. 28, 1927, [Convention date]. Coagulating.
      -Electric endosmose is employed in the removal of water from layers or articles of rubber or like substance 
      formed from an aqueous dispersion. In one method a metal plate coated with rubber is connected as anode while 
      immersed in mercury or an electrolyte; exposed parts of the plate not coated with rubber are coated with wax or 
      other insulating composition. In another method the coated plate, connected as anode, is placed between two 
      metal plates connected as cathodes. In the apparatus shown, a wire or strip 32 is passed through an insulating 
      seal 33 of soft vulcanized rubber in the base of a tubular tank 30 connected as cathode and is thereby coated 
      with rubber or like substance from a dispersion in the tank. The coated wire then passes through a vessel 35, 
      connected as a cathode, containing a conductive liquid for dehydration, and then to a final drying chamber 39. 
      The wire is connected to the generators B, E by brushes 34. The depositing voltage may be 20-60 and the drying 
      voltage 20-110. In a modification the coated wire is passed for dehydration between pairs of rollers connected 
      as cathodes instead of through a conductive liquid. The invention may be applied to the dehydration of coated 
      fabrics."}

I will convert the original data to below data

Target Data 
{"testid": "3930328",
 "features_content": ["anchor", "slidably", "adjustably", "carried", "tie-line", "floatable", "decoy", "duck",
                      "comprising", "generally", "elongated", "non-rustable", "metal", "body", "preferably", "wider",
                      "end", "continuous", "passageway", "tie-line", "extending", "returning", "end", "partially", 
                      "body", "passageway", "open", "side", "portion", "length", "elastic", "member", "carried", 
                      "body", "overlying", "open", "side", "passageway", "forcibly", "engaging", "tie-line", 
                      "resist", "free", "travel", "tie-line", "required", "useable", "length", "tie-line", "anchor", 
                      "desirably", "determined", "anchoring", "decoy", "limited", "movement", "floating", "unit", 
                      "storage", "tie-line", "tension", "decoy", "ready", "re-use"],
 "labels_index": [9, 17, 288],
 "labels_num": 3}
"""

# To-Do
# 1. Read the original data and make "testid"
# 2. Making "labels_index" for Classify All Class
# 3. Re-format the data and save

# id_kipi is not unique, but test_id is meaningless so it maybe ok if it is made roughly.
# labels_index will be saved to "labels_index.json"

patnum_idx = dict()
cpc_idx = dict()
idx_cpc = dict()


def patnum_to_idx(id_kipi):
    pat_cnt = len(patnum_idx)
    patnum_idx[pat_cnt] = id_kipi
    return str(pat_cnt)


def cpc_to_idx(cpcs):
    result = []
    # "cpc": "C25D1/18|B29B15/1",
    result = set([cpc[0:4] for cpc in cpcs.split("|")])
    # print('cpc: ', result)
    for cpc in result:
        if cpc in cpc_idx:
            pass
        else:
            cpc_idx[cpc] = len(cpc_idx)
    # print(cpc_idx)
    result = [cpc_idx[cpc] for cpc in result]
    # print(result)
    return result


def data_split(input_data):
    tr_set, te_set = train_test_split(input_data, test_size=0.2, random_state=1)
    tr_set, va_set = train_test_split(tr_set, test_size=0.2, random_state=1)
    return tr_set, te_set, va_set


def write_file(data, file_name):
    print('starting json writing')
    file_path = os.path.join('../data/', file_name)
    with open(file_path, 'a+', encoding='utf-8') as write_json:
        write_json.write('{}\n'.format('\n'.join(json.dumps(d) for d in data)))
        # try:
        #     # write_json.write('\n'.join([json.dumps(d) for d in data]))
        #     write_json.write('{}\n'.format('\n'.join(json.dumps(d) for d in data)))
        # except:
        #     print('Err occured')
        # finally:
        #     pass


for file in tqdm(file_list[:10]):
    with open(os.path.join('../data/Raw_data', file), encoding='utf-8') as data_file:
        data = []
        content = []
        for line in data_file:
            # print(line)
            try:
                d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                test_id = patnum_to_idx(d["id_kipi"])
                labels_index = cpc_to_idx(d["cpc"])
                features_content = preprocessing.text.text_to_word_sequence(
                    d["title"] + " " + d["p"], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
                labels_num = len(labels_index)
                c = d["title"] + " " + d["p"]
                d = {"testid": test_id,
                     "features_content": features_content,
                     "labels_index": labels_index,
                     "labels_num": labels_num}
                data.append(d)
                # print(c)
                content.append(c)
            except:
                pass
        # print(data)
        print(content[:3])
        train_set, test_set, val_set = data_split(data)
        write_file(train_set, 'Train.json')
        write_file(val_set, 'Validation.json')
        write_file(test_set, 'Test.json')
        write_file(content, 'content.txt')


def obj_dict(obj):
    return obj.__dict__


def write_file2(input_data, file_name):
    file_path = os.path.join('../data/', file_name)
    with open(file_path, 'wb+') as write_file1:
        pickle.dump(input_data, write_file1)
    print('Writing File {} finished'.format(file_name))


print(len(patnum_idx))  # 9362172
print(len(cpc_idx))     # 174833
print(len(idx_cpc))     # 174833

write_file2(cpc_idx, 'cpc_idx.json')
write_file2(patnum_idx, 'patnum_idx.json')
