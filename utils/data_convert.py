# -*- coding:utf-8 -*-
__author__ = 'SilverQ'

import json
import os


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
# labels_index is made for all classes and all hierarchy level
# ex) C25D1/18 -> {0: 'C', 1: 'C25', 2: 'C25D', 3: 'C25D1', 4: 'C25D1/18'}
# labels_index will be saved to "labels_index.json"


raw_path = '../data/Raw_data'

file_list = os.listdir(raw_path)
# print(file_list[0])

data = []
patnum_idx = dict()
cpc_idx = dict()
idx_cpc = dict()


def patnum_to_idx(patnum_idx, id_kipi):
    pat_cnt = len(patnum_idx)
    patnum_idx[pat_cnt] = id_kipi
    return pat_cnt


def cpc_to_idx(cpc_idx, idx_cpc, cpcs):
    result = []
    for cpc in cpcs.split("|"):
        # cpc_section = cpc[0], cpc_class = cpc[0:3], cpc_subclass = cpc[0:4], cpc_group = cpc.split("/")[0]
        # cpc_subgroup = cpc
        cpc_labels = [cpc[0], cpc[0:3], cpc[0:4], cpc.split("/")[0]]
        # print(cpc_labels)
        for label in cpc_labels:
            if label not in cpc_idx:
                cpc_cnt = len(cpc_idx)
                cpc_idx[cpc_cnt] = label
                idx_cpc[label] = cpc_cnt
                # print('added label: ', label)
        if cpc not in cpc_idx:
            cpc_cnt = len(cpc_idx)
            cpc_idx[cpc_cnt] = cpc
            idx_cpc[cpc] = cpc_cnt
            # print('added label: ', label)
        result.append(idx_cpc[cpc])
    # print('result: ')
    # print(result)
    return result


for file_num, file in enumerate(file_list):
    with open(os.path.join('../data/Raw_data', file), encoding='utf-8') as data_file:
        for line in data_file:
            # print(line)
            try:
                d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                # print(d)
                # id_kipi -> testid(=pat_cnt) : Done
                # cpc -> labels_index(list type),
                # p -> features_content(splitted with blank list type, cpc_cnt -> labels_num(integer)
                # print(d["id_kipi"])
                test_id = str(patnum_to_idx(patnum_idx, d["id_kipi"]))
                labels_index = d["cpc"]
                # print(labels_index)
                labels_index = cpc_to_idx(cpc_idx, idx_cpc, labels_index)
                features_content = d["p"].split()
                labels_num = len(labels_index)
                d = {"test_id": test_id,
                     "features_content": features_content,
                     "labels_index": labels_index,
                     "labels_num": labels_num}
                # print(d)
                data.append(d)
            except:
                pass
        print('Reading File_{} finished'.format(file_num+1))

print('Reading {} patents success'.format(len(data)))

# print(data[0:3])
# print(patnum_idx[3])

# print('starting json writing')
with open('../data/data.json', 'w', encoding='utf-8') as write_json:
    # for line in data:
    #     print('json writing')
    #     print(line)
    #     json.dump(line, write_json, ensure_ascii=False)
    json.dump(data, write_json, ensure_ascii=False)
    print('finished json writing')


# 일단 데이터 저장은 완료. 이제 샘플 데이터로 모델 학습 테스트를 해보자!!!