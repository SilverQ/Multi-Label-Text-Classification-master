# -*- coding:utf-8 -*-
__author__ = 'SilverQ'

import json
import os
from sklearn.model_selection import train_test_split
import pickle


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
<Training Data with Title of invention>
["Body assembled from at least two component bodies",
 "Bi-directional dot matrix printer with slant control",
 "Vertical lift assembly",
 "Layer-forming apparatus especially for particle board mats",
 "Fungicidal 5-dialkylamino-4-nitrosulfonamidothiophenes",
 "Multi-level knock-down framework structure for supporting a plurality of objects",
 "Apparatus for piling up plate-shaped articles",
 "Self-locking screw nut",
 "Process for the oxidation of waste liquors arising from the manufacture of paper pulp",
 "Pressure control arrangement for hydraulic brake systems"]
<Label Data>
["E", "B", "B", "B", "C", "E", "C", "F", "D", "B"]
"""

raw_path = '../data/Raw_data'

file_list = os.listdir(raw_path)
file_list = [file for file in file_list if file.endswith(".txt")]


def cpc_to_idx(cpcs):
    # result = []
    # for cpc in cpcs.split("|"):
    #     cpc_labels = [cpc[0], cpc[0:3], cpc[0:4], cpc.split("/")[0], cpc]
    #     result.append(cpc_labels)
    result = set([cpc[0] for cpc in cpcs.split('|')])
    return result


def data_split(input_data):
    tr_set, te_set = train_test_split(input_data, test_size=0.2, random_state=1)
    tr_set, va_set = train_test_split(tr_set, test_size=0.2, random_state=1)
    return tr_set, te_set, va_set


def write_file(data, file_name):
    file_path = os.path.join('../data/', file_name)
    with open(file_path, 'a+', encoding='utf-8') as write_json:
        try:
            # write_json.write('\n'.join([json.dumps(d) for d in data]))
            write_json.write('{}\n'.format(json.dumps(data)))
        except:
            pass


for file_num, file in enumerate(file_list):
    with open(os.path.join('../data/Raw_data', file), encoding='utf-8') as data_file:
        title = []
        cpc = []
        for line in data_file:
            try:
                d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                cpc.append(d["cpc"][0])
                title.append(d["title"])
            except:
                pass
        write_file(title, 'Section_Title.txt')
        write_file(cpc, 'Section_Label.txt')
        print('Reading File_{} finished'.format(file_num+1))


def write_file2(data, file_name):
    file_path = os.path.join('../data/', file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
        print('Writing File {} finished'.format(file_name))
