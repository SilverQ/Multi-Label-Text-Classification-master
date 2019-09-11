import json
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from tensorflow.keras import preprocessing
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# tf.data를 사용할 수 있도록, Section_Title.txt와 Section_Label.txt 파일을 만들어보자.
# 이번엔 토크나이징까지 해서 수치 데이터로 저장해보자
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
label_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'y': 8}


# def write_file(data, file_name):
#     file_path = os.path.join('../data/', file_name)
#     with open(file_path, 'a+', encoding='utf-8') as write_json:
#         try:
#             write_json.write('{}\n'.format('\n'.join(json.dumps(d) for d in data)))
#             # write_json.write(data)
#             # write_json.write('{}\n')
#             # write_json.write(json.dumps(d) for d in data
#             # json.dump(data, write_json, ensure_ascii=False)
#         except:
#             pass


# def making_source_v2():
#     source_file = open(os.path.join('../data/', 'title_Section_test.txt'), 'a', encoding='utf-8')
#     for file_num, file in enumerate(file_list):
#         title_Section = []
#         raw_data = open(os.path.join('../data/Raw_data', file), encoding='utf-8')
#         for line in raw_data:
#             try:
#                 d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
#                 title_Section.append([d["cpc"][0], d["title"]])
#             except:
#                 pass
#         source_file.write('{}\n'.format('\n'.join(json.dumps(d) for d in title_Section)))
#         print('Reading File_{}_{} finished'.format(file_num+1, file))


# making_source_v2()

# token_file = os.path.join('../data/', 'tokenizer.pickle')     # title to section
token_file = os.path.join('../data/', 'abst_tokenizer.pickle')       # abstraction to section


def load_tokenizer(token_file):
    print('Loading Saved Tokenizer')
    with open(token_file, 'rb') as handle:
        tokenizer = pickle.load(handle)
    # word_vocab = tokenizer.word_index                                                   # 단어 사전 형태
    return tokenizer


def fit_tokenizer(file_list):
    print('Starting to Fit Tokenizer')
    t = preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                     lower=False, split=' ', char_level=False,
                                     oov_token=None, document_count=0)
    for f in tqdm(file_list):
        with open(os.path.join('../data/Raw_data', f), 'r') as raw_data:
            # title_texts = json.loads(raw_data)
            title_texts = []
            for line in raw_data:
                try:
                    d = json.loads(line)
                    # print(d, '\n', d['title'])
                    title_texts.append(d['title'])
                    title_texts.append(d['p'])
                except:
                    pass
        # print('title_texts: ', title_texts)
        t.fit_on_texts(title_texts)
    word_vocab = t.word_index
    # print(word_vocab)
    print('word counts: ', len(word_vocab))     # word counts:  406014
    print('Index of word "extruding": ', word_vocab['extruding'])
    with open(token_file, 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return t


if os.path.isfile(token_file):
    tokenizer = load_tokenizer(token_file)
    print('Load Tokenizer Finished')
else:
    tokenizer = fit_tokenizer(file_list)
    print('Fit Tokenizer Finished')

# word_vocab = tokenizer.word_index
# print('word counts: ', len(word_vocab))     # word counts:  406014, word counts:  1223401
# print('Index of word "extruding": ', word_vocab['extruding'])


# source_file = open(os.path.join('../data/', 'title_Section_token.csv'), 'a', encoding='utf-8')
#
# for file_num, file in enumerate(file_list):
#     title_Section = []
#     title = []
#     label = []
#     raw_data = open(os.path.join('../data/Raw_data', file), encoding='utf-8')
#     for line in raw_data:
#         try:
#             d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
#             title.append(d['title'])
#             label.append(label_id[d["cpc"][0].lower()])
#             # title_Section.append([label_id[d["cpc"][0].lower()], d["title"]])
#         except:
#             pass
#     title_token = tokenizer.texts_to_sequences(title)
#     # print('title: \n', title, '\n', title_token,)
#     title_token = preprocessing.sequence.pad_sequences(title_token,
#                                                        maxlen=max_length,
#                                                        padding='pre')  # 학습 데이터를 벡터화
#     # title2 = tokenizer.sequences_to_texts(title_token)
#     # print('title: \n', title, '\n', title_token, '\n', title2)
#     data = {'label': label, 'title': title_token}
#
#     df = pd.DataFrame(data=data)
#     # df.to_csv(source_file, sep='\t', header=False, index=False)
#     print('Reading File_{}_{} finished'.format(file_num+1, file))
"""
title: 
 ['Method for extruding brittle materials', 'Toughening roll die work method for metallic material', 'Automatic apparatus and method for simultaneously forming eyes on each end of a leaf spring blank', 'Neckerflanger for metal cans', 'Method of forming constriction in tubing', 'Vibration densitometer'] 
 [[7, 2, 8454, 14806, 194], [61366, 1118, 930, 1268, 6, 2, 1320, 68], [249, 8, 1, 6, 2, 3023, 49, 14620, 36, 1959, 484, 3, 4, 5876, 912, 3105], [360698, 2, 120, 8378], [7, 3, 49, 25636, 15, 2635], [2134, 22801]] 
 ['Method for extruding brittle materials', 'Toughening roll die work method for metallic material', 'Automatic apparatus and method for simultaneously forming eyes on each end of a leaf spring blank', 'Neckerflanger for metal cans', 'Method of forming constriction in tubing', 'Vibration densitometer']
Reading File_1_cpc_json_1_6docs.txt finished
title: 
 ['Dehydrating rubber deposited from aqueous dispersions', 'Relay system', 'Process of manufacturing polyvinyl derivatives containing nitrogen', 'Application of liquid treating material to strip material', 'Extrusion of thermoplastic materials', 'Production of nitropolystyrene', 'Electrical vapor detector', 'Rigid shock-resistant vinyl halide polymer compositions and method of making same', 'Universal coupling', 'Light-sensitive polymers for making printing plates'] 
 [[38918, 1154, 5113, 50, 781, 3010], [5557, 10], [43, 3, 37, 5909, 140, 74, 1965], [1513, 3, 99, 202, 68, 20, 950, 68], [6301, 3, 1127, 194], [563, 3, 256132], [323, 1084, 376], [6523, 1932, 369, 1842, 1392, 260, 73, 1, 6, 3, 66, 23], [1129, 445], [246, 539, 475, 2, 66, 185, 1615]] 
 ['Dehydrating rubber deposited from aqueous dispersions', 'Relay system', 'Process of manufacturing polyvinyl derivatives containing nitrogen', 'Application of liquid treating material to strip material', 'Extrusion of thermoplastic materials', 'Production of nitropolystyrene', 'Electrical vapor detector', 'Rigid shock resistant vinyl halide polymer compositions and method of making same', 'Universal coupling', 'Light sensitive polymers for making printing plates']
Reading File_2_cpc_json_0_3docs.txt finished
"""
