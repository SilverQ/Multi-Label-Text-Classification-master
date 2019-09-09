import json
import os
import numpy as np
import pandas as pd
import pickle

# tf.data를 사용할 수 있도록, Section_Title.txt와 Section_Label.txt 파일을 만들어보자.

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

file_list = os.listdir(raw_path)
file_list = [file for file in file_list if file.endswith(".txt")]

label_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'y': 8}

# def label_to_id(label):
#     id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'y': 8}
#     return id(label)


def write_file(data, file_name):
    file_path = os.path.join('../data/', file_name)
    with open(file_path, 'a+', encoding='utf-8') as write_json:
        try:
            write_json.write('{}\n'.format('\n'.join(json.dumps(d) for d in data)))
            # write_json.write(data)
            # write_json.write('{}\n')
            # write_json.write(json.dumps(d) for d in data
            # json.dump(data, write_json, ensure_ascii=False)
        except:
            pass


def making_source_v2():
    source_file = open(os.path.join('../data/', 'title_Section_test.txt'), 'a', encoding='utf-8')
    for file_num, file in enumerate(file_list):
        title_Section = []
        raw_data = open(os.path.join('../data/Raw_data', file), encoding='utf-8')
        for line in raw_data:
            try:
                d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                title_Section.append([d["cpc"][0], d["title"]])
            except:
                pass
        source_file.write('{}\n'.format('\n'.join(json.dumps(d) for d in title_Section)))
        print('Reading File_{}_{} finished'.format(file_num+1, file))


# making_source_v2()


def making_source_csv():
    source_file = open(os.path.join('../data/', 'title_Section.csv'), 'a', encoding='utf-8')
    for file_num, file in enumerate(file_list):
        title_Section = []
        raw_data = open(os.path.join('../data/Raw_data', file), encoding='utf-8')
        for line in raw_data:
            try:
                d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                title_Section.append([label_id[d["cpc"][0].lower()], d["title"]])
            except:
                pass
        df = pd.DataFrame(data=title_Section)
        df.to_csv(source_file, sep='\t', header=False, index=False)
        print('Reading File_{}_{} finished'.format(file_num+1, file))


making_source_csv()

# read_list = np.load(os.path.join('../data/', 'title_Section_test.npy')).tolist()
# read_list = np.load(os.path.join('../data/', 'title_Section_test.npy'))
# print(read_list[:3], type(read_list))


def making_source_v1():
    for file_num, file in enumerate(file_list):
        with open(os.path.join('../data/', 'title_Section.txt'), 'a', encoding='utf-8') as f:
            with open(os.path.join('../data/Raw_data', file), encoding='utf-8') as data_file:
                # title = []
                # cpc = []
                title_Section = []
                for line in data_file:
                    try:
                        d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                        # print(d["cpc"][0], d["title"])
                        # cpc.append(d["cpc"][0])
                        # title.append(d["title"])
                        title_Section.append([d["cpc"][0], d["title"]])
                    except:
                        pass
                # print(title_Section[:3])
                # df = pd.DataFrame(data=title_Section)
                # df.to_csv(os.path.join('../data/', 'title_Section.txt'), sep='\t', header=False, index=False)
                # df.to_csv(f, sep='\t', header=False, index=False, encoding='utf-8')
                # write_file(title, 'Section_Title.txt')
                # write_file(cpc, 'Section_Label.txt')
                write_file(title_Section, 'title_Section.txt')
                print('Reading File_{}_{} finished'.format(file_num+1, file))


# making_source_v1()
