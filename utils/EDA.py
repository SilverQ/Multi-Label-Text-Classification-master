import json
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from tensorflow.keras import preprocessing

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
"""

raw_path = '../data/Raw_data'
file_list = os.listdir(raw_path)
file_list = [file for file in file_list if file.endswith(".txt")]
# file_list = [file for file in file_list if file.endswith("docs.txt")]
print('file_list: ', file_list)

d_list = []
for f in tqdm(file_list):
    with open(os.path.join('../data/Raw_data', f), 'r') as raw_data:
        for line in raw_data:
            try:
                d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                # print('d: ', d)
                d['section'] = d['cpc'][0]
                d_list.append(d)
            except:
                pass
        # print('d: ', d_list)
        df = pd.DataFrame(data=d_list, columns=["id_kipi", "pd", "cpc", "title", "p", 'section'])
        df2 = df[["title", 'section']]
        """
        # d: [{'id_kipi': 'US003967485A_19760706', 'pd': '19760706', 'cpc': 'B21C26/00|B21C23/',
        #      'title': 'Method for extruding brittle materials',
        #      'p': 'A method for extruding a billet made of brittle metals, alloys, intermetallic compounds, ferrites or ceramics, which comprises embedding the billet in a solid pressure transmitting medium having a configuration and a dimension fitting to those of the inner space of a container of an extruder and being capable of plastic deformation by the actuating pressure of a compressing plunger of the extruder, loading the billet in the container of the extruder, pressurizing the pressure transmitting medium by the compressing plunger, and extruding the billet by actuating the extrusion plunger while the pressure is being maintained.'},
        #     {'id_kipi': 'US003967486A_19760706', 'pd': '19760706', 'cpc': 'C21D7/13|C22F1/00',
        #      'title': 'Toughening roll die work method for metallic material',
        #      'p': 'A method of working metallic material with roll dies resulting in toughening of the worked material is herein described, which method is characterized in that said metallic material is worked with roll dies while generating super-plastic phenomena by applying a temperature cycle passing over a transformation point to said metallic material.'},
        #     {'id_kipi': 'US003967487A_19760706', 'pd': '19760706', 'cpc': 'B21D53/886',
        #      'title': 'Automatic apparatus and method for simultaneously forming eyes on each end of a leaf spring blank',
        #      'p': 'Automatic apparatus and method for simultaneously forming eyes at each end of a leaf spring blank in which hydraulic clamps clamp each end of a spring blank into a working position, hydraulic knock-downs knock down the ends of the spring blank about gooseneck forms advanced in alignment with and beneath the clamping surfaces of the clamps and partially form the ends of the blank, scarfing knives advance with scrolling dies to cut the extreme ends of the blank to conform to the underside of the gooseneck forms. The scrolling dies are advanced toward the ends of the spring blank to further form the ends of the spring blank to a semicircular form. Sizing pins are advanced through the semicircular form of the spring blank, and the scrolls are further advanced to conform the eyes with the anvil and form the spring to size about the sizing pins. The sizing pins are then removed and the formed blank may be passed to a quenching solution.'},
        #     {'id_kipi': 'US003967488A_19760706', 'pd': '19760706', 'cpc': 'B21D51/2615|B21D5',
        #      'title': 'Neckerflanger for metal cans',
        #      'p': 'An apparatus is disclosed for simultaneously necking and flanging metal cans. A collapsible mandrel is provided, which in the collapsed condition will accept a can and which in the expanded condition grips the can over its entire inside surface. The mandrel is constructed such that in its expanded condition it presents an uninterrupted cylindrical surface. The mandrel has a groove in the region where the necking-flanging operation is to be performed and an elastic material is seated in the groove. A die ring larger in diameter than said mandrel in its expanded condition is mounted rotatably coplanar with said groove and means are provided to move said die ring radially of the mandrel to produce the necking and flanging operation by cooperation with the elastic material in said groove. Expansion and collapse of the mandrel is accomplished by internal cam structure.'},
        #     {'id_kipi': 'US003967489A_19760706', 'pd': '19760706', 'cpc': 'B21D53/02|B21C37/',
        #      'title': 'Method of forming constriction in tubing',
        #      'p': 'The present invention provides a method of forming a flow restriction in a tube to be used as a capillary in a refrigeration system. The method includes shaping the tube longitudinally to a preselected configuration, and then squeezing a section of the tube while directing a fluid therethrough until the fluid reaches a preselected pressure.'},
        #     {'id_kipi': 'US003967490A_19760706', 'pd': '19760706', 'cpc': 'G01N9/002', 'title': 'Vibration densitometer',
        #      'p': 'A vibration densitometer having a strain gage pick-up to serve at least four functions one at a time or to serve two, three or four functions in any combination of two, three or four. The strain gage is bonded to or may embody a vibrating vane or body to be immersed in a fluid. The strain gage may be bonded to a vane with glass or an epoxy or other uniting agent. If glass is used, the densitometer may undergo unusually high temperatures and still perform reliably and accurately. The strain gage may be used (1) as a pick-up in an electromechanical oscillator embodied in the densitometer, (2) as a portion of a fluid temperature indicator, (3) for instrument frequency error correction (computable in two ways) as function of temperature, and (4) for producing an indication of what the fluid density would be at a selected constant temperature.'}]
        
        d:  [{'id_kipi': 'US003967485A_19760706', 'pd': '19760706', 'cpc': 'B21C26/00|B21C23/', 'title': 'Method for extruding brittle materials', 'p': 'A method for extruding a billet made of brittle metals, alloys, intermetallic compounds, ferrites or ceramics, which comprises embedding the billet in a solid pressure transmitting medium having a configuration and a dimension fitting to those of the inner space of a container of an extruder and being capable of plastic deformation by the actuating pressure of a compressing plunger of the extruder, loading the billet in the container of the extruder, pressurizing the pressure transmitting medium by the compressing plunger, and extruding the billet by actuating the extrusion plunger while the pressure is being maintained.'},
             {'id_kipi': 'US003967486A_19760706', 'pd': '19760706', 'cpc': 'C21D7/13|C22F1/00', 'title': 'Toughening roll die work method for metallic material', 'p': 'A method of working metallic material with roll dies resulting in toughening of the worked material is herein described, which method is characterized in that said metallic material is worked with roll dies while generating super-plastic phenomena by applying a temperature cycle passing over a transformation point to said metallic material.'},
             {'id_kipi': 'US003967487A_19760706', 'pd': '19760706', 'cpc': 'B21D53/886', 'title': 'Automatic apparatus and method for simultaneously forming eyes on each end of a leaf spring blank', 'p': 'Automatic apparatus and method for simultaneously forming eyes at each end of a leaf spring blank in which hydraulic clamps clamp each end of a spring blank into a working position, hydraulic knock-downs knock down the ends of the spring blank about gooseneck forms advanced in alignment with and beneath the clamping surfaces of the clamps and partially form the ends of the blank, scarfing knives advance with scrolling dies to cut the extreme ends of the blank to conform to the underside of the gooseneck forms. The scrolling dies are advanced toward the ends of the spring blank to further form the ends of the spring blank to a semicircular form. Sizing pins are advanced through the semicircular form of the spring blank, and the scrolls are further advanced to conform the eyes with the anvil and form the spring to size about the sizing pins. The sizing pins are then removed and the formed blank may be passed to a quenching solution.'},
             {'id_kipi': 'US003967488A_19760706', 'pd': '19760706', 'cpc': 'B21D51/2615|B21D5', 'title': 'Neckerflanger for metal cans', 'p': 'An apparatus is disclosed for simultaneously necking and flanging metal cans. A collapsible mandrel is provided, which in the collapsed condition will accept a can and which in the expanded condition grips the can over its entire inside surface. The mandrel is constructed such that in its expanded condition it presents an uninterrupted cylindrical surface. The mandrel has a groove in the region where the necking-flanging operation is to be performed and an elastic material is seated in the groove. A die ring larger in diameter than said mandrel in its expanded condition is mounted rotatably coplanar with said groove and means are provided to move said die ring radially of the mandrel to produce the necking and flanging operation by cooperation with the elastic material in said groove. Expansion and collapse of the mandrel is accomplished by internal cam structure.'},
             {'id_kipi': 'US003967489A_19760706', 'pd': '19760706', 'cpc': 'B21D53/02|B21C37/', 'title': 'Method of forming constriction in tubing', 'p': 'The present invention provides a method of forming a flow restriction in a tube to be used as a capillary in a refrigeration system. The method includes shaping the tube longitudinally to a preselected configuration, and then squeezing a section of the tube while directing a fluid therethrough until the fluid reaches a preselected pressure.'},
             {'id_kipi': 'US003967490A_19760706', 'pd': '19760706', 'cpc': 'G01N9/002', 'title': 'Vibration densitometer', 'p': 'A vibration densitometer having a strain gage pick-up to serve at least four functions one at a time or to serve two, three or four functions in any combination of two, three or four. The strain gage is bonded to or may embody a vibrating vane or body to be immersed in a fluid. The strain gage may be bonded to a vane with glass or an epoxy or other uniting agent. If glass is used, the densitometer may undergo unusually high temperatures and still perform reliably and accurately. The strain gage may be used (1) as a pick-up in an electromechanical oscillator embodied in the densitometer, (2) as a portion of a fluid temperature indicator, (3) for instrument frequency error correction (computable in two ways) as function of temperature, and (4) for producing an indication of what the fluid density would be at a selected constant temperature.'}]
    
        """
        # print(df.head())
        # print(df['section'].unique())       # ['C' 'G' 'B' 'A' 'H' 'D' 'F' 'E' 'Y']
        print(df2.groupby('section').count())

    # print(df.pivot_table('p', index=['section']))


# df = pd.read_json(file_list[0], orient='split')        # for k, v in loads(json, precise_float=self.precise_float).items(), ValueError: Expected object or value
# df = pd.read_json(file_list[0], orient='records')      # loads(json, precise_float=self.precise_float), dtype=None, ValueError: Expected object or value
# df = pd.read_json(file_list[0], orient='index')        # loads(json, precise_float=self.precise_float), ValueError: Expected object or value
# df = pd.read_json(file_list[0], orient='columns')      # loads(json, precise_float=self.precise_float), dtype=None, ValueError: Expected object or value
# df = pd.read_json(file_list[0], orient='values')       # loads(json, precise_float=self.precise_float), dtype=None, ValueError: Expected object or value
# print(df.head())

