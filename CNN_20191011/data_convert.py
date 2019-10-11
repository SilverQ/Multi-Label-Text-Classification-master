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


# raw_path = '../data/Raw_data'

"""
exam01 : 발명의 명칭 -> 섹션
exam02 : 발명의 명칭 + 요약 -> 섹션
exam03 : 발명의 명칭 + 요약 -> 서브클래스
exam04 : 발명의 명칭 + 요약 + 청구항 -> 서브클래스
"""

raw_path = '../Raw_Claim'  # new data with tl, abst, claim (20191009)

exam = 'exam04'     # 이걸 바꿔
trdata_path = os.path.join(raw_path, exam)
exam_cpc = {'exam01': 1, 'exam02': 1, 'exam03': 4, 'exam04': 4}
# print(trdata_path)      # ../data/Raw_Claim/trainingdata

file_list = os.listdir(raw_path)
file_list = [file for file in file_list if file.endswith(".txt")]
# file_list = ['cpc_json_0_3docs.txt']
print(file_list)

patnum_idx = dict()
cpc_idx = dict()
idx_cpc = dict()


def patnum_to_idx(patnum_idx, id_kipi):
    pat_cnt = len(patnum_idx)
    patnum_idx[pat_cnt] = id_kipi
    return pat_cnt


def cpc_to_idx(cpc_idx, idx_cpc, cpcs):
    cpc_labels = list(set([cpc[:exam_cpc[exam]] for cpc in cpcs.split("|")]))
    for label in cpc_labels:
        if label not in idx_cpc.keys():
                    cpc_cnt = len(cpc_idx)
                    cpc_idx[cpc_cnt] = label
                    idx_cpc[label] = cpc_cnt
    return cpc_labels


def data_split(input_data):
    tr_set, te_set = train_test_split(input_data, test_size=0.2, random_state=1)
    tr_set, va_set = train_test_split(tr_set, test_size=0.2, random_state=1)
    return tr_set, te_set, va_set


def write_file(data, file_path):
    with open(file_path, 'a+', encoding='utf-8') as write_json:
        try:
            write_json.write('\n'.join([json.dumps(d) for d in data]))
        except:
            print('Err occured')
        finally:
            pass


for file_num, file in enumerate(file_list):
    with open(os.path.join(raw_path, file), encoding='utf-8') as data_file:
        data = []
        content = []
        for line in data_file:
            # print(line)
            try:
                d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
                # print(d)
                test_id = patnum_to_idx(patnum_idx, d["id_kipi"])
                labels_index = d["cpc"]
                # print("cpc: ", labels_index)
                labels_index = cpc_to_idx(cpc_idx, idx_cpc, labels_index)
                # print("cpc_label: ", labels_index)
                title = d["title"]
                abst = d["ab"]
                claim = d["cl"]
                # features_content = d["p"].split()
                labels_num = len(labels_index)
                # c = d["title"] + " " + d["p"]
                if exam == 'exam01':
                    c = title                                   # exam01
                elif exam == 'exam02' or exam == 'exam03':
                    c = title + ' ' + abst                      # exam02, exam03
                elif exam == 'exam04':
                    c = title + ' ' + abst + ' ' + claim        # exam04
                # print("c: ", c)

                features_content = c.split()
                # print("features_content: ", features_content)

                d = {"testid": test_id,
                     "features_content": features_content,
                     "labels_index": labels_index,
                     "labels_num": labels_num}
                # print("d: ", d)
                data.append(d)
                content.append(c)

            except:
                pass
        train_set, test_set, val_set = data_split(data)

        # file_path = os.path.join(trdata_path, file_name)

        write_file(train_set, os.path.join(trdata_path, 'Train.json'))
        write_file(val_set, os.path.join(trdata_path, 'Validation.json'))
        write_file(test_set, os.path.join(trdata_path, 'Test.json'))
        write_file(content, os.path.join(trdata_path, 'content.txt'))

        print('Reading File_{} finished'.format(file_num+1))


def obj_dict(obj):
    return obj.__dict__


def write_file2(input_data, file_path):
    # file_path = os.path.join('../data/', file_name)
    with open(file_path, 'wb') as write_file1:
        pickle.dump(input_data, write_file1)
        print('Writing File {} finished'.format(file_path))


# print(patnum_idx)     # US001899016A_19330228
print(cpc_idx)        # C25
# print(idx_cpc)        # 833820

print(len(patnum_idx))  # 9362172
print(len(cpc_idx))     # 174833
print(len(idx_cpc))     # 174833

write_file2(cpc_idx, os.path.join(trdata_path, 'cpc_idx.json'))
write_file2(patnum_idx, os.path.join(trdata_path, 'patnum_idx.json'))


"""
when using abst
# print('Reading {} patents success'.format(len(data)))
# Reading File_95 finished

print(len(patnum_idx))  # 9362172
print(len(cpc_idx))     # 174833
print(len(idx_cpc))     # 174833

# Writing File cpc_idx.json finished
# Writing File patnum_idx.json finished
#
# Process finished with exit code 0
"""

"""
using claims
exam01 : 발명의 명칭 -> 섹션
{0: 'H', 1: 'A', 2: 'G', 3: 'Y', 4: 'B', 5: 'F', 6: 'C', 7: 'E', 8: 'D'}
1299999
9
9
Writing File ../data/Raw_Claim/exam01/cpc_idx.json finished
Writing File ../data/Raw_Claim/exam01/patnum_idx.json finished

exam02 : 발명의 명칭 + 요약 -> 섹션
{0: 'H', 1: 'A', 2: 'G', 3: 'Y', 4: 'B', 5: 'F', 6: 'C', 7: 'E', 8: 'D'}
1299999
9
9
Writing File ../data/Raw_Claim/exam02/cpc_idx.json finished
Writing File ../data/Raw_Claim/exam02/patnum_idx.json finished

exam03 : 발명의 명칭 + 요약 -> 서브클래스
{0: 'H04N', 1: 'H05K', 2: 'G06T', 3: 'A61B', 4: 'G06F', 5: 'H04H', 6: 'G11B', 7: 'G06Q', 8: 'G09G', 9: 'H04L', 10: 'G08C', 11: 'Y02D', 12: 'H04Q', 13: 'H04B', 14: 'H04J', 15: 'H04W', 16: 'H04R', 17: 'G01C', 18: 'G01P', 19: 'H03F', 20: 'G01H', 21: 'Y10T', 22: 'A63B', 23: 'A61F', 24: 'H01Q', 25: 'H04M', 26: 'Y02B', 27: 'G02C', 28: 'H03M', 29: 'H03G', 30: 'G10K', 31: 'G10L', 32: 'H04S', 33: 'B29B', 34: 'G08B', 35: 'G01S', 36: 'G07C', 37: 'Y04S', 38: 'G08G', 39: 'B62D', 40: 'B60W', 41: 'H02J', 42: 'B60L', 43: 'A63F', 44: 'G02B', 45: 'H05B', 46: 'H01C', 47: 'B32B', 48: 'H01R', 49: 'F21Y', 50: 'F21S', 51: 'H01L', 52: 'F21V', 53: 'F21K', 54: 'H02M', 55: 'G05F', 56: 'B23P', 57: 'A47F', 58: 'H01F', 59: 'G09F', 60: 'G01J', 61: 'B60Q', 62: 'F21W', 63: 'H01H', 64: 'G05D', 65: 'G03B', 66: 'H01J', 67: 'H05H', 68: 'C23C', 69: 'H01P', 70: 'Y10S', 71: 'Y02P', 72: 'H01G', 73: 'B01F', 74: 'C25D', 75: 'H01M', 76: 'B64C', 77: 'F28D', 78: 'F28F', 79: 'A01H', 80: 'E05Y', 81: 'E05D', 82: 'E21B', 83: 'F16H', 84: 'A01B', 85: 'B60K', 86: 'F16C', 87: 'A01C', 88: 'B05B', 89: 'A01M', 90: 'A01D', 91: 'A01F', 92: 'C12N', 93: 'C12M', 94: 'A01G', 95: 'B64D', 96: 'H05F', 97: 'Y02A', 98: 'C12Q', 99: 'A23V', 100: 'A23L', 101: 'A23D', 102: 'F16L', 103: 'A01J', 104: 'F16K', 105: 'A01K', 106: 'E01H', 107: 'A61L', 108: 'B29C', 109: 'C08L', 110: 'C08G', 111: 'F42B', 112: 'F41G', 113: 'F41A', 114: 'A01N', 115: 'C08F', 116: 'C09D', 117: 'A23B', 118: 'C07D', 119: 'C12P', 120: 'A21B', 121: 'A47J', 122: 'A21C', 123: 'A22C', 124: 'A22B', 125: 'B65B', 126: 'B01J', 127: 'A23P', 128: 'A23C', 129: 'A23G', 130: 'F25D', 131: 'A23J', 132: 'A61K', 133: 'A24B', 134: 'A24D', 135: 'B05D', 136: 'A24F', 137: 'A41B', 138: 'A41D', 139: 'A41C', 140: 'A41F', 141: 'A44B', 142: 'F41C', 143: 'A47G', 144: 'A41G', 145: 'A42B', 146: 'G01L', 147: 'A42C', 148: 'A45F', 149: 'A43D', 150: 'D10B', 151: 'D04B', 152: 'A43B', 153: 'A43C', 154: 'B60N', 155: 'B29K', 156: 'B29L', 157: 'A45D', 158: 'A44C', 159: 'A45B', 160: 'A61G', 161: 'A45C', 162: 'B65D', 163: 'D05B', 164: 'B26B', 165: 'A61M', 166: 'B41J', 167: 'A61Q', 168: 'A46B', 169: 'B25H', 170: 'A46D', 171: 'A47L', 172: 'A47C', 173: 'A47B', 174: 'A47K', 175: 'F16B', 176: 'D06F', 177: 'B68G', 178: 'G10G', 179: 'G10D', 180: 'G01N', 181: 'A61H', 182: 'A47D', 183: 'E05B', 184: 'B42F', 185: 'G02F', 186: 'F24F', 187: 'B63C', 188: 'Y02W', 189: 'F16J', 190: 'A23F', 191: 'G06K', 192: 'F24C', 193: 'E03C', 194: 'F16M', 195: 'B62K', 196: 'B25J', 197: 'B24D', 198: 'B25G', 199: 'B08B', 200: 'G05G', 201: 'B29D', 202: 'B22F', 203: 'C08K', 204: 'G01B', 205: 'G16H', 206: 'G01R', 207: 'A61N', 208: 'B60R', 209: 'G01G', 210: 'G01M', 211: 'G21F', 212: 'G01T', 213: 'A61J', 214: 'B23C', 215: 'B23B', 216: 'A61C', 217: 'F04C', 218: 'G01K', 219: 'H02K', 220: 'F16F', 221: 'G09B', 222: 'C07F', 223: 'B09C', 224: 'C11D', 225: 'C02F', 226: 'C09K', 227: 'C10G', 228: 'C07K', 229: 'C07C', 230: 'C07B', 231: 'C12Y', 232: 'A23K', 233: 'C01B', 234: 'C07H', 235: 'A61P', 236: 'B67D', 237: 'B64F', 238: 'G05B', 239: 'B01D', 240: 'B04B', 241: 'B01L', 242: 'B81C', 243: 'B81B', 244: 'B82Y', 245: 'G01D', 246: 'G06M', 247: 'B44C', 248: 'D04C', 249: 'A62B', 250: 'E06C', 251: 'E04G', 252: 'A62C', 253: 'A62D', 254: 'A63H', 255: 'A63C', 256: 'B63H', 257: 'G07F', 258: 'G06N', 259: 'A63G', 260: 'F25J', 261: 'F23J', 262: 'F26B', 263: 'Y02C', 264: 'F04D', 265: 'B04C', 266: 'F16N', 267: 'C10K', 268: 'C10J', 269: 'C10L', 270: 'C10B', 271: 'B07B', 272: 'F02C', 273: 'F02M', 274: 'Y02T', 275: 'F01N', 276: 'C08J', 277: 'C04B', 278: 'Y02E', 279: 'B23K', 280: 'C01G', 281: 'C01P', 282: 'C25B', 283: 'G21G', 284: 'G21C', 285: 'F27B', 286: 'C40B', 287: 'B02C', 288: 'B03C', 289: 'B03D', 290: 'E04H', 291: 'H02H', 292: 'B05C', 293: 'G03F', 294: 'F05D', 295: 'F01D', 296: 'B06B', 297: 'B07C', 298: 'B65G', 299: 'B43M', 300: 'B26D', 301: 'B60G', 302: 'B21B', 303: 'B21D', 304: 'B21J', 305: 'B21K', 306: 'B22C', 307: 'B22D', 308: 'F16D', 309: 'B60T', 310: 'C21D', 311: 'B23Q', 312: 'B25B', 313: 'E01C', 314: 'B28D', 315: 'E01D', 316: 'B27B', 317: 'B23D', 318: 'B23F', 319: 'B23G', 320: 'C03C', 321: 'B26F', 322: 'B24C', 323: 'C22C', 324: 'G01F', 325: 'B27M', 326: 'B24B', 327: 'B25C', 328: 'B25F', 329: 'F15B', 330: 'B25D', 331: 'B27C', 332: 'B62B', 333: 'B27D', 334: 'H01B', 335: 'D01G', 336: 'D04H', 337: 'B65C', 338: 'F02K', 339: 'B33Y', 340: 'G03G', 341: 'D06N', 342: 'B63B', 343: 'B60C', 344: 'C09J', 345: 'D01D', 346: 'C03B', 347: 'B30B', 348: 'B31B', 349: 'B64G', 350: 'C08H', 351: 'B65F', 352: 'B31F', 353: 'D03D', 354: 'B41F', 355: 'B41C', 356: 'B41M', 357: 'B44F', 358: 'D06P', 359: 'D06B', 360: 'H01S', 361: 'B65H', 362: 'E06B', 363: 'E05F', 364: 'B41L', 365: 'D21H', 366: 'B42C', 367: 'B42D', 368: 'B42B', 369: 'B27F', 370: 'B43K', 371: 'B43L', 372: 'D06Q', 373: 'B44D', 374: 'D06H', 375: 'B28B', 376: 'B60B', 377: 'B60D', 378: 'C22F', 379: 'B60J', 380: 'B60Y', 381: 'B66F', 382: 'B60H', 383: 'B60P', 384: 'B61D', 385: 'B60S', 386: 'B62J', 387: 'H02P', 388: 'F02P', 389: 'H03K', 390: 'F02N', 391: 'F02D', 392: 'F01L', 393: 'B61F', 394: 'B61G', 395: 'B61L', 396: 'B62M', 397: 'E04F', 398: 'F03D', 399: 'B66C', 400: 'F05B', 401: 'E02D', 402: 'B63G', 403: 'F02B', 404: 'F01P', 405: 'B64B', 406: 'F25B', 407: 'B67B', 408: 'H02G', 409: 'B31D', 410: 'B66B', 411: 'B66D', 412: 'E03B', 413: 'F17C', 414: 'H03H', 415: 'G01Q', 416: 'H02N', 417: 'C10M', 418: 'C01F', 419: 'C01D', 420: 'C05F', 421: 'F23G', 422: 'C30B', 423: 'C05G', 424: 'C05B', 425: 'C05C', 426: 'C05D', 427: 'C13K', 428: 'D21C', 429: 'C09C', 430: 'C07J', 431: 'C08B', 432: 'C11B', 433: 'C08C', 434: 'D01F', 435: 'E04D', 436: 'E04B', 437: 'C10C', 438: 'C09B', 439: 'C23F', 440: 'C09G', 441: 'C10N', 442: 'F23C', 443: 'F01K', 444: 'C11C', 445: 'F28B', 446: 'C12R', 447: 'A21D', 448: 'G07D', 449: 'F27D', 450: 'C21C', 451: 'C22B', 452: 'C23G', 453: 'F04B', 454: 'F05C', 455: 'C25C', 456: 'D06M', 457: 'D02G', 458: 'D02J', 459: 'D01H', 460: 'D04D', 461: 'D21D', 462: 'D21B', 463: 'D21F', 464: 'E01B', 465: 'E04C', 466: 'E01F', 467: 'E02F', 468: 'F24J', 469: 'H02S', 470: 'F24S', 471: 'E02B', 472: 'F41H', 473: 'E03D', 474: 'E03F', 475: 'B27N', 476: 'E05C', 477: 'G01V', 478: 'F22B', 479: 'F01B', 480: 'F23R', 481: 'F02G', 482: 'F01M', 483: 'F01C', 484: 'F02F', 485: 'F04F', 486: 'F24V', 487: 'F03B', 488: 'F03G', 489: 'F03H', 490: 'E21D', 491: 'F03C', 492: 'F16G', 493: 'F17D', 494: 'F23N', 495: 'F21L', 496: 'F23D', 497: 'F23K', 498: 'F24B', 499: 'F24D', 500: 'F24H', 501: 'F25C', 502: 'A63J', 503: 'F28C', 504: 'F41B', 505: 'F41J', 506: 'F42C', 507: 'A24C', 508: 'G01W', 509: 'G04G', 510: 'G03C', 511: 'B41N', 512: 'B44B', 513: 'G21K', 514: 'G03H', 515: 'G04B', 516: 'G04C', 517: 'G04F', 518: 'G04D', 519: 'H03L', 520: 'G04R', 521: 'H04K', 522: 'H03D', 523: 'G11C', 524: 'G10H', 525: 'G07G', 526: 'C14C', 527: 'C14B', 528: 'G09C', 529: 'G07B', 530: 'E05G', 531: 'G21B', 532: 'G21H', 533: 'H02B', 534: 'H05G', 535: 'H03B', 536: 'F23M', 537: 'C01C', 538: 'H01T', 539: 'B60M', 540: 'B63J', 541: 'H03J', 542: 'D21G', 543: 'A61D', 544: 'C12G', 545: 'A47H', 546: 'A23Y', 547: 'A63K', 548: 'F28G', 549: 'D06L', 550: 'B21C', 551: 'B21F', 552: 'B23H', 553: 'B28C', 554: 'A63D', 555: 'B41P', 556: 'B60F', 557: 'B61H', 558: 'E21F', 559: 'B61B', 560: 'B61C', 561: 'B62L', 562: 'D03J', 563: 'C12C', 564: 'C13B', 565: 'C21B', 566: 'B21H', 567: 'D05D', 568: 'F22G', 569: 'F41F', 570: 'F23Q', 571: 'F15D', 572: 'E21C', 573: 'F16P', 574: 'D05C', 575: 'F23L', 576: 'C06B', 577: 'F42D', 578: 'C12H', 579: 'G06E', 580: 'G21D', 581: 'B82B', 582: 'H01K', 583: 'G06G', 584: 'H03C', 585: 'A01L', 586: 'B27K', 587: 'B03B', 588: 'A41H', 589: 'B09B', 590: 'C25F', 591: 'B27G', 592: 'B62H', 593: 'D21J', 594: 'D01B', 595: 'D07B', 596: 'D06C', 597: 'F23B', 598: 'F24T', 599: 'C06C', 600: 'G06J', 601: 'A23N', 602: 'B67C', 603: 'B41K', 604: 'B42P', 605: 'C09F', 606: 'A44D', 607: 'F15C', 608: 'B41G', 609: 'C06D', 610: 'C07G', 611: 'F22D', 612: 'G06C', 613: 'G10C', 614: 'G10F', 615: 'B27L', 616: 'B68C', 617: 'B60V', 618: 'C10F', 619: 'F27M', 620: 'D03C', 621: 'B61K', 622: 'B31C', 623: 'B68B', 624: 'C23D', 625: 'D04G', 626: 'C12F', 627: 'G09D', 628: 'G21Y', 629: 'F23H', 630: 'H05C', 631: 'G12B', 632: 'B27J', 633: 'B01B', 634: 'F16S', 635: 'B02B', 636: 'D02H', 637: 'G10B', 638: 'B41D', 639: 'B27H', 640: 'D01C', 641: 'B61J', 642: 'B21L', 643: 'F16T', 644: 'B21G', 645: 'D06G', 646: 'G03D', 647: 'E02C', 648: 'B41B', 649: 'H04T', 650: 'A42D', 651: 'C09H', 652: 'C10H', 653: 'D06J', 654: 'B62C', 655: 'B68F', 656: 'C12J', 657: 'G06D', 658: 'C12L', 659: 'F17B', 660: 'G21J', 661: 'F21H'}
1299999
662
662
Writing File ../data/Raw_Claim/exam03/cpc_idx.json finished
Writing File ../data/Raw_Claim/exam03/patnum_idx.json finished
exam04 : 발명의 명칭 + 요약 + 청구항 -> 서브클래스
"""
