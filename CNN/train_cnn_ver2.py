# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import pickle
import json
from tensorflow.keras import preprocessing

tf.enable_eager_execution()

data_path = '../data/'
source_txt = os.path.join('../data', 'title_Section.txt')
# one_hot_file = os.path.join('../data/', 'tokenizer.pickle')
embedding_size = 100
BATCH_SIZE = 4
EPOCH_SIZE = 2
label_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'y': 8}

tf.keras.backend.clear_session()

source_csv = os.path.join('../data/', 'title_Section.csv')
feature_names = ['title']

# train_input_data = 'nsmc_train_input.npy'
# train_label_data = 'nsmc_train_label.npy'
# data_configs = 'data_configs.json'

# input_data = np.load(open(data_in_path + train_input_data, 'rb'))
# label_data = np.load(open(data_in_path + train_label_data, 'rb'))
# prepro_configs = json.load(open(data_in_path + data_configs, 'r'))

token_file = os.path.join('../data/', 'tokenizer.pickle')

if os.path.isfile(token_file):
    print('Loading Saved Tokenizer')
    with open(token_file, 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    print('Starting to Fit Tokenizer')
    data_file = open(source_txt, encoding='utf-8')
    # print(data_file)
    sentence = [json.loads(line)[1] for line in data_file]
    i = 0
    tokenizer = preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                             lower=False, split=' ', char_level=False,
                                             oov_token=None, document_count=0)
    tokenizer.fit_on_texts(sentence)
    with open(token_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Finished to fit Tokenizer')


def decode_csv(line):
    parsed_line = tf.decode_csv(line, [[0], ['']], field_delim='\t')
    # label = parsed_line[-1:]    # Last element is the label
    # del parsed_line[-1]         # Delete last element
    # features = parsed_line      # Everything (but last element) are the features
    # d = dict(zip(feature_names, features)), label
    sentence = parsed_line[-1:]
    print('sentence: ', sentence)
    sentence = tokenizer.texts_to_sequences(sentence)
    # sentence = preprocessing.sequence.pad_sequences(sentence, maxlen=6, padding='post')

    del parsed_line[-1]
    label = parsed_line
    d = dict(zip(feature_names, sentence)), label
    return d


dataset = (tf.data.TextLineDataset(source_csv).map(decode_csv))  # Read text file
# dataset = tf.data.TextLineDataset(source_csv)  # Read text file
# dataset = dataset.repeat(repeat_count)          # Repeats dataset this # times
dataset = dataset.batch(8)                      # Batch size to use
# iterator = dataset.make_one_shot_iterator()
# batch_features, batch_labels = iterator.get_next()
# print('batch_features, batch_labels: ', batch_features, batch_labels)

for line in dataset.take(5):
    print(line)

#
# def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
#
#     dataset = (tf.data.TextLineDataset(file_path).map(decode_csv)) # Read text file
#        # .skip(1) # Skip header row
#        # .map(decode_csv)) # Transform each elem by applying decode_csv fn
#     if perform_shuffle:
#         # Randomizes input using a window of 256 elements (read into memory)
#         dataset = dataset.shuffle(buffer_size=256)
#     dataset = dataset.repeat(repeat_count)          # Repeats dataset this # times
#     dataset = dataset.batch(8)                      # Batch size to use
#     iterator = dataset.make_one_shot_iterator()
#     batch_features, batch_labels = iterator.get_next()
#     # print('batch_features, batch_labels: ', batch_features, batch_labels)
#     return batch_features, batch_labels
#
#
# def train_section_title():
#     # sentence와 label이 서로 다른 파일로 구분된 경우
#     """
#     # tf.data에서 모든 데이터가 메모리에 로드되어야 해서 overflow 발생
#     # with open(os.path.join('../data', 'Section_Title.txt'), encoding='utf-8') as data_file:
#     #     samples = [json.loads(line) for line in data_file]
#     # sentence_file = open(os.path.join('../data', 'Section_Title.txt'), encoding='utf-8')
#     # label_file = open(os.path.join('../data', 'Section_Label.txt'), encoding='utf-8')
#     # input_file = np.array()
#     """
#
#     # Using json.loads
#     data_file = open(source_file, encoding='utf-8')
#     sentence = [json.loads(line)[1] for line in data_file]
#     print(sentence[0:3])
#
#     # Using pandas read_csv
#     """
#     # 실수 데이터는 소문자 변환이 안된다는 에러 메시지. 해당 데이터는 못보겠음.
#     # input_file = pd.read_csv(os.path.join('../data', 'title_Section.txt'), sep='\t',
#     #                          names=['Section', 'Title'], header=None, index_col=None, dtype=str)
#     # section = np.array(input_file['Section'])
#     # sentence = list(input_file['Title'].values)
#     # print(section[:3])
#     # print(sentence[0])
#     # print(len(input_file))      # 9362172
#     # pandas로 저장하면 TypeError가 유발된다. 이를 방지하는 방법은? 데이터를 구간별로 잘라서 fit하는건 너무 느려서 실행 포기.
#     """
#
#     tokenizer = fit_tokenize(sentence)    # 토크나이저 학습
#     word_index = tokenizer.word_index       # 계산된 단어 인덱스
#     print('Found %s unique tokens.' % len(word_index))
#     # without filter option : Found 252430 unique tokens.
#     # with filter option :    Found 406014 unique tokens. with tokenizer1
#
#     batch_size = BATCH_SIZE * 1000
#
#     # Self made iterator
#     """
#     def read_iter(file, b_size, max_epochs):
#         for i in range(max_epochs):
#             xb = []
#             yb = []
#             done = 0
#             for line in file:
#                 print('line: ', line, list(line))
#                 y = line[0]
#                 x = line[1]
#                 print('x: ', x)
#                 x = tokenizer.texts_to_sequences(x)
#                 print(x)
#                 x = preprocessing.sequence.pad_sequences(x, maxlen=15, padding='post')
#                 xb.append(x)
#                 yb.append(y)
#                 if len(xb) == b_size:
#                     yield xb, yb
#                     done += len(xb)
#                     print('xb: ', done)
#                     xb = []
#                     yb = []
#
#     def gen_fn(file):
#         # with open(os.path.join('../data', 'Section_Title.txt'), encoding='utf-8') as data_file:
#         for line in file:
#             yield json.loads(line)
#
#     for x, y in read_iter(data_file, batch_size, EPOCH_SIZE):
#         # sequences = tokenizer.texts_to_sequences(read_iter(sentence_file, batch_size, EPOCH_SIZE))
#         # sequences = tokenizer.texts_to_sequences(x)
#         # sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post')
#         print(x, y)
#         # targets = read_iter(label_file, batch_size, EPOCH_SIZE)
#
#         # print(sequences[:3])
#
#     # with open(os.path.join('../data', 'Section_Label.txt'), encoding='utf-8') as data_file:
#     #     targets = [[json.loads(line)] for line in data_file]
#     # targets = np.array(targets)
#     print(targets[:3])
#     word_index = tokenizer.word_index
#     # print("index text data : \n", sequences)
#     print("shape of sequences:", sequences.shape)
#     # print("index of each word : \n", word_index)
#     # print("targets: \n", targets)
#     print("shape of targets:", targets.shape)  # shape of targets: (6, 1)
#     """
#
#     # [기본 코드]
#     # dataset = tf.data.Dataset.from_tensor_slices((sentence, targets))  # 튜플로 감싸서 넣으면 tf.data가 알아서 잘라서 쓴다
#     dataset = tf.data.TextLineDataset(source_file)
#     # https://www.tensorflow.org/beta/guide/data
#
#     # for line in dataset.take(5):
#     #     print(line)
#     """
#     take 행위를 5번, 한번 할 때마다 레코드를 하나씩 가져온다, shape=()
#     tf.Tensor(b'["C", "Polyester catalyst system comprising an antimony-containing polycondensation catalyst and an ethylenically unsaturated compound and process employing same"]', shape=(), dtype=string)
#     tf.Tensor(b'["C", "Acid derivative immobilized cephalosporin carboxylic carrying polymer"]', shape=(), dtype=string)
#     tf.Tensor(b'["C", "Curable compositions containing chloroprene rubber"]', shape=(), dtype=string)
#     tf.Tensor(b'["C", "N,N\'-bis(3,4-dicyanophenyl) alkanediamide, polyphthalocyanines, and preparation thereof"]', shape=(), dtype=string)
#     tf.Tensor(b'["C", "Anionic polymerization of lactams in an extruder with controlled output rate"]', shape=(), dtype=string)
#     """
#
#     dataset = dataset.shuffle(buffer_size=100).repeat(EPOCH_SIZE).batch(BATCH_SIZE)
#
#     # for line in dataset.take(2):
#     #     print(line)
#     # 2번 가져오는데, 한번 가져올 때마다 shape=(10, )
#     """
#     tf.Tensor(
#     [b'["B", "Lithium ferrite catalysed oxidative dehydrogenation process"]'
#      b'["C", "Mixing two immiscible fluids of differing density"]'
#      b'["C", "3-Amino-2-aza benzoquinone diimines"]'
#      b'["G", "Method for forming a light transmission glass fiber equipped with an optical lens"]'
#      b'["C", "Heat-resistant flameproof compositions"]'
#      b'["C", "Condensation products of phosphine oxides"]'
#      b'["F", "Process for making radomes"]'
#      b'["C", "Novel di(aryl)methyl alkyl sulfones"]'
#      b'["C", "N,N\'-bis(3,4-dicyanophenyl) alkanediamide, polyphthalocyanines, and preparation thereof"]'
#      b'["A", "Pharmaceutical compositions containing cardiac glycoside"]'], shape=(10,), dtype=string)
#     tf.Tensor(
#     [b'["C", "Manufacture of arylamines"]'
#      b'["B", "Process for the production of formaldehyde"]'
#      b'["C", "Oral compositions containing trifluoromethyl phenyl bis-biguanides as antiplaque agents"]'
#      b'["C", "Reticulated anisotropic porous vitreous carbon"]'
#      b'["C", "Aqueous urea metal complex composition"]'
#      b'["C", "Stabilization of solutions of 3-isothiazolones"]'
#      b'["B", "Casting of articles containing calcined gypsum"]'
#      b'["C", "Preparation of 2-amino-n-butanol"]'
#      b'["C", "Process for the preparation of selectively and symmetrically di-halogenated ketals"]'
#      b'["A", "Treatment of psoriasis with 6-aminonicotinamide and thionicotinamide"]'], shape=(10,), dtype=string)
#     """
#
#     # buffer_size: A tf.int64 scalar tf.Tensor,
#     # representing the maximum number elements that will be buffered when prefetching.
#     # 1000으로 낮춰서 실행은 성공함
#     iterator = dataset.make_one_shot_iterator()  # 없어도 됨,
#     # If using `tf.estimator`, return the `Dataset` object directly from your input function
#     next_data = iterator.get_next()
#
#     lab = next_data
#     # print(lab)
#     # for line in lab:
#     #     # print(line.numpy().decode('utf-8'))
#     #     # print(ast.literal_eval(line.numpy().decode('utf-8'))[0])
#     #     print(eval(line.numpy().decode('utf-8'))[0])        # list로 바뀌는데, 샘플 데이터에서도 느려지는게 느껴진다. ㅠㅠ
#     #     # print([item.numpy() for item in line])
#     """
#     ["C", "Condensation products of phosphine oxides"]
#     ["B", "Container with improved heat-shrunk cellular sleeve"]
#     ["C", "Terpenophenols"]
#     ["C", "8-Benzimido substituted quinaphthalone derivatives"]
#     ["C", "Oral compositions containing trifluoromethyl phenyl bis-biguanides as antiplaque agents"]
#     ["C", "Method of preparing vinyl halide polymers and copolymers with polyolefins"]
#     ["C", "Heat-resistant flameproof compositions"]
#     ["D", "Copolycondensed vinylphosphonates and their use as flame retardants"]
#     ["C", "Process for preparing pyrazine precursor of methotrexate or an N-substituted derivative thereof and/or a di(lower)alkyl ester thereof"]
#     ["C", "Phenolic phosphites as stabilizers for polymers"]
#     """
#
#     def view_sample_data():
#         with tf.Session() as sess:
#             # seq, lab = next_data
#             # print(sess.run([seq, lab]))
#             """
#             [array([   90,    15,  6913,    62, 10744,   282,     1,    15,  7230,
#                     1832,   267,     1,    19,   373,    13], dtype=int32), array([b'C'], dtype=object)]
#             after
#             [array([[b'B'],
#                     [b'C'],
#                     [b'F'],
#                     [b'A'],
#                     [b'C'],
#                     [b'C'],
#                     [b'B'],
#                     [b'B'],
#                     [b'B'],
#                     [b'H']], dtype=object)]
#             """
#
#     # view_sample_data()
#
#     # with tf.Session() as sess:
#     #     while True:
#     #         try:
#     #             seq, lab = next_data
#     #             print(sess.run([seq, lab]))
#     #         except:
#     #             break  # 갯수가 끝나면 에러뜨면서 종료
#
#
# if __name__ == '__main__':
#     # tokenizer = load_tokenizer()
#     token_file = os.path.join('../data/', 'tokenizer.pickle')
#
#     if os.path.isfile(token_file):
#         print('Loading Saved Tokenizer')
#         with open(token_file, 'rb') as handle:
#             tokenizer = pickle.load(handle)
#     else:
#         print('Starting to Fit Tokenizer')
#         data_file = open(source_txt, encoding='utf-8')
#         # print(data_file)
#         sentence = [json.loads(line)[1] for line in data_file]
#         i = 0
#         tokenizer = preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#                                                  lower=False, split=' ', char_level=False,
#                                                  oov_token=None, document_count=0)
#         tokenizer.fit_on_texts(sentence)
#         with open(token_file, 'wb') as handle:
#             pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print('Finished to fit Tokenizer')
#
#     next_batch = my_input_fn(source_file, True, repeat_count=2)
#
#     with tf.Session() as sess:
#         first_batch = sess.run(next_batch)
#         print("title 1'st batch data: ", first_batch[0]['title'][1].decode())
#     # train_section_title()
#
#     # feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]
#     # print('feature_columns: ', feature_columns)
#     # # [_NumericColumn(key='title', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
#     #
#     # word_embedding_column = tf.feature_column.embedding_column(feature_columns, dimension=embedding_size)
#     # classifier = tf.estimator.DNNClassifier(feature_columns=[word_embedding_column],   # The input features to our model
#     #                                         hidden_units=[10, 10],              # Two layers, each with 10 neurons
#     #                                         n_classes=9,
#     #                                         model_dir=os.getcwd())          # Path to where checkpoints etc are stored
#     # classifier.train(input_fn=lambda: my_input_fn(source_file, True, 8))    # 현재 에러
#     #
#     # # https://developers-kr.googleblog.com/2017/09/introducing-tensorflow-datasets-and-estimators.html
#
#
#
# # raw_path = '../data/Raw_data'
#
#
# # def load_tokenizer():
# #     # Using json.loads
# #     source_txt = os.path.join('../data', 'title_Section.txt')
# #     data_file = open(source_txt, encoding='utf-8')
# #     print(data_file)
# #     sentence = [json.loads(line)[1] for line in data_file]
# #     print(sentence[0:3])
# #     tokenizer = fit_tokenize(sentence)    # 토크나이저 학습
# #     return tokenizer
#
# # import keras
# # import ast
# # import sys
# # import logging
# # import pandas as pd
# # import time
# # import numpy as np
# # from tensorboard.plugins import projector
# # from text_cnn import TextCNN
#
# # tf.enable_eager_execution()
#
# # cur_path = os.getcwd()
# # print(cur_path)
#
#
# # def fit_tokenize(samples):
# #     # Check One Hot
# #     if os.path.isfile(one_hot_file):
# #         print('Loading Saved Tokenizer')
# #         with open(one_hot_file, 'rb') as handle:
# #             tokenizer = pickle.load(handle)
# #     else:
# #         print('Starting to Fit Tokenizer')
# #         # print(samples[:3])
# #         # tokenizer = preprocessing.text.Tokenizer()
# #         i = 0
# #         # for line in samples:
# #         #     try:
# #         #         tokenizer.fit_on_texts(line)  # 토크나이저 학습은 전체 데이터를 대상으로!!
# #         #         i += 1
# #         #         if i % 100000 == 0:
# #         #             print("{}'th line processed".format(i))
# #         #             # print(line)
# #         #             # word_index = tokenizer.word_index  # 계산된 단어 인덱스
# #         #             # print(word_index)
# #         #             # print('Found %s unique tokens.' % len(word_index))  # Found 252430 unique tokens.
# #         #     except TypeError:
# #         #         print('TypeError: ', line)
# #         #     # finally:
# #         #     #     print(line)
# #         tokenizer = preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
# #                                                  lower=False, split=' ', char_level=False,
# #                                                  oov_token=None, document_count=0)
# #         tokenizer.fit_on_texts(samples)  # 토크나이저 학습은 전체 데이터를 대상으로!!
# #         # 그러면 실 서비스 환경에서는 신조어들이 막 튀어나올텐데, 그때는 에러나나?
# #         # 참고 : https://subinium.github.io/Keras-6-1/
# #         with open(one_hot_file, 'wb') as handle:
# #             pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# #         print('Finished to Save Tokenizer')
# #     return tokenizer
#
#
# # def write_file(input_data, file_path):
# #     # file_path = os.path.join('../data/', file_name)
# #     with open(file_path, 'wb') as write_file1:
# #         pickle.dump(input_data, write_file1)
# #         print('Writing File {} finished'.format(file_path))
#
#
# # def read_txt(file):
# #     with open(file, encoding='utf-8') as data_file:
# #         for l in data_file:
# #             yield json.loads(l)
#
#
# # def batch_iterator(file, batch_size, max_epochs):
# #     with open(file, encoding='utf-8') as data_file:
# #         for i in range(max_epochs):
# #             xb = []
# #             yb = []
# #             for ex in data_file:
# #                 x, y = ex
# #                 xb.append(x)
# #                 yb.append(y)
# #                 if len(xb) == batch_size:
# #                     yield xb, yb
# #                     xb, yb = [], []
#
#
# # def read_sequence(sequence):
# #     for i in sequence:
# #         yield i
#