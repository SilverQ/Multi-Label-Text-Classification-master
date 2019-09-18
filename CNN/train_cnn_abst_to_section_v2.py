# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import pickle
import json
import numpy as np
from tqdm import tqdm
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

# tf.compat.v1.enable_eager_execution
tf.enable_eager_execution()

data_path = '../data/'
EMB_SIZE = 100
RNG_SEED = 100   # 어제 실험한 것과 오늘 실험한게 일관성을 가지려면 초기값 고정 필요
BATCH_SIZE = 64
NUM_EPOCHS = 2
label_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'y': 8}
id_label = {i: l for l, i in label_id.items()}

tf.keras.backend.clear_session()

max_length = 50
TEST_SPLIT = 0.7

raw_path = '../data/Raw_data'
file_names = os.listdir(raw_path)
file_names = [file for file in file_names if file.endswith(".txt")]
# file_names = [file for file in file_list if file.endswith("docs.txt")]
filenames = [os.path.join('../data/Raw_data', file_name) for file_name in file_names]
print(file_names)

test_data_path = '../data/Raw_data/TestData'
test_file_names = os.listdir(test_data_path)
test_file_names = [file for file in test_file_names if file.endswith(".txt")]

print('Loading Saved Tokenizer')
token_file = os.path.join('../data/', 'abst_tokenizer.pickle')
with open(token_file, 'rb') as handle:
    tokenizer = pickle.load(handle)
VOCAB_SIZE = len(tokenizer.word_index)

# https://www.tensorflow.org/beta/guide/data
"""
[Consuming text data]
  See Loading Text for an end to end example.
  Many datasets are distributed as one or more text files.
  The tf.data.TextLineDataset provides an easy way to extract lines from one or more text files.
  Given one or more filenames, a TextLineDataset will produce one string-valued element per line of those files.
"""
# https://www.tensorflow.org/beta/tutorials/load_data/text

dataset = tf.data.TextLineDataset(filenames=filenames)

for line in dataset.take(2):
    # print(line)
    print(json.loads(line))


def _parse_function(example):
  features = {"id_kipi": tf.VarLenFeature(tf.string),
              "pd": tf.VarLenFeature(tf.string),
              "cpc": tf.VarLenFeature(tf.string),
              "title": tf.VarLenFeature(tf.string),
              "p": tf.VarLenFeature(tf.string)
              }
  parsed_features = tf.parse_single_example(example, features)
  # parsed_features = tf.parse_tensor(dict(example), features)
  return parsed_features["cpc"], parsed_features["p"]

"""
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

# data = _parse_function(np.array(dataset.take(1)))
# print(data)

# dataset = dataset.map(_parse_function)
#
# for line in dataset.take(1):
#     print(line)
#     # for item in line:
#     #     print(item)
#     # tf.Tensor(b'{"id_kipi":"US004067857A_19780110","pd":"19780110","cpc":"C08G63/87|C08G63/","title":"Polyester catalyst system comprising an antimony-containing polycondensation catalyst and an ethylenically unsaturated compound and process employing same","p":"A polycondensation catalyst system comprising an antimony-containing polycondensation catalyst and selected ethylenically unsaturated compounds, such as pentaerythritol triacrylate. The ethylenically unsaturated compound stabilizes tne antimony-containing catalyst."}', shape=(), dtype=string)
#     # for key, value in dict(line).items():
#     #     print("  {!r:20s}: {}".format(key, value))

# def read_data(file):
#     title = []
#     label = []
#     raw_data = open(os.path.join('../data/Raw_data', file), encoding='utf-8')
#     for line in raw_data:
#         try:
#             d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
#             title.append(d['title'] + ', ' + d['p'])
#             label.append(label_id[d["cpc"][0].lower()])
#         except:
#             pass
#     title_token = tokenizer.texts_to_sequences(title)
#     title_token = preprocessing.sequence.pad_sequences(title_token,
#                                                        maxlen=max_length,
#                                                        padding='pre')  # 학습 데이터를 벡터화
#     # title2 = tokenizer.sequences_to_texts(title_token)
#     # print('title: \n', title, '\n', title_token, '\n', title2)
#     return label, title_token
#
#
# def read_data_v2(file_list):
#     for file in tqdm(file_list):
#         print('processing file: ', file)
#         title = []
#         label = []
#         raw_data = open(os.path.join('../data/Raw_data', file), encoding='utf-8')
#         for line in raw_data:
#             try:
#                 d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
#                 title.append(d['title'] + ', ' + d['p'])
#                 label.append(label_id[d["cpc"][0].lower()])
#             except:
#                 pass
#         title_token = tokenizer.texts_to_sequences(title)
#         title_token = preprocessing.sequence.pad_sequences(title_token,
#                                                            maxlen=max_length,
#                                                            padding='pre')  # 학습 데이터를 벡터화
#         # title2 = tokenizer.sequences_to_texts(title_token)
#         # label = np.array(label)
#         # title_token = np.array(title_token)
#         # return_array = np.array([label, title_token])
#         # print('\nlabel: ', label.shape, '\n', 'token: ', title_token.shape)
#         # print('\nreturn: ', return_array.shape)
#         yield [title_token, label]
#
# # def train_input_fn():
# #     dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
# #     dataset = dataset.shuffle(buffer_size=len(input_train))
# #     dataset = dataset.batch(BATCH_SIZE)
# #     dataset = dataset.map(mapping_fn)
# #     dataset = dataset.repeat(count=NUM_EPOCHS)
# #
# #     return dataset
#
#
# # dataset = (tf.data.TextLineDataset(source_csv).map(decode_csv))  # Read text file
#
# # [input_eval, label_eval] = read_data()
# #
# # input_train, input_eval, label_train, label_eval = train_test_split(input_eval, label_eval,
# #                                                                     test_size=TEST_SPLIT,
# #                                                                     random_state=RNG_SEED)
# # print('\n')
# # print('input_train: ', input_train)
# # print('input_eval: ', input_eval)
# # print('label_train: ', label_train)
# # print('label_eval: ', label_eval)
#
#
# def mapping_fn(X, Y):
#     inputs, label = {'x': X}, Y
#     return inputs, label
#
#
# def train_input_fn():
#     dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
#     dataset = dataset.shuffle(buffer_size=len(input_train))
#     dataset = dataset.batch(BATCH_SIZE)
#     dataset = dataset.map(mapping_fn)
#     dataset = dataset.repeat(count=NUM_EPOCHS)
#     iterator = dataset.make_one_shot_iterator()
#     return iterator
#
#
# def eval_input_fn():
#     dataset = tf.data.Dataset.from_tensor_slices((input_eval, label_eval))
#     #     dataset = dataset.shuffle(buffer_size=len(input_eval))
#     dataset = dataset.batch(BATCH_SIZE)
#     dataset = dataset.map(mapping_fn)
#     return dataset
#
#
# def model_fn(features, labels, mode, params):
#     TRAIN = mode == tf.estimator.ModeKeys.TRAIN
#     EVAL = mode == tf.estimator.ModeKeys.EVAL
#     PREDICT = mode == tf.estimator.ModeKeys.PREDICT
#     # feature['x'] => (bs, 20)
#
#     embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMB_SIZE)(features['x'])  # (bs, 20, EMD_SIZE)
#
#     dropout_emb = tf.keras.layers.Dropout(rate=0.5)(embedding_layer)  # (bs, 20, EMD_SIZE)
#
#     filter_sizes = [3, 4, 5]
#     pooled_outputs = []
#     for filter_size in filter_sizes:
#         conv = tf.keras.layers.Conv1D(
#             filters=100,
#             kernel_size=filter_size,
#             padding='valid',
#             activation=tf.nn.relu,
#             kernel_constraint=tf.keras.constraints.max_norm(3.))(dropout_emb)  # (bs, 20, 100)
#         # 최대 norm 지정, weight clipping이 바로 이 부분
#
#         pool = tf.keras.layers.GlobalMaxPool1D()(conv)  # [(bs, 100), (bs, 100), (bs, 100)]
#         pooled_outputs.append(pool)
#
#     h_pool = tf.concat(pooled_outputs, axis=1)  # (bs, 300)
#
#     hidden = tf.keras.layers.Dense(units=250, activation=tf.nn.relu,
#                                    kernel_constraint=tf.keras.constraints.max_norm(3.))(h_pool)  # (bs, 200)
#     dropout_hidden = tf.keras.layers.Dropout(rate=0.5)(hidden, training=TRAIN)
#     # logits = tf.keras.layers.Dense(units=1)(dropout_hidden)  # sigmoid를 해주겠다  # (bs, 1)
#     logits = tf.keras.layers.Dense(units=9)(dropout_hidden)  # 이렇게하면 one-hot 필요
#
#     if labels is not None:
#         # labels = tf.reshape(labels, [-1, 1])  # (bs, 1)
#         print('labels: ', labels)
#         labels = tf.one_hot(indices=labels, depth=9)  # (bs, 2)
#         print('labels one_hot: ', labels)
#
#     if TRAIN:
#         global_step = tf.train.get_global_step()
#         # loss = tf.losses.sigmoid_cross_entropy(labels, logits)
#         loss = tf.losses.softmax_cross_entropy(labels, logits)
#
#         train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)
#
#         return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)
#
#     elif EVAL:
#         loss = tf.losses.sigmoid_cross_entropy(labels, logits)
#         pred = tf.nn.sigmoid(logits)
#         accuracy = tf.metrics.accuracy(labels, tf.round(pred))
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})
#
#     elif PREDICT:
#         return tf.estimator.EstimatorSpec(
#             mode=mode,
#             predictions={
#                 'prob': tf.nn.sigmoid(logits),
#             }
#         )
#     # plot_model(model, to_file='model.png')
#
#
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
#
# est = tf.estimator.Estimator(model_fn, model_dir="abst_to_section/checkpoint")
#
#
# # for file in tqdm(file_list):
# #     [label, title_token] = read_data(file)
# #
# #     input_train, input_eval, label_train, label_eval = train_test_split(title_token, label,
# #                                                                         test_size=TEST_SPLIT,
# #                                                                         random_state=RNG_SEED)
# #     # print('\n')
# #     # print('title_token: ', title_token)
# #     # print('label_train: ', label_train)
# #
# #     est.train(train_input_fn)
# #     valid = est.evaluate(eval_input_fn)
#
# # [label, title_token] = read_data_v2(file_list)
# read_file = next(read_data_v2(file_list))
# print('label: ', read_file)
#
# # input_train, input_eval, label_train, label_eval = train_test_split(title_token, label,
# #                                                                     test_size=TEST_SPLIT,
# #                                                                     random_state=RNG_SEED)
# # # print('\n')
# # # print('title_token: ', title_token)
# # # print('label_train: ', label_train)
# #
# # est.train(train_input_fn)
# # valid = est.evaluate(eval_input_fn)
#
#
# # [test_label, test_title_token] = read_data(test_file_list[0])
# #
# #
# # def test_input_fn():
# #     dataset = tf.data.Dataset.from_tensor_slices((test_title_token, test_label))
# #     dataset = dataset.batch(BATCH_SIZE)
# #     dataset = dataset.map(mapping_fn)
# #     iterator = dataset.make_one_shot_iterator()
# #
# #     return iterator.get_next()
# #
# #
# # test_output = [pred['prob'] for pred in est.predict(test_input_fn)]
# # test_output = np.argmax(np.array(test_output), axis=1)
# # # test_label = test_label.numpy()
# #
# # print('test data shape: ', np.shape(test_output))
# # print('top3 label and tokens: ', test_label[0:3], '\n', test_title_token[0:3])
# # test_title = tokenizer.sequences_to_texts(test_title_token[0:3])
# # print('title test: ', test_title)
# # print('expected out: ', test_output[0:3])
# # print('expected label: ', [id_label[i] for i in test_output[:3]])
# #
# # acc = sum(1 for a, b in zip(test_label, test_output) if a == b) / len(test_label) * 100.0
# # print('test accuracy : ', acc)      # 74.33373324719804
# #
# # error_list = np.array([[a, b, a == b] for a, b in zip(test_label, test_output)])
# # print(error_list[0:30])
# # error_list = np.array([i for i, item in enumerate(error_list) if item[2] == 0])
# #
# # # print(error_list)
# # print('label: ', id_label[error_list[0]], 'expected label: ', id_label[test_label[error_list[0]]])
# # print(list(test_title_token[error_list[0]]))
# # error_title = tokenizer.sequences_to_texts(list(test_title_token[error_list[0]]))
# # print('error_title: ', error_title)
# # # title2 = tokenizer.sequences_to_texts(title_token)
