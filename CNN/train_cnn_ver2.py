# -*- coding:utf-8 -*-
import sys
import tensorflow as tf
import os
import time
import numpy as np
import json
from tensorflow.keras import preprocessing
from tensorboard.plugins import projector

# import keras
# import logging
# from text_cnn import TextCNN
# tf.enable_eager_execution()


def train_section_title():
    # raw_path = '../data/Raw_data'
    # file_list = ['Section_Label.txt', 'Section_Title.txt']

    with open(os.path.join('../data', 'Section_Title.txt'), encoding='utf-8') as data_file:
        samples = [json.loads(line) for line in data_file]
    # print(len(targets))     # 9362172
    # print(len(samples))     # 9362172

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(samples)  # 토크나이저 학습은 전체 데이터를 대상으로!!
    # 그러면 실 서비스 환경에서는 신조어들이 막 튀어나올텐데, 그때는 에러나나?

    sequences = tokenizer.texts_to_sequences(samples)
    sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post')

    with open(os.path.join('../data', 'Section_Label.txt'), encoding='utf-8') as data_file:
        targets = [[json.loads(line)] for line in data_file]
    targets = np.array(targets)
    word_index = tokenizer.word_index
    # print("index text data : \n", sequences)
    print("shape of sequences:", sequences.shape)
    # print("index of each word : \n", word_index)
    # print("targets: \n", targets)
    print("shape of targets:", targets.shape)  # shape of targets: (6, 1)

    # [기본 코드]
    dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))  # 튜플로 감싸서 넣으면 tf.data가 알아서 잘라서 쓴다
    iterator = dataset.make_one_shot_iterator()  # 없어도 됨, If using `tf.estimator`, return the `Dataset` object directly from your input function
    next_data = iterator.get_next()

    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             seq, lab = next_data
    #             print(sess.run([seq, lab]))
    #         except:
    #             break  # 갯수가 끝나면 에러뜨면서 종료

    #
    # tokenizer = preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
    #                                          split=' ', char_level=False, oov_token=None, document_count=0)
    # # tokenizer = preprocessing.text.Tokenizer()
    # dataset = tf.data.TextLineDataset(file_list)
    # # tokenizer.fit_on_texts(dataset)  # 토크나이저 학습은 전체 데이터를 대상으로!!
    # # 그러면 실 서비스 환경에서는 신조어들이 막 튀어나올텐데, 그때는 에러나나?
    #
    # BATCH_SIZE = 10
    # EPOCH_SIZE = 2
    #
    # # dataset = tf.data.TextLineDataset(file_list).map(parse_fn)
    # dataset = dataset.shuffle(buffer_size=10000).repeat(EPOCH_SIZE).batch(BATCH_SIZE)
    # iterator = dataset.make_one_shot_iterator()
    # next_data = iterator.get_next()
    # # idkipi, pd, cpc, title, p = iterator.get_next()
    #
    # # dataset = dataset.shuffle(buffer_size=10000)       # 셔플을 먼저 해주고 배치를 자르자.
    # # dataset = dataset.batch(BATCH_SIZE)
    # # dataset = dataset.repeat(EPOCH_SIZE)
    #
    # with tf.Session() as sess:
    #     tokenizer.fit_on_texts(dataset)  # 토크나이저 학습은 전체 데이터를 대상으로!!
    #     tmp = sess.run(next_data)
    #     print(tmp)
    #     # print('data: ', tmp('id_kipi'), len(tmp))
    #     # while True:
    #     #     try:
    #     #         # seq, lab = next_data
    #     #         print('data: ', sess.run(next_data))
    #     #     except:
    #     #         break  # 갯수가 끝나면 에러뜨면서 종료


if __name__ == '__main__':
    # train_cnn()
    # logging_part()
    train_section_title()
