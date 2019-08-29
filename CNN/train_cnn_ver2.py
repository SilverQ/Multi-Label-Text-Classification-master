# -*- coding:utf-8 -*-
import sys
import tensorflow as tf
import os
import pickle
import time
import numpy as np
import json
from tensorflow.keras import preprocessing
from tensorboard.plugins import projector

# import keras
# import logging
# from text_cnn import TextCNN
# tf.enable_eager_execution()

data_path = '../data/'
one_hot_file = os.path.join('../data/', 'tokenizer.pickle')
BATCH_SIZE = 10
EPOCH_SIZE = 2

tf.keras.backend.clear_session()

# raw_path = '../data/Raw_data'
# file_list = ['Section_Label.txt', 'Section_Title.txt']


def write_file(input_data, file_path):
    # file_path = os.path.join('../data/', file_name)
    with open(file_path, 'wb') as write_file1:
        pickle.dump(input_data, write_file1)
        print('Writing File {} finished'.format(file_path))


def tokenize(samples):
    # Check One Hot
    if os.path.isfile(one_hot_file):
        print('Loading Saved Tokenizer')
        with open(one_hot_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        # print(len(targets))     # 9362172
        # print(len(samples))     # 9362172
        tokenizer = preprocessing.text.Tokenizer()
        # tokenizer = preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        #                                          lower=True, split=' ', char_level=False,
        #                                          oov_token=None, document_count=0)

        tokenizer.fit_on_texts(samples)  # 토크나이저 학습은 전체 데이터를 대상으로!!
        # 그러면 실 서비스 환경에서는 신조어들이 막 튀어나올텐데, 그때는 에러나나?
        # 참고 : https://subinium.github.io/Keras-6-1/
        with open(one_hot_file, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def train_section_title():
    def read_txt(file):
        with open(file, encoding='utf-8') as data_file:
            for l in data_file:
                yield json.loads(l)

    def read_sequence(sequence):
        for i in sequence:
            yield i

    with open(os.path.join('../data', 'Section_Title.txt'), encoding='utf-8') as data_file:
        samples = [json.loads(line) for line in data_file]

    def gen_fn(samples):
        with open(os.path.join('../data', 'Section_Title.txt'), encoding='utf-8') as data_file:
            for line in data_file:
                yield json.loads(line)

    tokenizer = tokenize(samples)

    word_index = tokenizer.word_index       # 계산된 단어 인덱스
    # print('Found %s unique tokens.' % len(word_index))    # Found 252430 unique tokens.

    sequences = tokenizer.texts_to_sequences(gen_fn(samples))
    sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post')

    # print(sequences[:3])

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
    # dataset = tf.data.Dataset.from_tensor_slices(targets)  # sequence를 tokenize하는게 너무 무거운듯.
    dataset = dataset.shuffle(buffer_size=100).repeat(EPOCH_SIZE).batch(BATCH_SIZE)
    # buffer_size: A tf.int64 scalar tf.Tensor,
    # representing the maximum number elements that will be buffered when prefetching.
    # 1000으로 낮춰서 실행은 성공함
    iterator = dataset.make_one_shot_iterator()  # 없어도 됨,
    # If using `tf.estimator`, return the `Dataset` object directly from your input function
    next_data = iterator.get_next()

    def view_sample_data():
        with tf.Session() as sess:
            seq, lab = next_data
            print(sess.run([seq, lab]))
            # lab = next_data
            # print(sess.run([lab]))
            """
            [array([   90,    15,  6913,    62, 10744,   282,     1,    15,  7230,
                    1832,   267,     1,    19,   373,    13], dtype=int32), array([b'C'], dtype=object)]
            after
            [array([[b'B'],
                    [b'C'],
                    [b'F'],
                    [b'A'],
                    [b'C'],
                    [b'C'],
                    [b'B'],
                    [b'B'],
                    [b'B'],
                    [b'H']], dtype=object)]
            """

    view_sample_data()

    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             seq, lab = next_data
    #             print(sess.run([seq, lab]))
    #         except:
    #             break  # 갯수가 끝나면 에러뜨면서 종료


if __name__ == '__main__':
    train_section_title()
