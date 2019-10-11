# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import pickle
import json
import numpy as np
from tqdm import tqdm
from random import shuffle
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

tf.compat.v1.enable_eager_execution

data_path = '../data/'
EMB_SIZE = 300
RNG_SEED = 100   # 어제 실험한 것과 오늘 실험한게 일관성을 가지려면 초기값 고정 필요
BATCH_SIZE = 8
NUM_EPOCHS = 1
label_id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'Y': 8}
id_label = {i: l for l, i in label_id.items()}

tf.keras.backend.clear_session()

max_length = 10
TEST_SPLIT = 0.7

raw_path = '../Raw_data'
tr_file_list = os.listdir(raw_path)
tr_file_list = [file for file in tr_file_list if file.endswith(".txt")]
# file_list = [file for file in file_list if file.endswith("docs.txt")]

test_data_path = '../data/Raw_data/TestData'
test_file_list = os.listdir(test_data_path)
test_file_list = [file for file in test_file_list if file.endswith(".txt")]

print('Loading Saved Tokenizer')
token_file = os.path.join('../data/', 'abst_tokenizer.pickle')
with open(token_file, 'rb') as handle:
    tokenizer = pickle.load(handle)
VOCAB_SIZE = len(tokenizer.word_index)


def read_data(file):
    title = []
    label = []
    raw_data = open(os.path.join('../Raw_Claim', file), encoding='utf-8')
    for line in raw_data:
        try:
            d = json.loads(line)    # 이 과정을 생략하면 str타입으로 읽어서 append함
            title.append(d['title'])
            label.append(label_id[d["cpc"][0].lower()])
        except:
            pass
    title_token = tokenizer.texts_to_sequences(title)
    title_token = preprocessing.sequence.pad_sequences(title_token,
                                                       maxlen=max_length,
                                                       padding='pre')  # 학습 데이터를 벡터화
    return label, title_token


def mapping_fn(X, Y):
    inputs, label = {'x': X}, Y
    return inputs, label


def read_lines(indices, path):
    line_count = 0
    questions = []
    answers = []

    with open(path, 'r') as f:
        for line in f.readlines():
            if line_count in indices:
                try:
                    question, answer, _ = next(csv.reader([line], skipinitialspace=True))
                    questions.append(question)
                    answers.append(answer)
                except:
                    print(line)
                    print(line_count)
            line_count += 1
    return questions, answers


def data_generator(is_train):
    if is_train:
        is_shuffle = True  # 셔플을 여기서 해줘야해. 밖에서는 느려
    else:
        is_shuffle = False

    for file in shuffle(tr_file_list):
        with open(os.path.join('../Raw_Claim', file), encoding='utf-8') as f:    # 일단 읽어서 길이는 알아둔다.
            data_length = len(f.readlines())
            print('Num of pat: ', data_length)

    indices = list(range(data_length))    # 인덱스를 미리 만들어주는게 제너레이터 사용의 핵심.
    if is_shuffle:
        shuffle(indices)    # 셔플할꺼라면 이걸... 내장 라이브러리 random에 있는 함수.

    current_count = 0
    while True:
        if current_count >= data_length:
            return
        else:
            target_indices = indices[current_count:current_count + BATCH_SIZE]
            questions, answers = self.read_lines(target_indices, path)

            tokenized_questions = self.tokenize_by_morph(questions)
            tokenized_answers = self.tokenize_by_morph(answers)

            tokenized_encoder_inputs = tokenized_questions  # teacher forcing을 써보자
            tokenized_decoder_inputs, tokenized_labels = self.make_decoder_input_and_label(tokenized_answers)

            indexed_encoder_inputs = self.text_to_sequence(tokenized_encoder_inputs)
            indexed_decoder_inputs = self.text_to_sequence(tokenized_decoder_inputs)
            indexed_labels = self.text_to_sequence(tokenized_labels)

            padded_encoder_inputs = pad_sequences(indexed_encoder_inputs,
                                                  maxlen=self.max_length,
                                                  padding='pre')
            padded_decoder_inputs = pad_sequences(indexed_decoder_inputs,
                                                  maxlen=self.max_length,
                                                  padding='pre')

            padded_labels = pad_sequences(indexed_labels,
                                          maxlen=self.max_length,
                                          padding='pre')

            yield padded_encoder_inputs, padded_decoder_inputs, padded_labels

    a = 0
    yield a


def train_input_fn():
    dataset = tf.data.Dataset.from_generator(generator=lambda: data_generator(is_train=True))
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_generator(generator=lambda: data_generator(is_train=False))
    dataset = dataset.map(mapping_fn)
    return dataset


def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    # feature['x'] => (bs, 20)

    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMB_SIZE)(features['x'])  # (bs, 20, EMD_SIZE)

    dropout_emb = tf.keras.layers.Dropout(rate=0.5)(embedding_layer)  # (bs, 20, EMD_SIZE)

    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    for filter_size in filter_sizes:
        conv = tf.keras.layers.Conv1D(
            filters=100,
            kernel_size=filter_size,
            padding='valid',
            activation=tf.nn.relu,
            kernel_constraint=tf.keras.constraints.max_norm(3.))(dropout_emb)  # (bs, 20, 100)
        # 최대 norm 지정, weight clipping이 바로 이 부분

        pool = tf.keras.layers.GlobalMaxPool1D()(conv)  # [(bs, 100), (bs, 100), (bs, 100)]
        pooled_outputs.append(pool)

    h_pool = tf.concat(pooled_outputs, axis=1)  # (bs, 300)

    hidden = tf.keras.layers.Dense(units=250, activation=tf.nn.relu,
                                   kernel_constraint=tf.keras.constraints.max_norm(3.))(h_pool)  # (bs, 200)
    dropout_hidden = tf.keras.layers.Dropout(rate=0.5)(hidden, training=TRAIN)
    # logits = tf.keras.layers.Dense(units=1)(dropout_hidden)  # sigmoid를 해주겠다  # (bs, 1)
    logits = tf.keras.layers.Dense(units=9)(dropout_hidden)  # 이렇게하면 one-hot 필요

    if labels is not None:
        # labels = tf.reshape(labels, [-1, 1])  # (bs, 1)
        print('labels: ', labels)
        labels = tf.one_hot(indices=labels, depth=9)  # (bs, 2)
        print('labels one_hot: ', labels)

    if TRAIN:
        global_step = tf.train.get_global_step()
        # loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        loss = tf.losses.softmax_cross_entropy(labels, logits)

        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

    elif EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})

    elif PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'prob': tf.nn.sigmoid(logits),
            }
        )
    # plot_model(model, to_file='model.png')


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

est = tf.estimator.Estimator(model_fn=model_fn,
                             model_dir="abst_to_section/checkpoint")

est.train(train_input_fn)
valid = est.evaluate(eval_input_fn)
