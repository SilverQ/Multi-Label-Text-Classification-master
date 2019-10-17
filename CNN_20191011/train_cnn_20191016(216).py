# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import pickle
import json
# from tqdm import tqdm
from random import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from collections import Counter
import re
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
# import spacy
# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
# nlp = English()
# tokenizer = nlp.Defaults.create_tokenizer(nlp)
# en = spacy.load('en_core_web_sm')
# en.pipeline = [en.tagger, en.parser]

tf.compat.v1.enable_eager_execution

# data_path = '../data/Raw_Claim/'
# EMB_SIZE = 300
# RNG_SEED = 100   # 어제 실험한 것과 오늘 실험한게 일관성을 가지려면 초기값 고정 필요
# BATCH_SIZE = 8
# NUM_EPOCHS = 1

tf.keras.backend.clear_session()

# max_length = 10
# TEST_SPLIT = 0.7

raw_path = '../data/Raw_Claim'
tr_file_list = os.listdir(raw_path)
tr_file_list = [file for file in tr_file_list if file.endswith(".txt")]
# file_list = [file for file in file_list if file.endswith("docs.txt")]

test_data_path = '../data/Raw_Claim/test_data'
test_file_list = os.listdir(test_data_path)
test_file_list = [file for file in test_file_list if file.endswith(".txt")]


class Dataset:

    def __init__(self, train_path, test_path, is_shuffle, train_bs, test_bs, epoch, max_length, vocab_path, label_path):
        self.train_path = train_path
        self.test_path = test_path
        self.is_shuffle = is_shuffle
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.epoch = epoch
        self.max_length = max_length
        # self.is_header_first = is_header_first
        # self.okt = Okt()
        self.special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.label_path = label_path
        self.vocab_path = vocab_path

        if not os.path.exists(self.vocab_path):
            print('No vocabulary.')
            print('Making vocabulary.')
            # self.build_vocab_by_patdata(vocab_path)
            self.build_vocab_by_patent(vocab_path)
            print('Complete build vocabulary!')

        # print('Loading vocabulary...')
        self.idx2word, self.word2idx = pickle.load(open(vocab_path, 'rb'))
        print('Successfully load vocabulary!')
        self.idx2label, self.label2idx = pickle.load(open(label_path, 'rb'))
        print('Successfully load labels')

    def build_freq(self, word_list):
        word_counts = Counter(word_list)
        # print('word_list: ', len(word_list), word_list)
        # print('word_counts_1: ', len(word_counts), word_counts)
        # print('word_counts_2: ', len(word_counts.most_common()), word_counts.most_common())
        freq = Counter()
        freq_file = raw_path + '/word_freq.pickle'
        # print(freq_file)
        if os.path.exists(freq_file):
            with open(freq_file, 'rb') as freq_dist_f:
                freq = pickle.load(freq_dist_f)
                # print('frequency distribution loaded', len(freq), freq)
        for word, cnt in word_counts.items():
            # print(word, freq[word])
            freq[word] += cnt
            # print(word, freq[word])
        print('freq len: ', len(freq))
        with open(freq_file, 'wb') as freq_dist_f:
            pickle.dump(freq, freq_dist_f)
        return freq

    def build_vocab_by_patent(self, vocab_path):
        error_cnt = 0
        label_list = []
        for file in self.train_path:
            word_list = []
            with open(os.path.join(raw_path, file), encoding='utf-8') as f:
                for line in f:
                    try:
                        # print('line: ', line)
                        patent = json.loads(line)
                        text = re.sub('[-=.#/?:$}(){,]', ' ', patent['title'] + patent['ab'] + patent['cl'])
                        token = text.split()
                        # token = tokenizer(patent['title'])
                        # print('token: ', token)
                        # doc = en.tokenizer(patent['title']+patent['ab']+patent['cl'])
                        labels = patent['cpc'].split('|')
                        for tok in token:
                            word_list.append(tok.lower())
                        labels = [label[0] for label in labels]
                        for label in labels:
                            if label not in label_list:
                                label_list.append(label)
                    except:
                        error_cnt += 1
                        # print('error: ', line)
            print('\nIn "%s" word_list: %d, error_cnt: %d\n' % (file, len(word_list), error_cnt))
            # idx2word, word2idx = self.build_freq(word_list)
            idx2word = self.build_freq(word_list)
            # idx2label = self.build_labels(label_list)
            # print('idx2word: ', len(idx2word), idx2word[:10])
        idx2word = self.special_tokens + [word for word, _ in idx2word.most_common()]
        # print('idx2word: ', len(idx2word), idx2word[:10])
        # print('idx2label: ', len(label_list), label_list)
        word2idx = {word: idx for idx, word in enumerate(idx2word)}
        label2idx = {label: idx for idx, label in enumerate(label_list)}
        vocab = (idx2word, word2idx)
        label = (label_list, label2idx)
        pickle.dump(vocab, open(vocab_path, 'wb'))
        pickle.dump(label, open(self.label_path, 'wb'))

    def text_to_sequence(self, text_list):
        sequences = []
        for text in text_list:
            sequences.append([self.word2idx[word] for word in text if word in self.word2idx.keys()])
        return sequences

    def sequence_to_text(self, sequence):
        return [self.idx2word[idx] for idx in sequence if idx != 0]

    def read_lines(self, indices, path):
        line_count = 0
        texts = []
        labels = []

        with open(path, encoding='utf-8') as f:
            for line in f:
                if line_count in indices:
                    try:
                        patent = json.loads(line)
                        text = re.sub('[-=.#/?:$}(){,]', ' ', patent['title'] + patent['ab'])
                        label = patent['cpc'].split('|')
                        texts.append(text.lower().split())
                        labels.append(list(set([cpc[0] for cpc in label])))
                    except:
                        print(line)
                        print(line_count)
                line_count += 1
        # print('texts: \n', len(texts), texts[:5])
        # print('\nlabels: \n', len(labels), labels)
        # print('\n')
        return texts, labels

    def create_multiplehot_labels(self, labels_index):
        labels = []
        # print(len(label))
        for batch in labels_index:
            label = [0] * len(self.label2idx)
            # print(item)
            for cpc in batch:
                label[self.label2idx[cpc]] = 1
            labels.append(label)
        # print('label_repr: ', labels)
        return labels

    def data_generator(self, is_train):
        if is_train:
            batch_size = self.train_bs
            is_shuffle = self.is_shuffle  # 셔플을 여기서 해줘야해. 밖에서는 느려
            file_list = tr_file_list
            # path = self.train_path
        else:
            batch_size = self.test_bs
            is_shuffle = False
            file_list = test_file_list
            # path = self.test_path

        # file_list = [file for file in file_list if file.endswith("docs.txt")]
        # print(file_list)
        # for file in tqdm(shuffle(file_list)):
        for file in file_list:
            cur_file = os.path.join(raw_path, file)
            with open(cur_file, encoding='utf-8') as f:  # 일단 읽어서 길이는 알아둔다.
                data_length = len(f.readlines())
                # print('Num of pat: ', data_length)

            indices = list(range(data_length))  # 인덱스를 미리 만들어주는게 제너레이터 사용의 핵심.
            if is_shuffle:
                shuffle(indices)  # 셔플할꺼라면 이걸... 내장 라이브러리 random에 있는 함수.

            current_count = 0
            while True:
                if current_count >= data_length:
                    return
                else:
                    target_indices = indices[current_count:current_count + batch_size]
                    texts, labels = self.read_lines(target_indices, cur_file)
                    # tokenized_title = self.tokenize_by_morph(title)
                    # tokenized_title = texts.split()
                    tokenized_title = texts
                    labels = self.create_multiplehot_labels(labels)
                    indexed_encoder_inputs = self.text_to_sequence(tokenized_title)
                    padded_encoder_inputs = pad_sequences(indexed_encoder_inputs,
                                                          maxlen=self.max_length,
                                                          padding='pre')
                    yield padded_encoder_inputs, labels

    def mapping_fn(self, x, y=None):
        inputs, label = {'x': x}, y
        return inputs, label

    def train_input_fn(self):
        dataset = tf.data.Dataset.from_generator(generator=lambda: self.data_generator(is_train=True),
                                                 output_types=(tf.int64, tf.int64),
                                                 output_shapes=(
                                                     (None, self.max_length),  # 넣어주면 graph그릴때 잘못 들어온 입력을 잡아줄 수 있다.
                                                     (None, None)))  # labels count: unknown
        # id_kipi, pd, cpc, title, ab, cl
        # for text, label in dataset.take(2):
        #     text = _print_function(text)
        #     print('value: ', text, label)
        dataset = dataset.map(self.mapping_fn)
        dataset = dataset.repeat(count=self.epoch)
        return dataset

    def test_input_fn(self):
        dataset = tf.data.Dataset.from_generator(generator=lambda: self.data_generator(is_train=False),
                                                 output_types=(tf.int64, tf.int64),
                                                 output_shapes=((None, self.max_length),
                                                                (None, None)))
        dataset = dataset.map(self.mapping_fn)
        return dataset

    def eval_input_fn(self):
        dataset = tf.data.Dataset.from_generator(generator=lambda: self.data_generator(is_train=False))
        dataset = dataset.map(self.mapping_fn)
        return dataset


vocab_path = os.path.join(raw_path + '/vocab.voc')
label_file = raw_path + '/labels.pickle'

dataset = Dataset(train_path=tr_file_list,
                  test_path=test_file_list,
                  is_shuffle=True,
                  train_bs=256,
                  test_bs=128,
                  epoch=5,
                  max_length=30,
                  vocab_path=vocab_path,
                  label_path=label_file
                  # is_header_first=True
                  )

# vocab_size = len(dataset.word2idx)
# print('vocab_size: ', vocab_size)


def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    # feature['x'] => (bs, 20)

    """
    print('features: ', features)
    print('labels: ', labels)
    features:  {'x': <tf.Tensor 'IteratorGetNext:0' shape=(?, 30) dtype=int64>}
    labels:  Tensor("IteratorGetNext:1", shape=(?, ?), dtype=int64)
    """
    train_op = features
    loss = features
    predicted_token = features
    embedding_layer = tf.keras.layers.Embedding(params['vocab_size'],
                                                params['embedding_dimension'])(features['x'])  # (bs, 20, EMD_SIZE)

    # K.print_tensor(embedding_layer, message='Embedding Layer')
    # embedding = K.print_tensor(embedding_layer, message='Embedding Layer')
    # print('embedding: ', embedding)
    # embedding:  Tensor("Identity:0", shape=(?, 30, 128), dtype=float32)

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
    logits = tf.keras.layers.Dense(units=params['label_size'])(dropout_hidden)  # 이렇게하면 one-hot 필요

    # if labels is not None:
    #     # labels = tf.reshape(labels, [-1, 1])  # (bs, 1)
    #     print('labels: ', labels)
    #     labels = tf.one_hot(indices=labels, depth=params['label_size'])  # (bs, 2)
    #     print('labels one_hot: ', labels)

    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, logits,
                                                       weights=1.0, label_smoothing=0.01)
                                               # )
        # loss = tf.losses.softmax_cross_entropy(logits=logits, labels=labels)

        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

    elif EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits,
                                               weights=1.0, label_smoothing=0.01)
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

    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions={'prediction': predicted_token})


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

hyper_params = {'vocab_size': len(dataset.word2idx),
                'label_size': len(dataset.label2idx),
                'embedding_dimension': 128,
                # 'gru_dimension': 128,
                # 'attention_dimension': 256,
                # 'start_token_index': dataset.word2idx['<BOS>'],
                'max_length': 30,
                'teacher_forcing_rate': 0.5,
                'use_attention': True}

strategy = tf.contrib.distribute.OneDeviceStrategy(device="/gpu:1")
config = tf.estimator.RunConfig(train_distribute=strategy)

est = tf.estimator.Estimator(model_fn=model_fn,
			params=hyper_params,
			config=config,
			model_dir="abst_to_section/checkpoint")

est.train(dataset.train_input_fn)
valid = est.evaluate(dataset.eval_input_fn)
