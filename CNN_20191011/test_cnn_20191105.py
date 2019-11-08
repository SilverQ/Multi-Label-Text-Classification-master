import os
import numpy as np
import tensorflow as tf
import pickle
import json
from tqdm import tqdm
from random import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from collections import Counter
import re
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

base_path = '../data/Raw_Claim/'
exam_param_file = 'exam_param.txt'
result_file = 'validation.txt'
# EXAM_NUM = '09'
# train_batch_size = 100
# test_batch_size = 100
# EPOCHS = 6
# MAX_WORD_LENGTH = 100
# EMB_DIM = 256

data_in_path = base_path + 'input_data/'
test_data_path = base_path + 'test_data/'
meta_data_path = base_path + 'meta_data/'
vocab_file = meta_data_path + 'vocab.voc'
label_file = meta_data_path + 'labels_section.pickle'
freq_file = meta_data_path + 'word_freq.pickle'


def get_file_n_folder():
    # if not os.path.exists(data_in_path):
    #     os.makedirs(data_in_path)
    # if not os.path.exists(test_data_path):
    #     os.makedirs(test_data_path)
    if not os.path.exists(data_out_path):
        os.makedirs(data_out_path)
    if not os.path.exists(meta_data_path):
        os.makedirs(meta_data_path)

    tr = os.listdir(data_in_path)
    tr = [file for file in tr if file.endswith(".txt")]
    # tr_file_list = [file for file in tr_file_list if file.startswith("cpc")]

    te = os.listdir(test_data_path)
    te = [file for file in te if file.endswith(".txt")]
    return tr, te


class Dataset:

    def __init__(self, train_path, test_path, is_shuffle, train_bs, test_bs, epoch, max_length):
        self.train_path = train_path
        self.test_path = test_path
        self.is_shuffle = is_shuffle
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.epoch = epoch
        self.max_length = max_length
        self.special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

        if not os.path.exists(vocab_file):
            print('No vocabulary.')
            print('Making vocabulary.')
            self.build_vocab_by_patent(vocab_file)
            print('Complete build vocabulary!')

        if not os.path.exists(label_file):
            print('No labels.')
            print('Making labels.')
            self.build_labels()
            print('Complete build labels!')

        # print('Loading vocabulary...')
        self.idx2word, self.word2idx = pickle.load(open(vocab_file, 'rb'))
        print('Successfully load %d vocabulary!' % (len(self.idx2word)))
        self.idx2label, self.label2idx = pickle.load(open(label_file, 'rb'))
        print('Successfully load %d labels' % (len(self.idx2label)))

    def build_labels(self):
        error_cnt = 0
        label_list = []
        for file in self.train_path:
            with open(data_in_path + file, encoding='utf-8') as f:
                for line in f:
                    try:
                        patent = json.loads(line)
                        labels = patent['cpc'].split('|')
                        labels = [label[0] for label in labels]
                        for label in labels:
                            if label not in label_list:
                                label_list.append(label)
                    except:
                        error_cnt += 1
        label2idx = {label: idx for idx, label in enumerate(label_list)}
        label = (label_list, label2idx)
        pickle.dump(label, open(label_file, 'wb'))

    def build_freq(self, word_list):
        word_counts = Counter(word_list)
        freq = Counter()
        if os.path.exists(freq_file):
            with open(freq_file, 'rb') as freq_dist_f:
                freq = pickle.load(freq_dist_f)
                print('frequency distribution loaded', len(freq))
        for word, cnt in word_counts.items():
            freq[word] += cnt
        print('freq len: ', len(freq))
        with open(freq_file, 'wb') as freq_dist_f:
            pickle.dump(freq, freq_dist_f)
        return freq

    def build_vocab_by_patent(self, vocab_file):
        error_cnt = 0
        label_list = []
        for file in self.train_path:
            word_list = []
            with open(data_in_path + file, encoding='utf-8') as f:
                for line in tqdm(f):
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
            # print('\nIn "%s" word_list: %d, error_cnt: %d\n' % (file, len(word_list), error_cnt))
            idx2word = self.build_freq(word_list)
        idx2word = self.special_tokens + [word for word, _ in idx2word.most_common(99996)]
        print('idx2word: ', len(idx2word), idx2word[:10])
        print('idx2label: ', len(label_list), label_list)
        word2idx = {word: idx for idx, word in enumerate(idx2word)}
        label2idx = {label: idx for idx, label in enumerate(label_list)}
        vocab = (idx2word, word2idx)
        label = (label_list, label2idx)
        pickle.dump(vocab, open(vocab_file, 'wb'))
        pickle.dump(label, open(label_file, 'wb'))

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
        # print('indices: ', indices)
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line_count in indices:
                    try:
                        patent = json.loads(line)
                        # text = re.sub('[-=.#/?:$}(){,]', ' ', patent['title'] + patent['ab'])
                        text = re.sub('[-=.#/?:$}(){,]', ' ', patent['title'])
                        label = patent['cpc'].split('|')
                        texts.append(text.lower().split())
                        labels.append(list(set([cpc[0] for cpc in label])))
                    except:
                      pass
                line_count += 1
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
            path = data_in_path
        else:
            batch_size = self.test_bs
            is_shuffle = False
            file_list = test_file_list
            path = test_data_path
        # print(file_list)
        for file in tqdm(file_list):
            cur_file = path + file
            # print(file)
            with open(cur_file, encoding='utf-8') as f:  # 일단 읽어서 길이는 알아둔다.
                data_length = len(f.readlines())
                # print('Num of pat: ', data_length)

            indices = list(range(data_length))  # 인덱스를 미리 만들어주는게 제너레이터 사용의 핵심.
            if is_shuffle:
                shuffle(indices)  # 셔플할꺼라면 이걸... 내장 라이브러리 random에 있는 함수.
                # print('suffled indices: ', indices)
            current_count = 0
            # while True:
            #     if current_count >= data_length:
            #         return
            #     else:
            while current_count < data_length:
                target_indices = indices[current_count:current_count + batch_size]
                texts, labels = self.read_lines(target_indices, cur_file)
                tokenized_title = texts
                labels = self.create_multiplehot_labels(labels)
                indexed_encoder_inputs = self.text_to_sequence(tokenized_title)
                padded_encoder_inputs = pad_sequences(indexed_encoder_inputs,
                                                      maxlen=self.max_length,
                                                      padding='pre')
                # print(padded_encoder_inputs, labels)
                current_count += batch_size
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
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.data_generator(is_train=False),
            output_types=(tf.int64, tf.int64),
            output_shapes=((None, self.max_length), (None, None)))
        dataset = dataset.map(self.mapping_fn)
        return dataset


def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    # feature['x'] => (bs, 20)

    train_op = features
    loss = features
    predicted_token = features
    embedding_layer = tf.keras.layers.Embedding(params['vocab_size'],
                                                params['EMB_DIM'])(features['x'])  # (bs, 20, EMD_SIZE)

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
    logits = tf.keras.layers.Dense(units=params['label_size'])(dropout_hidden)  # 이렇게하면 one-hot 필요

    if TRAIN:
        global_step = tf.train.get_global_step()
        if params['smoothing'] == 0:
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        else:
            loss = tf.losses.sigmoid_cross_entropy(labels, logits, weights=1.0, label_smoothing=params['smoothing'])
        # loss = tf.losses.softmax_cross_entropy(labels, logits)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        precision = tf.metrics.precision(labels, tf.round(pred))
        recall = tf.metrics.recall(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode,
                                          train_op=train_op,
                                          loss=loss,
                                          eval_metric_ops={'acc': accuracy,
                                                           'prec': precision,
                                                           'recall': recall})

    elif EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        precision = tf.metrics.precision(labels, tf.round(pred))
        recall = tf.metrics.recall(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops={'acc': accuracy,
                                                           'prec': precision,
                                                           'recall': recall})

    elif PREDICT:
        pred = tf.nn.sigmoid(logits)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                # 'prob': tf.nn.sigmoid(logits),
                # tf.to_int32(a > 0.5)
                'prob': tf.round(pred)})

    plot_model(model_fn(), to_file=data_out_path + 'model.png')

    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions={'prediction': predicted_token})


tf.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# valid = {}

exams = open(exam_param_file, encoding='utf-8')

# for line in exams:
#     valid = {}
#     ld_params = json.loads(line)
#     print('Exam No: %d\n' % ld_params['EXAM_NUM'])
#     #, params["train_batch_size"], params["test_batch_size"], params["EPOCHS"], params["MAX_WORD_LENGTH"])
#     data_out_path = base_path + 'result_' + str(ld_params['EXAM_NUM']) + '/'
#     tr_file_list, test_file_list = get_file_n_folder()
#     print('\nFile List for Training: ', tr_file_list, '\n', 'File List for Testing: ', test_file_list)
#
#     dataset = Dataset(train_path=tr_file_list,
#                       test_path=test_file_list,
#                       is_shuffle=True,
#                       train_bs=ld_params["train_batch_size"],
#                       test_bs=ld_params["test_batch_size"],
#                       epoch=ld_params["EPOCHS"],
#                       max_length=ld_params["MAX_WORD_LENGTH"])
#
#     hyper_params = {'vocab_size': len(dataset.word2idx),
#                     'label_size': len(dataset.label2idx),
#                     'embedding_dimension': EMB_DIM,
#                     'smoothing': ld_params["smoothing"]}
#
#     est = tf.estimator.Estimator(model_fn=model_fn,
#                                  params=hyper_params,
#                                  model_dir=data_out_path)
#
#     est.train(dataset.train_input_fn)
#     valid[ld_params['EXAM_NUM']] = est.evaluate(dataset.eval_input_fn, steps=10)
#     print(valid)
#     with open(data_out_path + str(ld_params['EXAM_NUM']) + '_' + result_file, 'w+') as res:
#         res.write(json.dumps(str(valid)))


valid = {}
hyper_params = {"EXAM_NUM": 59, "train_batch_size": 500, "test_batch_size": 100, "EPOCHS": 10,
                "MAX_WORD_LENGTH": 15, "EMB_DIM": 256, "smoothing": 0.1}

"""
{"EXAM_NUM":59,"train_batch_size":500,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":15,"EMB_DIM":256,"smoothing":0.1}
{"EXAM_NUM":60,"train_batch_size":500,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":15,"EMB_DIM":256,"smoothing":0.2}
{"EXAM_NUM":61,"train_batch_size":500,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":15,"EMB_DIM":256,"smoothing":0.3}
{"EXAM_NUM":62,"train_batch_size":500,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":15,"EMB_DIM":256,"smoothing":0.4}
{"EXAM_NUM":57,"train_batch_size":500,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":10,"EMB_DIM":256,"smoothing":0}
{"EXAM_NUM":58,"train_batch_size":500,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":15,"EMB_DIM":256,"smoothing":0}
{"EXAM_NUM":55,"train_batch_size":100,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":15,"EMB_DIM":256,"smoothing":0.3}
{"EXAM_NUM":56,"train_batch_size":100,"test_batch_size":100,"EPOCHS":10,"MAX_WORD_LENGTH":15,"EMB_DIM":256,"smoothing":0.4}
"""

print('Exam No: %d\n' % hyper_params['EXAM_NUM'])
# , params["train_batch_size"], params["test_batch_size"], params["EPOCHS"], params["MAX_WORD_LENGTH"])
data_out_path = base_path + 'result_' + str(hyper_params['EXAM_NUM']) + '/'

tr_file_list, test_file_list = get_file_n_folder()
print('\nFile List for Training: ', tr_file_list, '\n', 'File List for Testing: ', test_file_list)

dataset = Dataset(train_path=tr_file_list,
                  test_path=test_file_list,
                  is_shuffle=True,
                  train_bs=hyper_params["train_batch_size"],
                  test_bs=hyper_params["test_batch_size"],
                  epoch=hyper_params["EPOCHS"],
                  max_length=hyper_params["MAX_WORD_LENGTH"])

hyper_params['vocab_size'] = len(dataset.word2idx)
hyper_params['label_size'] = len(dataset.label2idx)

est = tf.estimator.Estimator(model_fn=model_fn,
                             params=hyper_params,
                             model_dir=data_out_path)

pred = est.predict(input_fn=dataset.eval_input_fn)
# print('pred: ', pred)
# for item in pred:
#     print(item)
pred = [str(logits) for logits in pred]
print(pred[:10])

# for texts, label_origin in dataset.eval_input_fn().take(1):
#     # print(texts['x'], label_origin)
#     pred = est.predict(texts['x'][0])
#     print(pred)
#     for item in pred:
#         print(item)
#     # pred = [str(logits) for logits in pred]
#     # print('pred: ', pred)
# # valid[ld_params['EXAM_NUM']] = est.predict(dataset.eval_input_fn, steps=10)
# # print(valid)
