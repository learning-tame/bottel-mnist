import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import csv
from csvControll import read_csv
import copy
import learning_tensorflow
import learning_tensorflow_cnn


# WARNING回避
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model():

    # 現在地を取得
    base = os.path.dirname(os.path.abspath(__file__))
    csv_record = "./csv/record.csv"
    record_data_list = read_csv(csv_record)

    # 学習用データを格納する
    # learning_data = list()
    labels = list()
    vecs = list()
    for record_data in record_data_list:
        # row = dict()
        # 正解ラベルをone-hot表現に変えてから格納
        data = [0,0,0,0,0,0,0,0,0,0]
        data[int(record_data[0])] = 1
        labels.append(data)
        vecs.append(record_data[1:])

        # parse_y_ = np.array(list([list(data)]))
        # row['LABEL'] = parse_y_
        # (10000)のvecを、(,10000)に変換する
        # parse_x = np.array(list([list(record_data[1:])]))
        # row['DATA'] = parse_x
        # learning_data.append(copy.copy(row))

    #ハイパーパラメータ
    # neural_size1 = 1000 # 1層目のサイズ
    # neural_size2 = 300 # 2層目のサイズ
    neural_size1 = 1000 # 1層目のサイズ
    neural_size2 = 500 # 2層目のサイズ
    # filter_size1 = 16
    # filter_size2 = 32
    # filter_size3 = 256
    lr = 0.3 #学習率
    batch_size = 500 #バッチサイズ(1回のループで学習するサイズ)
    iters_num = 1000 #繰り返し回数

    # vec2 = np.zeros(10000)

    # vecs = [[0,0,1],[0,1,0],[1,0,0],[1,1,0],[0,1,1]]
    # labels = [[0,1],[1,0],[1,0],[1,0],[0,1]]
    # learning_tensorflow_cnn.train_model(vecs, labels, filter_size1, filter_size2, filter_size3, lr, iters_num, batch_size)

    learning_tensorflow.train_model(vecs, labels, neural_size1, neural_size2, lr, iters_num, batch_size)

train_model()
