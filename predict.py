import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class Predict():

    def __init__(self):
        # 現在地を取得
        base = os.path.dirname(os.path.abspath(__file__))
        MODEL_META_PATH = base + '/tensorflow_model/linear.meta'
        MODEL_DIR = base + '/tensorflow_model/'

        # TensorFlowのセッション
        self.sess = tf.Session()

        # 訓練済みモデルのmetaファイルを読み込み
        self.saver = tf.train.import_meta_graph(MODEL_META_PATH)

        # モデルの復元
        self.saver.restore(self.sess,tf.train.latest_checkpoint(MODEL_DIR))

        self.vector_size = 10000

        # WとBを復元
        self.graph = tf.get_default_graph()
        self.weight1 = self.graph.get_tensor_by_name("weight1:0")
        self.bias1 = self.graph.get_tensor_by_name("bias1:0")
        self.weight2 = self.graph.get_tensor_by_name("weight2:0")
        self.bias2 = self.graph.get_tensor_by_name("bias2:0")
        self.weight3 = self.graph.get_tensor_by_name("weight3:0")
        self.bias3 = self.graph.get_tensor_by_name("bias3:0")

        # WとBをプリント
        # print(sess.run('weight1:0'))
        # print(sess.run('bias1:0'))

        # 重みパタメータを再現し、推論まで行う
        self.x = tf.placeholder(tf.float32, [None, self.vector_size])
        self.W1 = self.sess.run('weight1:0')
        self.W2 = self.sess.run('weight2:0')
        self.W3 = self.sess.run('weight3:0')

        self.b1 = self.sess.run('bias1:0')
        self.b2 = self.sess.run('bias2:0')
        self.b3 = self.sess.run('bias3:0')

        self.x2 = tf.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        self.x3 = tf.sigmoid(tf.matmul(self.x2, self.W2) + self.b2)
        self.predict = tf.nn.softmax(tf.matmul(self.x3, self.W3) + self.b3)


    def prediction(self, vec):
        # vec = np.zeros(784)

        # (10000)のvecを、(,10000)に変換する
        parse_x = list([list(vec)])
        result = self.sess.run(self.predict, feed_dict={self.x: parse_x})
        print(result)
        print(result.argmax())

        return result

# vec2 = np.zeros(10000)
# prediction(vec2)
