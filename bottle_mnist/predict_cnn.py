import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class Predict():


    def __init__(self):
        #畳み込み演算
        def conv2d(x, W):
              return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        #2 x 2のMAXプーリング
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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
        self.weight4 = self.graph.get_tensor_by_name("weight4:0")
        self.bias4 = self.graph.get_tensor_by_name("bias4:0")

        # 重みパタメータを再現し、推論まで行う
        self.x = tf.placeholder(tf.float32, [None, self.vector_size])
        self.W1 = self.sess.run('weight1:0')
        self.W2 = self.sess.run('weight2:0')
        self.W3 = self.sess.run('weight3:0')
        self.W4 = self.sess.run('weight4:0')

        self.b1 = self.sess.run('bias1:0')
        self.b2 = self.sess.run('bias2:0')
        self.b3 = self.sess.run('bias3:0')
        self.b4 = self.sess.run('bias4:0')

        #[バッチ数、縦、横、チャンネル数]
        x_image = tf.reshape(self.x, [-1,100,100,1])
        #畳み込み層１（フィルター数は32個）
        #フィルターのパラメタをセット
        #[縦、横、チャンネル数、フィルター数]
        W_conv1 = self.W1
        #32個のバイアスをセット
        b_conv1 = self.b1
        #畳み込み演算後に、ReLU関数適用
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        #2x2のMAXプーリングを実行
        #2x2のMAXプーリングをすると縦横共に半分の大きさになる
        h_pool1 = max_pool_2x2(h_conv1)

        #畳み込み層２（フィルター数は64個）
        #フィルターのパラメタをセット
        #チャンネル数が32なのは、畳み込み層１のフィルター数が32だから。
        #32個フィルターがあると、出力結果が[-1, 28, 28, 32]というshapeになる。
        #入力のチャンネル数と重みのチャンネル数を合わせる。
        W_conv2 = self.W2
        b_conv2 = self.b2
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        #全結合層（ノードの数は1024個）
        #2x2MAXプーリングを2回やってるので、この時点で縦横が、100/(2*2)の25になっている。
        #h_pool2のshapeは、[-1, 25, 25, 64]となっているので、25*25*64を入力ノード数とみなす。
        W_fc1 = self.W3
        b_fc1 = self.b3
        #全結合層の入力仕様に合わせて、2次元にreshape
        h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #出力層
        W_fc2 = self.W4
        b_fc2 = self.b4
        self.predict = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


        # WとBをプリント
        # print(sess.run('weight1:0'))
        # print(sess.run('bias1:0'))



        # self.x2 = tf.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        # self.x3 = tf.sigmoid(tf.matmul(self.x2, self.W2) + self.b2)
        # self.predict = tf.nn.softmax(tf.matmul(self.x3, self.W3) + self.b3)


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
