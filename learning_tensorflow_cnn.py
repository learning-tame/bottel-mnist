import tensorflow as tf

# WARNING回避
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 現在地を取得
base = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = base + '/tensorflow_model/linear'


#重み変数の初期化
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

#バイアスの初期化
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

#畳み込み演算
def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#2 x 2のMAXプーリング
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def train_model(vecs, labels, filter_size1, filter_size2, filter_size3, lr, iters_num, batch_size):
    # neural_size = 10# 2層目のサイズ
    # batch_size = 1000
    # lr = 0.1 学習率
    # 畳み込み層の幅と高さ
    filter_height = 20
    filter_width = 20

    # 学習データの数
    data_size = len(labels)
    # 学習データの1次元目のサイズ 今回はdoc2vecでvector_sizeにしている300になる
    vector_size = len(vecs[0])
    # 正解ラベルの種類数(分類数)
    classifier_size = len(labels[0])

    print('data_size:',data_size)
    print('vector_size:',vector_size)
    print('classifier_size:',classifier_size)

    x = tf.placeholder(tf.float32, [None, vector_size])


    #[バッチ数、縦、横、チャンネル数] 100
    x_image = tf.reshape(x, [-1,100,100,1])
    #畳み込み層１（フィルター数は32個）
    #フィルターのパラメタをセット
    #[縦、横、チャンネル数、フィルター数]
    W_conv1 = weight_variable([filter_height, filter_width, 1, filter_size1], name='weight1')
    #32個のバイアスをセット
    b_conv1 = bias_variable([filter_size1], name='bias1')
    #畳み込み演算後に、ReLU関数適用
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #2x2のMAXプーリングを実行
    #2x2のMAXプーリングをすると縦横共に半分の大きさになる
    h_pool1 = max_pool_2x2(h_conv1)

    #畳み込み層２（フィルター数は64個）
    #フィルターのパラメタをセット
    #チャンネル数が32なのは、畳み込み層１のフィルター数が32だから。
    #32個フィルターがあると、出力結果が[-1, 100, 100, 32]というshapeになる。
    #入力のチャンネル数と重みのチャンネル数を合わせる。
    W_conv2 = weight_variable([filter_height, filter_width, filter_size1, filter_size2], name='weight2')
    b_conv2 = bias_variable([filter_size2], name='bias2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #全結合層（ノードの数は1024個）
    #2x2MAXプーリングを2回やってるので、この時点で縦横が、100/(2*2)の25になっている。
    #h_pool2のshapeは、[-1, 25, 25, 64]となっているので、25*25*64を入力ノード数とみなす。
    W_fc1 = weight_variable([25 * 25 * filter_size2, filter_size3], name='weight3')
    b_fc1 = bias_variable([filter_size3], name='bias3')
    #全結合層の入力仕様に合わせて、2次元にreshape
    h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*filter_size2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #出力層
    W_fc2 = weight_variable([filter_size3, classifier_size], name='weight4')
    b_fc2 = bias_variable([classifier_size], name='bias4')
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


    # # サイズを確認するために書いたもの
    # x2 = tf.placeholder(tf.float32, [None,1])
    # y2 = tf.placeholder(tf.float32, [None,1])
    # test = tf.add(x2, y2)


    # W1 = tf.Variable(tf.random_normal([vector_size, filter_size1]), name='weight1')
    # b1 = tf.Variable(tf.zeros([filter_size1]), name='bias1')
    # x2 = tf.sigmoid(tf.add(tf.matmul(x, W1), b1))
    #
    # W2 = tf.Variable(tf.random_normal([filter_size1, filter_size2]), name='weight2')
    # b2 = tf.Variable(tf.zeros([filter_size2]), name='bias2')
    # x3 = tf.sigmoid(tf.add(tf.matmul(x2, W2), b2))
    #
    # W3 = tf.Variable(tf.random_normal([filter_size2, classifier_size]), name='weight3')
    # b3 = tf.Variable(tf.zeros([classifier_size]), name='bias3')
    #
    # y = tf.nn.softmax(tf.add(tf.matmul(x3, W3), b3))
    #
    # 正解変数の定義
    # ここで、正解データ(画像xxは数値xx)の格納変数を定義します。
    y_ = tf.placeholder(tf.float32, [None, classifier_size])

    # 交差エントロピー(クロスエントロピー)の定義。交差エントロピーは予測値と実際値の差異です。
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

    # 訓練方法の定義
    # "tf.train.GradientDescentOptimizer"(勾配降下法)を使って訓練をします。
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

    # セッションの生成と変数初期化
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    #
    # # # 学習したら、予測はこちら
    # # predict = tf.nn.softmax(tf.add(tf.matmul(x, W)), b)
    # # sess.run(predict, feed_dict={x: xs})


    # 訓練実行
    # 100個のランダムデータを使って1000回訓練を繰り返す。
    for i in range(batch_size):
        print(i, ' : ', lr)
        batch_xs = vecs
        batch_ys = labels
        if i < 100:
          batch_xs = vecs[:500]
          batch_ys = labels[:500]
        elif i < 200:
          batch_xs = vecs[500:1000]
          batch_ys = labels[500:1000]
        elif i < 300:
          batch_xs = vecs[1000:1500]
          batch_ys = labels[1000:1500]
        elif i < 400:
          batch_xs = vecs[1500:2000]
          batch_ys = labels[1500:2000]
        else:
          batch_xs = vecs[2000:]
          batch_ys = labels[2000:]
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

        if i % 10 == 0:
            # バッチ回数の残10%になったら学習率を都度0.9倍する
            if i > batch_size / 10 * 9:
                lr = lr * 0.9
            # 評価
            # ここで、予測値と正解の答え合わせをしています。correct_predictionではTrue/Falseでデータを保持しています。
            # tf.argmaxはデータの中で一番大きい値を取り出す(正解の数値)のこと。tf.argmaxを使って最も可能性(評価値y)が高い数値を出力します。
            correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))

            # 評価の計算と出力
            # True/Falseのデータをtf.castで1 or 0に変えて正答率を計算しています。
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # sess.run(accuracy, feed_dict={x: vecs, y_:labels})
            print('accuracy:', sess.run(accuracy, feed_dict={x: vecs, y_:labels}))

    # 訓練済みモデルを保存
    saver = tf.train.Saver()
    saver.save(sess, MODEL_PATH)
