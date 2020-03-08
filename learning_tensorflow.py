import tensorflow as tf

# WARNING回避
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 現在地を取得
base = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = base + '/tensorflow_model/linear'

def train_model(vecs, labels, neural_size1, neural_size2, lr, iters_num, batch_size):
    # neural_size = 10# 2層目のサイズ
    # batch_size = 1000
    # lr = 0.1 学習率

    # 学習データの数
    data_size = len(labels)
    # 学習データの1次元目のサイズ 今回はdoc2vecでvector_sizeにしている300になる
    vector_size = len(vecs[0])
    # 正解ラベルの種類数(分類数)
    classifier_size = len(labels[0])

    print('data_size:',data_size)
    print('vector_size:',vector_size)
    print('classifier_size:',classifier_size)


    # # サイズを確認するために書いたもの
    # x2 = tf.placeholder(tf.float32, [None,1])
    # y2 = tf.placeholder(tf.float32, [None,1])
    # test = tf.add(x2, y2)

    # 1層の場合はこちら
    '''
    x = tf.placeholder(tf.float32, [None, vector_size])
    W = tf.Variable(tf.zeros([vector_size, classifier_size]))
    b = tf.Variable(tf.zeros([classifier_size]))
    y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))
    '''
    # 層を深くする

    x = tf.placeholder(tf.float32, [None, vector_size])
    W1 = tf.Variable(tf.random_normal([vector_size, neural_size1]), name='weight1')
    b1 = tf.Variable(tf.zeros([neural_size1]), name='bias1')
    x2 = tf.sigmoid(tf.add(tf.matmul(x, W1), b1))

    W2 = tf.Variable(tf.random_normal([neural_size1, neural_size2]), name='weight2')
    b2 = tf.Variable(tf.zeros([neural_size2]), name='bias2')
    x3 = tf.sigmoid(tf.add(tf.matmul(x2, W2), b2))

    W3 = tf.Variable(tf.random_normal([neural_size2, classifier_size]), name='weight3')
    b3 = tf.Variable(tf.zeros([classifier_size]), name='bias3')

    y = tf.nn.softmax(tf.add(tf.matmul(x3, W3), b3))

    # 正解変数の定義
    # ここで、正解データ(画像xxは数値xx)の格納変数を定義します。
    y_ = tf.placeholder(tf.float32, [None, classifier_size])

    # 交差エントロピー(クロスエントロピー)の定義。交差エントロピーは予測値と実際値の差異です。
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # 訓練方法の定義
    # "tf.train.GradientDescentOptimizer"(勾配降下法)を使って訓練をします。
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

    # セッションの生成と変数初期化
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # # 学習したら、予測はこちら
    # predict = tf.nn.softmax(tf.add(tf.matmul(x, W)), b)
    # sess.run(predict, feed_dict={x: xs})


    # 訓練実行
    # 100個のランダムデータを使って1000回訓練を繰り返す。

    import random

    for i in range(iters_num):
        l = [i for i in range(data_size)]
        random_list = random.sample(l, batch_size)
        batch_xs = []
        batch_ys = []
        for j in random_list:
            batch_xs.append(vecs[j])
            batch_ys.append(labels[j])

        print(i, ' : ', lr)
        # batch_xs = vecs
        # batch_ys = labels
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

        if i % 10 == 0:
            # 学習回数の残10%になったら学習率を都度0.9倍する
            if i > iters_num / 10 * 9:
                lr = lr * 0.9
            # 評価
            # ここで、予測値と正解の答え合わせをしています。correct_predictionではTrue/Falseでデータを保持しています。
            # tf.argmaxはデータの中で一番大きい値を取り出す(正解の数値)のこと。tf.argmaxを使って最も可能性(評価値y)が高い数値を出力します。
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

            # 評価の計算と出力
            # True/Falseのデータをtf.castで1 or 0に変えて正答率を計算しています。
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # sess.run(accuracy, feed_dict={x: vecs, y_:labels})
            print('accuracy:', sess.run(accuracy, feed_dict={x: vecs, y_:labels}))

    # 訓練済みモデルを保存
    saver = tf.train.Saver()
    saver.save(sess, MODEL_PATH)
