# -*- coding:utf-8 -*-

from bottle import route, get, post, run, template, request, error
import numpy as np

import base64
from io import BytesIO
from PIL import Image
from binascii import a2b_base64
import predict as pr
import predict_cnn as prcnn
import csv
import common

# bottle.debug(True)

# モデルのロード
pred = pr.Predict()
# pred = prcnn.Predict()

#メニュー画面
@route('/')
def menu():
  #メニュー画面表示
  return template('index', flag=False)


#画像を受け取り、学習データに変換して保存
@post('/data')
def data():
    data = request.forms.get("learn_data")
    number = request.forms.get("number")
    # one-hot表現で正解ラベルを格納 添字がそのまま正解の値
    # label = [0,0,0,0,0,0,0,0,0,0]
    # label[int(number)] = 1

    # 画像のbase64をdecodeして数値の配列にする
    b64_str = data.split(',')[1]
    img = Image.open(BytesIO(a2b_base64(b64_str))).convert('L')
    img_arr = np.array(img).reshape(1, -1)

    data = img_arr[0]
    record = np.concatenate([np.array([number]), data], axis=0)

    # 最後にcsvに出力
    #writeモードでCSVファイルオープン
    f = open('./csv/record.csv', 'a')
    writer = csv.writer(f, lineterminator='\n')
    #格納した情報を全て書き込み
    writer.writerow(record)
    f.close()

    return template('index')


#画像を受け取り、学習済みモデルで推論し値を返す
@post('/predict')
def predict():
    data = request.forms.get("predict_data")

    # 画像のbase64をdecodeして数値の配列にする
    b64_str = data.split(',')[1]
    img = Image.open(BytesIO(a2b_base64(b64_str))).convert('L')
    img_arr = np.array(img).reshape(1, -1)

    result = pred.prediction(img_arr.flatten())
    cert_list = result[0]
    zero = common.round_number(cert_list[0])
    one = common.round_number(cert_list[1])
    two = common.round_number(cert_list[2])
    three = common.round_number(cert_list[3])
    four = common.round_number(cert_list[4])
    five = common.round_number(cert_list[5])
    six = common.round_number(cert_list[6])
    seven = common.round_number(cert_list[7])
    eight = common.round_number(cert_list[8])
    nine = common.round_number(cert_list[9])
    answer = result.argmax()

    # result = y.eval(feed_dict={x: [img_arr]})
    # return template('result', result=result)
    return template('result', answer=answer, result=result, zero=zero, one=one, two=two, three=three, four=four, five=five, six=six, seven=seven, eight=eight, nine=nine)

@error(404)
def error404(error):
  return 'Nothing here, sorry'

run(host='localhost', port=8000, debug=True, reloader=True)
