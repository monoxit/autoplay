# single_learner.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
# The face recognition part of the code is
#  based on facerec_from_webcam.py 
#  distributed with face_recognition Python library.
#  Copyright (c) 2017, Adam Geitgey
#  SPDX-License-Indentifier: MIT
#

# 使用するライブラリの取り込み
print('ライブラリの取り込み')
import face_recognition
import cv2
import pickle
import argparse

print('顔１つを１枚の画像で学習')

#　学習結果保存先ファイル名
OUTPUT_FILE = 'knownface.pkl'

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', default='test_data/image1.jpg',
                help='path to input files')
ap.add_argument('-l', '--label', default='Indigo',
                help='label name')
args = ap.parse_args()

label_name = args.label
input_file = args.input 

# ステップ１　－　画像をimageに読み込む
print('画像読み込み:',end='');print(input_file)
image = cv2.imread(input_file)

# 取り込んだ画像の青と赤のチャンネルを入れ替える
# OpenCVは青緑赤の順で画像データを扱うがface_recognitionライブラリは赤緑青で扱うため
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ステップ２　－　顔の検出と特長量抽出
print('人工知能で128次元顔特徴量を得る')
# 128次元顔特徴量を取得しface_encodingに保存
encodings = face_recognition.face_encodings(rgb)
if len(encodings) == 0:
  print('ERROR!：顔が見つかりませんでした')
  exit(-1)
  
encoding = encodings[0]

# ステップ３　－　学習

# 学習結果をファイルに保存
print("学習結果保存:",end=''); print(OUTPUT_FILE)
known_face_data = {"label": label_name, "encoding": encoding}
with open(OUTPUT_FILE, 'wb') as file:
  pickle.dump(known_face_data, file)

# 学習結果のターミナルへの表示
print('学習結果')
print('ラベル:', label_name)
print('既知の顔の特徴量:', encoding)

print('学習終了')