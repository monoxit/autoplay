# multi_learner.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
# The face recognition part of the code is
#  based on facerec_from_webcam.py
#  distributed with face_recognition Python library.
#  Copyright (c) 2017, Adam Geitgey
#  SPDX-License-Indentifier: MIT
#
# 使用するライブラリの取り込み
import face_recognition
import cv2
import pickle
import argparse
import os
import numpy as np

# 定数の定義
OUTPUT_FILE = 'knownfaces.pkl'

# 画像のリストを作成する
# 次のようなディレクトリ構造を前提としている
# サブディレクトリ名（「hanako」の部分）がラベル名として使われる
# [dataset]
# |-[hanako]
#   |-image1.jpg
#   |-image2.jpg
# |-[ichiro]
#   |-image1.jpg

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', default='dataset',
                help='path to input files')
args = ap.parse_args()

input_file_path = args.input

# 画像ファイル一覧作成
image_paths = []
for path, dirs, files in os.walk(input_file_path):
    if len(files) > 0:
        for file in files:
            image_paths.append(os.path.join(path, file))

print(image_paths)

# 学習結果を入れておく配列を用意しておく
known_face_encodings = []
known_face_names = []

print('顔の学習')
for image_path in image_paths:

    print('')
    # 画像をimageに読み込む

    print('画像読み込み:', end='')
    print(image_path)

    image = cv2.imread(image_path)

    # 取り込んだ画像の青と赤のチャンネルを入れ替える
    # OpenCVは青緑赤の順で画像データを扱うがface_recognitionライブラリは赤緑青で扱うため
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # パスのフォルダ名をnameに保存し名前ラベルとして使う
    name = image_path.split(os.path.sep)[-2]
    print('名前:', end='')
    print(name)

    print('HOG方式で顔を検出')

    # HOG方式で顔を検出しボックス座標をface_locationsに入れる
    face_locations = face_recognition.face_locations(rgb)
    print(face_locations)

    if len(face_locations) == 0:
        print('ERROR!：顔が見つかりませんでした')
        continue

    # 見つかったすべての顔に枠を描画しながら横幅リストを作成
    width_list = []

    for (top, right, bottom, left) in face_locations:

        image = cv2.rectangle(image, (left, top),
                              (right, bottom),
                              (0, 0, 255), 1)

        width_list.append(right - left)

    max_width_index = np.argmax(width_list)
    max_box = face_locations[max_width_index]
    (top, right, bottom, left) = max_box

    # 一番大きな枠を緑で描画
    image = cv2.rectangle(image, (left, top),
                          (right, bottom), (0, 255, 0), 2)

    # 確認のため画像を画面に表示
    cv2.imshow('Detected', image)
    cv2.waitKey(1000)

    # 128次元特徴量を得る
    print('切り取った顔を深層学習型AIに入力し128次元特徴量を得る')
    encoding = face_recognition.face_encodings(rgb, [max_box])[0]

    # 名前と128次元顔特徴量を配列に追加
    known_face_names.append(name)
    known_face_encodings.append(encoding)

# 既知の顔の学習終了

# 学習結果の表示
print('学習結果')
print('既知の顔の特徴量リスト')
print(known_face_encodings)
print('ラベルリスト')
print(known_face_names)

# 学習結果をファイルに保存
print("学習結果保存:", end='')
print(OUTPUT_FILE)

known_face_data = {"names": known_face_names,
                   "encodings": known_face_encodings}

with open(OUTPUT_FILE, 'wb') as file:
    pickle.dump(known_face_data, file)

print('完了')
