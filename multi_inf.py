# multi_inf.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
# The face recognition part of the code is
#  based on facerec_from_webcam.py 
#  distributed with face_recognition Python library.
#  Copyright (c) 2017, Adam Geitgey
#  SPDX-License-Indentifier: MIT
#

# 使用するライブラリの取り込み
import pickle
import face_recognition
import cv2
import numpy as np
import argparse

# 定数の定義
FACE_DETECTION_MODEL = 'haarcascade_frontalface_default.xml'
KNOWN_FACE_DATA_FILE = 'knownfaces.pkl'
CAMERA_BUF_FLUSH_NUM = 6

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--distance', default=0.4, type=float,
                help='Distance threshold')

args = ap.parse_args()

distance_threshold = args.distance

# ビデオカメラ開始
print('ビデオカメラ開始...')
cap = cv2.VideoCapture(0)


# ハールカスケード型顔検出モデルを読み込みface_detectorに保存
print('顔検出モデル読み込み')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                      'haarcascade_frontalface_default.xml')

# 既知の顔データの読み込み
print('既知の顔データの読み込み')
print(KNOWN_FACE_DATA_FILE)
known_face_data = pickle.loads(open(KNOWN_FACE_DATA_FILE, "rb").read())
known_face_encodings = known_face_data['encodings']
known_face_names = known_face_data['names']

while True:

    # バッファ滞留カメラ画像を読み飛ばし最新画像をframeに読み込む
    for i in range(CAMERA_BUF_FLUSH_NUM):
        ret, frame = cap.read()

    # 640幅500ピクセルに縮小
    ratio = 500 / frame.shape[1]
    frame = cv2.resize(frame, dsize=None, fx=ratio, fy=ratio)

    # 画像をグレースケールにしgrayに保存
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 画像をBGRからRGBへ変更しrgbに保存
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ハールカスケードで（複数）顔部分検出しボックス座標をface_locationsへ保存
    face_locations = face_detector.detectMultiScale(
                                    gray,
                                    scaleFactor=1.1, 
                                    minNeighbors=5,
                                    minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
        
    # ボックス座標の表現形式を変換
    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]
    
    # 見つかったすべての顔をCNNに入力しそれぞれの顔の128次元特徴量をface_encodingsに保存
    # face_encodingsには多次元配列[[0.782, 0.195,...128要素],[0.009, 1.902...128要素]]が入る
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    # 見つかったすべての顔の数だけ繰り返す
    # 繰り返しの都度face_encodingにそれぞれの顔の128次元特徴量が保存されていく
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        name = "Unknown"

        # face_distanceを呼び出し既知の顔との差（128次元ベクトルのユークリッド距離）を求める
        # face_distanceには既知の顔との距離[0.32862521,0.823423...]が入る
        face_distances = face_recognition.face_distance(
                                    known_face_encodings,
                                    face_encoding)
        print(face_distances)
        
        # numpyライブラリのargminを使い、距離が最小の既知の顔のインデックスを得る
        best_match_index = np.argmin(face_distances)
        
        # もし最も特徴量が近い（距離が短い）既知の顔の距離が閾値より小さければ
        if face_distances[best_match_index] < distance_threshold:
            # 名前にその最も特徴量が近い既知の名前を保存する
            name = known_face_names[best_match_index]

        # 見つけた顔の周りに枠を描画する
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 枠の下方に名前を描画する
        cv2.rectangle(frame, (left, bottom - 20),
                      (right, bottom), (255, 255, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 0.5, (0, 0, 0), 1)

    # 画面に画像を表示
    cv2.imshow('Video', frame)

    # 画面に「q」が入力されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 画面を閉じてカメラの制御を開放
cap.release()
cv2.destroyAllWindows()