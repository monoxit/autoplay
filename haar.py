# haar.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
# The face recognition part of the code is
#  based on facerec_from_webcam.py 
#  distributed with face_recognition Python library.
#  Copyright (c) 2017, Adam Geitgey
#  SPDX-License-Indentifier: MIT
#

# ライブラリ取り込み
import cv2

# カメラバッファ読み飛ばし回数定義
CAMERA_BUF_FLUSH_NUM = 6

# 学習済HAAR CASCADE 顔検出ファイル名の定義
FACE_DETECTION_MODEL = 'haarcascade_frontalface_default.xml'

# ハールカスケード型顔検出モデルを読み込みface_detectorに保存
print('顔検出モデル読み込み')
print(FACE_DETECTION_MODEL)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_DETECTION_MODEL)

# ビデオカメラ開始
print('ビデオカメラ開始...')
cap = cv2.VideoCapture(0)

# OpenCVのストップウオッチ機能をtmという名前で使えるようにする
tm = cv2.TickMeter()

# ストップウオッチスタート
tm.start()

while True:

    #バッファ滞留カメラ画像を読み飛ばし最新画像をframeに読み込む
    for i in range(CAMERA_BUF_FLUSH_NUM):
      ret, frame = cap.read()
    
    # 画像縦横比を維持し幅を500ピクセルに縮小
    ratio = 500 / frame.shape[1]
    frame = cv2.resize(frame, dsize=None, fx=ratio, fy=ratio)
    
    # 画像をグレースケールにしgrayに保存
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ハールカスケードで顔部分を検出しボックス座標をlocationsへ保存
    locations = face_detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
        
    # ボックス座標の表現形式を変換
    locations = [(y, x + w, y + h, x) for (x, y, w, h) in locations]    
    print(locations)
    
    # 検出顔数繰り返す
    for top, right, bottom, left in locations:

        # 枠を顔の周りに描画
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
    # 時間測定し測定結果を描画
    tm.stop()
    time = tm.getTimeMilli()
    cv2.putText(frame, '{:.2f}(ms)'.format(time),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    # 画像を表示
    cv2.imshow('Video', frame)

    # 「q」の入力でプログラム終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # ストップウオッチリスタート
    tm.reset()
    tm.start()
    
# カメラリリース
cap.release()
cv2.destroyAllWindows()


