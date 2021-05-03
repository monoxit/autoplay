# hog.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
# The face recognition part of the code is
#  based on facerec_from_webcam.py 
#  distributed with face_recognition Python library.
#  Copyright (c) 2017, Adam Geitgey
#  SPDX-License-Indentifier: MIT
#

# ライブラリ取り込み
# face_recognitionライブラリのHOG顔検出機能を使う
print('ライブラリ取り込み開始')

import cv2
import face_recognition

print('ライブラリ取り込み完了')

# カメラバッファ読み飛ばし回数定義
CAMERA_BUF_FLUSH_NUM = 6

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
    
    # 画像の青赤チャンネル入れ替え
    # OpenCVは青緑赤順,face_recognitionは赤緑青順のため
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # HOG特徴で顔検出
    locations = face_recognition.face_locations(rgb,
                                                number_of_times_to_upsample=1,
                                                model='hog')
    # 検出座標をターミナルへ出力
    print(locations)
    
    # 検出顔数繰り返す
    for top, right, bottom, left in locations:

        # 検出された顔の部分を切り取りfaceへ保存
        face = frame[top:bottom, left:right]

        # 顔の縦横サイズをface_height, face_widthに取得
        (face_height, face_width) = face.shape[:2]

        # 小さな顔を無視
        if face_width < 20 or face_height < 20:
            continue

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


