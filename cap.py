# cap.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
# The face recognition part of the code is
#  based on facerec_from_webcam.py 
#  distributed with face_recognition Python library.
#  Copyright (c) 2017, Adam Geitgey
#  SPDX-License-Indentifier: MIT
#

print('顔写真撮影')

# 使用するライブラリの取り込み
import cv2
import os
import argparse

# カメラバッファ読み飛ばし回数定義
CAMERA_BUF_FLUSH_NUM = 6

# 学習済HAAR CASCADE 顔検出ファイル名の定義
FACE_DETECTION_MODEL = 'haarcascade_frontalface_default.xml'

# 顔検知とみなす確信度の閾値を定義
CONFIDENCE_THRESHOLD = 0.5

# 保存画像ファイル名
IMAGE_FILE_NAME_PREFIX = 'image'

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--out', default='test_data',
                help='path to output files')
args = ap.parse_args()

output_dir_path = args.out              

# ハールカスケード型顔検出モデルを読み込みface_detectorに保存
print('顔検出モデル読み込み')
print(FACE_DETECTION_MODEL)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_DETECTION_MODEL)

# ビデオカメラ開始
print('ビデオカメラ開始...')
cap = cv2.VideoCapture(0)

# OpenCVのチックメータ（ストップウオッチ）機能をtmという名前で使えるようにする
tm = cv2.TickMeter()

# ストップウオッチスタート
tm.start()

file_number = 1

print('sキーで撮影')
print('qキーで終了')
print('注意：カメラ目線で撮影！')

while True:

    #バッファ滞留カメラ画像を読み飛ばし最新画像をframeに読み込む
    for i in range(CAMERA_BUF_FLUSH_NUM):
      ret, frame_org = cap.read()
    
    # 取り込んだ画像の幅を縦横比を維持して500ピクセルに縮小
    ratio = 500 / frame_org.shape[1]
    frame = cv2.resize(frame_org, dsize=None, fx=ratio, fy=ratio)
    
    # 画像をグレースケールにしgrayに保存
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ハールカスケードで顔部分を検出しボックス座標をlocationsへ保存
    locations = face_detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
        
    # ボックス座標の表現形式を変換
    locations = [(y, x + w, y + h, x) for (x, y, w, h) in locations]    
    
    # 検出した顔の数だけ繰り返す
    for top, right, bottom, left in locations:
        # ボックスを顔の周りに描画
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
    # 時間測定
    tm.stop()
    time = tm.getTimeMilli()
    cv2.putText(frame, '{:.2f}(ms)'.format(time),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    # 画像を表示
    cv2.imshow('Video', frame)

    # kにキーボード入力された文字を保存
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        os.makedirs(output_dir_path, exist_ok=True);
        file_name = IMAGE_FILE_NAME_PREFIX + str(file_number) + '.jpg'
        full_file_path = os.path.join(output_dir_path,file_name)
        cv2.imwrite(full_file_path,frame_org)
        print(full_file_path,end='');print('に保存しました')
        file_number += 1
        
        
    # ストップウオッチリスタート
    tm.reset()
    tm.start()
    
# カメラリリース
cap.release()
cv2.destroyAllWindows()
