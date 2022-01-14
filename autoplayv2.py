# autoplay.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
# Face recognition part of the code is based on facerec_from_webcam.py 
#  distributed with face_recognition Python library.
#  Copyright (c) 2017, Adam Geitgey
#  SPDX-License-Indentifier: MIT
#

# 使用するライブラリの取り込み
import pickle
import time
import argparse
import json
import netrc
import face_recognition
import cv2
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth 
from json.decoder import JSONDecodeError

# 定数
# カメラバッファ読み飛ばし回数
CAMERA_BUF_FLUSH_NUM = 6

MIN_API_INTERVAL = 60 * 1
UNLOCK_PERIOD = 60 * 2
FACE_DETECTION_MODEL = 'haarcascade_frontalface_default.xml'
KNOWN_FACE_DATA_FILE = 'knownfaces.pkl'
VIDEO_NUMBER = 0

REDIRECT_URI = 'http://localhost:8080/'

API_ENDPOINT = 'https://api.spotify.com/v1/me'
SCOPE = 'playlist-read-private user-read-playback-state user-modify-playback-state'

# デバイスIDからデバイス名と状態を取得
def get_device_name(sp, device_id):
    devices = sp.devices()
    for device in devices['devices']:
        if device['id'] == device_id:
            return device['name'],device['is_active']
    return None, None

def main():
    # 引数の定義
    ap = argparse.ArgumentParser()

    ap.add_argument('-t', '--threshold', type=float, default=0.4,
                    help='Distance threshold')

    ap.add_argument('-p', '--playlists', default='playlists.json',
                    help='Json playlists file')

    ap.add_argument('-k', '--knownfaces', default='knownfaces.pkl',
                    help='Know face pkl file')
                    
    ap.add_argument('-i', '--device_id', help='Spotify Device ID',
                    required=True)

    args = ap.parse_args()

    distance_threshold = args.threshold
    playlists_file_name = args.playlists
    knowface_file_name = args.knownfaces
    device_id = args.device_id

    # .netrcからユーザー名,クライアントIDとシークレットを読み込む
    secrets = netrc.netrc('/home/pi/.netrc')
    username, client_id, client_secret = secrets.authenticators('my_spotify')

    # 顔とプレイリスト一覧の読み込み
    playlists_file = open(playlists_file_name, mode='r')
    songs = json.load(playlists_file)

    # Spotify OAuth2認証
    spotify_oauth = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=REDIRECT_URI,scope=SCOPE)
    sp = spotipy.Spotify(auth_manager=spotify_oauth)

    # ハールカスケード型顔検出モデルを読み込みface_detectorに保存
    print('Create HAAR CASCADE face detector.')
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 既知の顔データの読み込み
    print('Read features and labels:', end='')
    print(knowface_file_name)
    known_face_data = pickle.loads(open(knowface_file_name, "rb").read())
    known_face_encodings = known_face_data['encodings']
    known_face_names = known_face_data['names']

    # 前回の検知結果記憶領域
    last_name = 'none'
    last_time = time.time() - MIN_API_INTERVAL

    # ビデオカメラ開始
    print('ビデオカメラ開始...')
    cap = cv2.VideoCapture(0)

    print('Start inference...')
    while True:
        
        #バッファに滞留しているカメラ画像を指定回数読み飛ばし、最新画像をframeに読み込む
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
        face_locations = face_detector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
            
        # ボックス座標の表現形式を変換
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]

        #　見つかったすべての顔をCNNに入力しそれぞれの顔の128次元特徴量をface_encodingsに保存
        # face_encodingsには多次元配列[[0.782, 0.195,...128要素],[0.009, 1.902...128要素]]が入る
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        width_list = []
        name_list = []

        # 見つかったすべての顔の数だけ繰り返す
        # 繰り返しの都度face_encodingにそれぞれの顔の128次元特徴量が保存されていく
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # face_distanceを呼び出し既知の顔との差（128次元ベクトルのユークリッド距離）を求める
            # face_distanceには既知の顔との距離[0.32862521,0.823423...]が入る
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # numpyライブラリのargminを使い、距離が最小の既知の顔のインデックスを得る
            best_match_index = np.argmin(face_distances)

            # もし最も特徴量が近い（距離が短い）既知の顔の距離が閾値より小さければ
            if face_distances[best_match_index] < distance_threshold:
                # 名前にその最も特徴量が近い既知の名前を保存する
                name = known_face_names[best_match_index]
            else:
                name = 'Unknown'

            width_list.append(right - left)
            name_list.append(name)

            # 見つけた顔の周りに枠を描画する
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # 枠の下方に名前を描画する
            cv2.rectangle(frame, (left, bottom - 20),
                          (right, bottom), (255, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = name + ' ' + f'{face_distances[best_match_index]:.3f}'
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        font, 0.5, (0, 0, 0), 1)

        # 画面に画像を表示
        cv2.imshow('Video', frame)

        # 画面に「q」が入力されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        max_name = 'none'
        
        # 最も大きく撮影されている顔の種別をmax_nameに得る
        if len(width_list) > 0:
            max_width_index = np.argmax(width_list)
            max_name = name_list[max_width_index]
        
        # 現在時刻をcurrent_timeへセット
        current_time = time.time()

        # 種別が最後に処理した種別と同じでかつ
        # 最後に処理した時刻からアンロック期間が経過していないときは何もしない
        if (max_name == last_name and
           current_time - last_time <= UNLOCK_PERIOD):
            continue

        #　APIを利用の最短間隔より短いときには何もしない (APIを使わない)
        if current_time - last_time <= MIN_API_INTERVAL:
            continue

        #　種別と曲を紐づけたjsonファイルに種別が存在しなかったら何もしない
        if max_name not in songs.keys():
            continue
            
        last_name = max_name
        last_time = time.time()

        try:
            device_name, device_active = get_device_name(sp, device_id)

            print('device name:', device_name)
            print('device id:', device_id)
            print('device active:', device_active)
            current_track = sp.current_user_playing_track()

            if (current_track is not None
               and current_track['is_playing']
               and current_track['context'] is not None
               and current_track['context']['uri'] == songs[max_name]):
                       
               if not device_active:
                    print('Transfer to',device_id)
                    sp.transfer_playback(device_id, force_play=True)
                    time.sleep(1)
            else:                   
                print('Play list:', songs[max_name])
                        
                sp.start_playback(device_id, context_uri=songs[max_name])
                time.sleep(1)

            time.sleep(2)
            current_track = sp.current_user_playing_track()

            print(F'Hi, {max_name}')
            print('This is the music for you.')
            print(current_track['item']['name'])

        except Exception as e:
            print('Exception!:', e)

    # 画面を閉じてカメラの制御を開放
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
