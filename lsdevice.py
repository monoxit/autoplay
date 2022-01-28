# Copyright (c) 2021 Masami Yamakawa (MONOxIT Inc.)
#  SPDX-License-Identifier: MIT
#

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import netrc

REDIRECT_URI = 'http://localhost:8080/'
        
API_ENDPOINT = 'https://api.spotify.com/v1/me'
SCOPE = 'playlist-read-private user-read-playback-state user-modify-playback-state'

# .netrcからトークンを読み込む
secrets = netrc.netrc('/home/pi/.netrc')
username, client_id, client_secret = secrets.authenticators('my_spotify')

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=REDIRECT_URI,scope=SCOPE))

devices = sp.devices()
device_list = devices['devices']

for device in device_list:
    #print(device)
    print('device name:',device['name'])
    print('device id:',device['id'])
    print('')
