# autoplay

Raspberry Pi face recognition sample programs.

## Description

This repository contains series of programs helping to understand edge AI by utilizing face recognition technology.

## Getting Started

### System requirement

* Raspberry Pi 3B or Raspberry Pi 3B+ or Raspberry Pi 4B
* Powerful AC Adapter for Rasbperry Pi
* Raspberry Pi OS Buster (legacy)
* USB Camera or Raspberry Pi Camera

### Installing

```
sudo apt install -y libhdf5-103 libatlas-base-dev
sudo apt install -y libjasper-dev libqt4-test
sudo apt install -y libqtgui4 libqt4-dev
pip3  --default-timeout=5000 install --prefer-binary opencv-python
python3 -c "import cv2; print(cv2.__version__)"
sudo apt install -y libjpeg-dev
pip3 --default-timeout=5000 install dlib
pip3 --default-timeout=5000 install face_recognition
git clone https://github.com/monoxit/autoplay.git
```

### Executing program

```
cd autoplay
python3 hog.py
```

### Sample programs

* cap.py
  * Take a photo and save it to a file.
  
* haar.py
  * Detect faces by HAAR CASCADE.
  
* hog.py
  * Detect face by HOG+SVM.

* single_learner.py
  * Lagy learn features of a face.

* single_inf.py
  * Recognize faces by lagy learned features of a face.

* multi_learner.py
  * Lagy learn multiple faces.

* multi_inf.py
  * Recognize faces by multiple lagy learned (K-NN k=1) face features.
  
* lsplaylist.py
  * List Spotify playlist.  
  
* lsdevice.py
  * List Spotify devices.
  
* autoplay.py
  * Play Spotify playlist mapped to a detected face.
 
* autoplayv2.py
  * Play Spotify playlist mapped to a detected face. Transfer playlist between multiple devices.  
  
## License

This project is licensed under the MIT License
