import cv2 as cv
import numpy as np
import time
import os

path = r'C:\Users\Ryan Sevidal\Desktop\haarcascade_frontalface_default.xml'
face_detector = cv.CascadeClassifier(path)

def detect():
    rects = face_detector.detectMultiScale(gray_s, 
        scaleFactor=1.1,
        minNeighbors=5, 
        minSize=(30, 30), 
        flags=cv.CASCADE_SCALE_IMAGE)

    for rect in rects:
        cv.rectangle(gray_s, rect, 255, 2)

LOOK_AWAY_TIME = 3
look_away = None
cap = cv.VideoCapture(0)

M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
size = (640, 360)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_s = cv.warpAffine(gray, M, size)

    detect()
    
    cv.imshow('window', gray_s)
    t = time.time()
    #cv.displayOverlay('window', f'time={t-t0:.3f}')
    faces = face_detector.detectMultiScale(gray, 1.3, 4) 
    if len(faces) == 0:
        if look_away is None:
            look_away = t
        elif t - look_away > LOOK_AWAY_TIME:
            if os.name == 'nt':
                os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            break
    else:
        look_away = None
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()