import cv2
import numpy as np

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(color,gray):
    faces = face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(color, (x,y), (x+w,y+h), (255,0,0), 2)
        req_gray = gray[x:x+w , y:y+h]
        req_color = color[x:x+w , y:y+h]
        eyes = eye.detectMultiScale(req_gray,1.1,22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(req_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

cap = cv2.VideoCapture(0)
while 1:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect(frame,gray_frame)
    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
