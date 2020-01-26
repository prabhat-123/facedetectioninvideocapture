import cv2 as cv, time
import numpy as np 
face_cascade = cv.CascadeClassifier("C:/Users/ASUS/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("C:/Users/ASUS/Anaconda3/Lib/site-packages/cv2/data/haarcascade_smile.xml")
video = cv.VideoCapture(0,cv.CAP_DSHOW)
while True:
    check, frame = video.read()
    print(frame)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv.imshow("Capture",frame)
    key = cv.waitKey(0)
    if key == ord('q'):
        break
video.release()
cv.destroyAllWindows()