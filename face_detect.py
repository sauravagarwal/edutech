import dlib
import numpy as np
import cv2
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        landmarks=predictor(gray,face)
        for i in range(0,68):
            x=landmarks.part(i).x
            y=landmarks.part(i).y
            cv2.circle(frame,(x,y),2,(255,0,0),3,-1)
    cv2.imshow("show",frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
