import cv2
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,1)
ret=True
path="filepath/"
num=1
while ret:
    ret,frame=cap.read()
    cv2.imwrite(path+str(num)+".jpg",frame)
    num++
    if cv2.waitKey(1000)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
