import cv2
import face_recognition
import os
import dlib
print(dlib.DLIB_USE_CUDA)
known_faces=[]
known_names=[]
font=cv2.FONT_ITALIC
for name in os.listdir("known_faces"):
    for filename in os.listdir(f"known_faces/{name}"):
        img=face_recognition.load_image_file(f"known_faces/{name}/{filename}")
        encoding=face_recognition.face_encodings(img)

        known_faces.append(encoding[0])
        known_names.append(name)

print("processing unknown faces")

for filename in os.listdir("unknown_faces"):
    print(filename)
    img=face_recognition.load_image_file(f"unknown_faces/{filename}")
    locations=face_recognition.face_locations(img,model="cnn")
    encodings=face_recognition.face_encodings(img,locations)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    for loc,enc in zip(locations,encodings):
        ret=face_recognition.compare_faces(known_faces,enc,tolerance=0.6)
        if True in ret:
            name=known_names[ret.index(True)]
            cv2.rectangle(img,(loc[3],loc[0]),(loc[1],loc[2]),(255,0,0),2)
            cv2.putText(img,name,(loc[3]+10,loc[2]+10),font,2,(255,0,0),2)

    cv2.imshow(filename,img)
    cv2.waitKey(3000)





