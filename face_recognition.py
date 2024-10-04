import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')


cap = cv2.VideoCapture(0)

while True:
    #captures video frame by frame
    flag, frame = cap.read()
    
    #makes the captured image monochromatic
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
            cv2.putText(roi_color, 'Smile', (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    #displays the result on the camera feed
    cv2.imshow('Video Feed', frame)
    
    #break when esc key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#release capture once processing is done
cap.release()
cv2.destroyAllWindows()