import cv2

if __name__ == '__main__':
    #obiectul video care acceseaza camera
    capture = cv2.VideoCapture(0)

    #importuri pentru clasificatori(ne ajuta sa identificam diferite caracteristici ce ne vor ajuta pe parcursul calcului )
    face_frames = cv2.CascadeClassifier("face_capture.xml")
    eyes_frames = cv2.CascadeClassifier("eyes_capture.xml")


    while True: #citim pana inchidem camera
       ret, frame = capture.read()

       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       faces = face_frames.detectMultiScale(gray, 1.3, 5)
       for (x,y,w,h) in faces:
           cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
           roi_gray = gray[y:y+h,x:x+w]
           roi_color = frame[y:y+h,x:x+w]
           eyes = eyes_frames.detectMultiScale(roi_gray)
           for (ex,ey,ew,eh) in eyes:
               cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
       cv2.imshow('Eyes detections', frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    capture.release()
    cv2.destroyAllWindows()
