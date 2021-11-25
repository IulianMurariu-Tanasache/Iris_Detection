

import cv2
import numpy
import numpy as np


def getArea(frame, zone):  # zone: zone x, zone y - > stanga sus si zone w, zone h -> lungime, latime
    return frame[zone[1]:zone[1] + zone[3], zone[0]:zone[0] + zone[2]]

class Cercul:
    def __init__(self):
        self.centrus = 0
        self.centrud = 0
        self.raza = 0
        print('centrus....', self.centrus)
        print('centrus....', self.centrud)
        print('raza', self.raza)

if __name__ == '__main__':
    # obiectul video care acceseaza camera
    capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # importuri pentru clasificatori(ne ajuta sa identificam diferite caracteristici ce ne vor ajuta pe parcursul calcului )
    face_frames = cv2.CascadeClassifier("face_capture.xml")
    eyes_frames = cv2.CascadeClassifier("eyes_capture.xml")

    while True:  # citim pana inchidem camera
        ret, frame = capture.read()
        height, width, channels = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_frames.detectMultiScale(gray, 1.3, 5)

        eyes = []
        roi_color = None

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eyes_frames.detectMultiScale(roi_gray, 1.15, 5)
            for (ex, ey, ew, eh) in eyes:
                if ey + ew <= (y + w) / 2:  # ochii pot fi doar in jumatatea de sus a fetei
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    otsu_thresh, thresh = cv2.threshold(getArea(roi_gray, (ex, ey, ew, eh)), 0, 255,
                                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    edges = cv2.Canny(getArea(roi_gray, (ex, ey, ew, eh)), otsu_thresh * 0.5, otsu_thresh)
                    # cv2.imshow('oachi', edges)
                    edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                    c = cv2.HoughCircles(image=edges, method=cv2.HOUGH_GRADIENT, dp=2, minDist=ew, param1=otsu_thresh,
                                         param2=44.5, minRadius=6, maxRadius=13)
                    # Then mask the pupil from the image and store it's coordinates.
                    memorie = []
                    for i in range(30):
                       memorie.append(Cercul())
                    if c is not None:

                        for l in c:
                            # OpenCV returns the circles as a list of lists of circles
                            for circle in l:
                                if type(circle) == numpy.ndarray:
                                    center = (int(x + ex + circle[0]), int(y + ey + circle[1]))
                                    memorie[l].centrus = center[0]
                                    memorie[l].centrud = center[1]
                                    radius = int(circle[2])
                                    memorie[l].raza = radius
                                    # print(c, l, circle)
                                    cv2.circle(frame, center, int(radius), (0, 0, 255), thickness=2)
                                    pupil = (center[0], center[1], radius)


        cv2.putText(frame, 'Calibration: Look straight', (50, 50), font, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
        #cv2.imwrite(path + 'pillar_text.jpg', im)
        cv2.imshow('Eyes detections', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Eyes detections', cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyAllWindows()
