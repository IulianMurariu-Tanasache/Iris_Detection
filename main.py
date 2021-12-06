from statistics import mean

import cv2
import numpy

marja_eroare = 12


def getArea(frame, zone):  # zone: zone x, zone y - > stanga sus si zone w, zone h -> lungime, latime
    return frame[zone[1]:zone[1] + zone[3], zone[0]:zone[0] + zone[2]]


class Cercul:
    def __init__(self, cerc, raza):
        self.centru = cerc
        self.raza = raza


def to_list(c):
    list_x = [cerc.centru[0] for cerc in c]
    list_y = [cerc.centru[1] for cerc in c]
    list_raza = [cerc.raza for cerc in c]
    return list_x, list_y, list_raza


def mean_circle(lists_left):
    mean_c = Cercul((int(mean(lists_left[0])), int(mean(lists_left[1]))), int(mean(lists_left[2])))
    return mean_c


# TODO:
#       -sa fie incadrat in patratul ochilui?

def mean_of_mean(mean, cerc):  #primul este media care contine cele doua chestii, centru si raza
    n_mean = int((mean.centru[0] + cerc.centru[0])/2)
    n_mean2 = int((mean.centru[1] + cerc.centru[1])/2)
    n_raza = int((cerc.raza + mean.raza)/2)
    nou = (n_mean, n_mean2)
    return Cercul(nou, n_raza)


def check_mean(circle, mean_c):
    centru_x = circle.centru[0]
    centru_y = circle.centru[1]
    if not mean_c.centru[0] - (mean_c.centru[0] * marja_eroare) / 100 <= circle.centru[0] <= mean_c.centru[0] + (
            mean_c.centru[0] * marja_eroare) / 100:
        centru_x = mean_c.centru[0]
    if not mean_c.centru[1] - (marja_eroare * mean_c.centru[1]) / 100 <= circle.centru[1] <= mean_c.centru[1] + (
            mean_c.centru[1] * marja_eroare) / 100:
        centru_y = mean_c.centru[1]
    if not mean_c.raza - (mean_c.raza * marja_eroare) / 100 <= circle.raza <= mean_c.raza + (
            mean_c.raza * marja_eroare) / 100:
        circle.raza = mean_c.raza
    circle.centru = (centru_x, centru_y)
    return circle


def main():
    iris_left = []
    iris_right = []
    frames_correction = 10
    index_left = 0
    index_right = 0

    mean_circle_left = None
    mean_circle_right = None

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
#aici punem if ul ala dubios
        if index_left == frames_correction:
            mean_circle_left = mean_circle(to_list(iris_left))
            index_left += 1

        if index_right == frames_correction:
            mean_circle_right = mean_circle(to_list(iris_right))
            index_right += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eyes_frames.detectMultiScale(roi_gray, 1.15, 5)
            for (ex, ey, ew, eh) in eyes:
                if ey + ew <= (y + h) / 2:  # ochii pot fi doar in jumatatea de sus a fetei
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    otsu_thresh, thresh = cv2.threshold(getArea(roi_gray, (ex, ey, ew, eh)), 0, 255,
                                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    edges = cv2.Canny(getArea(roi_gray, (ex, ey, ew, eh)), otsu_thresh * 0.5, otsu_thresh)
                    # cv2.imshow('oachi', edges)
                    edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                    c = cv2.HoughCircles(image=edges, method=cv2.HOUGH_GRADIENT, dp=2, minDist=ew, param1=otsu_thresh,
                                         param2=44, minRadius=6, maxRadius=13)
                    # Then mask the pupil from the image and store it's coordinates.

                    if c is not None:
                        for l in c:
                            # OpenCV returns the circles as a list of lists of circles
                            for circle in l:
                                if type(circle) == numpy.ndarray:
                                    center = (int(x + ex + circle[0]), int(y + ey + circle[1]))
                                    radius = int(circle[2])
                                    print(f'Cerc initial: centru: {center} / raza: {radius} / ', end='')
                                    cerc = Cercul(center, radius)
                                    if x < cerc.centru[0] < (x + w / 2):
                                        print('Stanga')
                                        if index_left < frames_correction:
                                            iris_left.append(cerc)
                                            index_left += 1
                                        else:
                                            cerc = check_mean(cerc, mean_circle_left)
                                            if index_left > frames_correction:
                                                mean_circle_left = mean_of_mean(mean_circle_left, cerc)
                                    elif (x + w / 2) < cerc.centru[0] < x + w:
                                        print('Dreapta')
                                        if index_right < frames_correction:
                                            iris_right.append(cerc)
                                            index_right += 1
                                        else:
                                            cerc = check_mean(cerc, mean_circle_right)
                                            if index_right > frames_correction:
                                                mean_circle_right = mean_of_mean(mean_circle_right, cerc)

                                    print(f'Cerc corectat: centru: {cerc.centru} / raza: {cerc.raza}')
                                    print(index_left, index_right)
                                    cv2.circle(frame, cerc.centru, cerc.raza, (0, 0, 255), thickness=2)
                                #aici contorizam
        cv2.putText(frame, 'Calibration: Look straight', (50, 50), font, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
        # cv2.imwrite(path + 'pillar_text.jpg', im)
        cv2.imshow('Eyes detections', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Eyes detections', cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
