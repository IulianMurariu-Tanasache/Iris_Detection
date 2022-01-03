from statistics import mean

import cv2
import numpy

marja_eroare = 12
frames_outside = 8
marja_dir = 11


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


def mean_of_mean(mean, cerc):  # primul este media care contine cele doua chestii, centru si raza
    n_mean = int((mean.centru[0] + cerc.centru[0]) / 2)
    n_mean2 = int((mean.centru[1] + cerc.centru[1]) / 2)
    n_raza = int((cerc.raza + mean.raza) / 2)
    nou = (n_mean, n_mean2)
    return Cercul(nou, n_raza)


def check_mean(circle, mean_c):
    if not mean_c.centru[0] - (mean_c.centru[0] * marja_eroare) / 100 <= circle.centru[0] <= mean_c.centru[0] + (
            mean_c.centru[0] * marja_eroare) / 100:
        return False
    if not mean_c.centru[1] - (marja_eroare * mean_c.centru[1]) / 100 <= circle.centru[1] <= mean_c.centru[1] + (
            mean_c.centru[1] * marja_eroare) / 100:
        return False
    return True


def check_diff(circle1, circle2):
    error = 0
    error += abs(circle2.centru[0] - circle1.centru[0])
    error += abs(circle2.centru[1] - circle1.centru[1])
    return error / 2


def check_in_eye(cerc, eye):
    if not eye[0] <= cerc.centru[0] <= eye[0] + eye[2]:
        return False
    if not eye[1] <= cerc.centru[1] <= eye[1] + eye[3]:
        return False
    return True


def get_direction(cerc, centru):
    dir_vert = ''
    dir_hor = ''
    if cerc.centru[1] < centru[1] - marja_dir:
        dir_vert = 'N'
    elif cerc.centru[1] > centru[1] + marja_dir:
        dir_vert = 'S'
    if cerc.centru[0] < centru[0] - marja_dir:
        dir_vert = 'W'
    elif cerc.centru[0] > centru[0] + marja_dir:
        dir_vert = 'E'
    return dir_vert + dir_hor


def strIntersection(s1, s2):
    out = ""
    for c in s1:
        if c in s2 and not c in out:
            out += c
    return out


def main():
    directii = {
        'N': (0, -1),
        'NE': (1, -1),
        'E': (1, 0),
        'SE': (1, 1),
        'S': (0, 1),
        'SW': (-1, 1),
        'W': (-1, 0),
        'NW': (-1, -1),
    }

    centru_left = None
    centru_right = None

    iris_left = []
    iris_right = []
    frames_correction = 30
    index_left = 0
    index_right = 0

    mean_circle_left = None
    mean_circle_right = None

    def init():
        nonlocal iris_left, iris_right, frames_correction, index_left, index_right, mean_circle_left, mean_circle_right
        iris_left = []
        iris_right = []
        frames_correction = 10
        index_left = 0
        index_right = 0

        mean_circle_left = None
        mean_circle_right = None

    # obiectul video care acceseaza camera
    capture = cv2.VideoCapture('fata.mp4')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # importuri pentru clasificatori(ne ajuta sa identificam diferite caracteristici ce ne vor ajuta pe parcursul calcului )
    face_frames = cv2.CascadeClassifier("face_capture.xml")
    eyes_frames = cv2.CascadeClassifier("eyes_capture.xml")

    flag_reset = False
    contor = 0

    while True:  # citim pana inchidem camera
        ret, frame = capture.read()
        height, width, channels = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_frames.detectMultiScale(gray, 1.3, 5)

        eyes = []
        roi_color = None
        direction = ''

        if flag_reset:
            flag_reset = False
            init()

        if index_left == frames_correction:
            mean_circle_left = mean_circle(to_list(iris_left))
            centru_left = mean_circle_left.centru
            index_left += 1

        if index_right == frames_correction:
            mean_circle_right = mean_circle(to_list(iris_right))
            centru_right = mean_circle_right.centru
            index_right += 1

        for (x, y, w, h) in faces:
            cerc_left = Cercul((9999, 9999), 0)
            cerc_right = Cercul((9999, 9999), 0)

            reset_left = False  # impotriva ochilor falsi
            reset_right = False
            found_left = False
            found_right = False

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eyes_frames.detectMultiScale(roi_gray, 1.15, 5)

            allGood = True
            for (ex, ey, ew, eh) in eyes:

                if ey + ew <= (y + h) / 2:  # ochii pot fi doar in jumatatea de sus a fetei
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    blur = cv2.GaussianBlur(getArea(roi_gray, (ex, ey, ew, eh)), (3, 3), cv2.BORDER_DEFAULT)
                    otsu_thresh, thresh = cv2.threshold(blur, 0, 255,
                                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    edges = cv2.Canny(blur, otsu_thresh * 0.5, otsu_thresh)
                    dilate = cv2.dilate(edges, numpy.ones((3, 3), numpy.uint8), iterations=1)
                    op = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
                    eroded = cv2.erode(op, numpy.ones((3, 3), numpy.uint8), iterations=1)
                    c = None
                    param2 = 45
                    while c is None:
                        c = cv2.HoughCircles(image=eroded, method=cv2.HOUGH_GRADIENT, dp=2, minDist=1,
                                             param1=otsu_thresh,
                                             param2=param2, minRadius=1, maxRadius=13)
                        param2 -= 1

                    if c is not None:
                        for l in c:
                            # OpenCV returns the circles as a list of lists of circles
                            for circle in l:
                                if type(circle) == numpy.ndarray:
                                    center = (int(x + ex + circle[0]), int(y + ey + circle[1]))
                                    radius = int(circle[2])
                                    cerc = Cercul(center, radius)

                                    if x < cerc.centru[0] < (x + w / 2):
                                        # print('Stanga')
                                        if index_left < frames_correction:
                                            if check_diff(cerc, Cercul((x + ex + ew / 2, y + ey + eh / 2), 8)) < 10:
                                                iris_left.append(cerc)
                                                index_left += 1
                                                cerc_left = cerc

                                        elif index_left > frames_correction:
                                            if not check_mean(cerc, mean_circle_left):
                                                if not check_in_eye(mean_circle_left,
                                                                    (ex, ey, ew, eh)) and not found_left:
                                                    reset_left = True
                                                else:
                                                    reset_left = False
                                            else:
                                                reset_left = False
                                                found_left = True
                                                if check_diff(cerc, mean_circle_left) < check_diff(cerc_left,
                                                                                                   mean_circle_left):
                                                    cerc_left = cerc

                                    elif (x + w / 2) < cerc.centru[0] < x + w:
                                        # print('Dreapta')
                                        if index_right < frames_correction:
                                            if check_diff(cerc, Cercul((x + ex + ew / 2, y + ey + eh / 2), 8)) < 10:
                                                iris_right.append(cerc)
                                                index_right += 1
                                                cerc_right = cerc

                                        elif index_right > frames_correction:
                                            if not check_mean(cerc, mean_circle_right):
                                                if not check_in_eye(mean_circle_right,
                                                                    (ex, ey, ew, eh)) and not found_right:
                                                    reset_right = True
                                                else:
                                                    reset_right = False
                                            else:
                                                found_right = True
                                                reset_right = False
                                                if check_diff(cerc, mean_circle_right) < check_diff(cerc_right,
                                                                                                    mean_circle_right):
                                                    cerc_right = cerc

                        cv2.imshow('eroded', eroded)
            if reset_left or reset_right:
                contor += 1
            else:
                contor = 0
            if contor > 3:
                flag_reset = True
            if not flag_reset:  # nu trebuie resetata calibrarea
                if index_left > frames_correction:
                    # nu s-a gasit un cerc bun
                    if cerc_left.raza == 0 or cerc_left is None:
                        cerc_left = mean_circle_left
                    mean_circle_left = mean_of_mean(mean_circle_left, cerc_left)
                    direction = get_direction(cerc_left, centru_left)
                    # print('left: ' + get_direction(cerc_left, centru_left))

                if index_right > frames_correction:
                    # nu s-a gasit un cerc bun
                    if cerc_right.raza == 0 or cerc_right is None:
                        cerc_right = mean_circle_right
                    mean_circle_right = mean_of_mean(mean_circle_right, cerc_right)
                    direction = strIntersection(direction, get_direction(cerc_right, centru_right))
                    # print('right: ' + get_direction(cerc_right, centru_right))
                    print(direction)

                if cerc_left is not None:  # in perioada de calibrare nu se gaseste niciun cerc
                    cv2.circle(frame, cerc_left.centru, cerc_left.raza, (0, 0, 255), thickness=2)
                if cerc_right is not None:  # in perioada de calibrare nu se gaseste niciun cerc
                    cv2.circle(frame, cerc_right.centru, cerc_right.raza, (0, 0, 255), thickness=2)

        if index_left < frames_correction or index_right < frames_correction:
            cv2.putText(frame, 'Calibration: Look straight', (50, 50), font, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Eyes detections', frame)

        #miscare mouse

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Eyes detections', cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
