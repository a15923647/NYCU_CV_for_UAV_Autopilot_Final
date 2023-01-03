import cv2
import dlib
import numpy as np
from cal import cal

scale = 1.05
face_x = 14.5
face_y = 14.5

detector = dlib.get_frontal_face_detector()

def detect_face(img, intrinsic, distortion, draw_img=False):
    face_rects = detector(img, 0)
    # list of [dist, (x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    ret = list()
    for d in face_rects:
        x1, y1 = d.left(), d.top()
        x2, y2 = d.right(), d.bottom()
        
        imgPoints = np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        objp = np.float32([(0, 0, 0), (face_x, 0, 0), (face_x, face_y, 0), (0, face_y, 0)])
        retval, rvec, tvec = cv2.solvePnP(objp, imgPoints, intrinsic, distortion)

        ret.append([float(tvec[2]), rvec, tvec, (x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        if draw_img:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, str(round(float(tvec[2]), 2)), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
    return ret

def detect_nearest_face(img, intrinsic, distortion, draw_img=False):
    faces = detect_face(img, intrinsic, distortion, draw_img)
    closet_dist = float("inf")
    ret = None
    for li in faces:
        dist = li[0]
        if dist < closet_dist:
            ret = li[1:]
            closet_dist = dist
    return closet_dist, ret
