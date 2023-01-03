import os
import cv2
import numpy as np
from djitellopy import Tello

def cal(src, frame_func=lambda src: src.streamon().get_frame_read().frame, store_path='cal.xml'):
    if not os.path.exists(store_path):
        objp = np.zeros((9*6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        obj_pts = list()
        img_pts = list()
        while len(img_pts) < 12:
            #drone.streamon()
            frame = frame_func(src)#drone.get_frame_read().frame

            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corner = cv2.findChessboardCorners(gframe, (9, 6))
            cv2.imshow('frame', frame)
            cv2.waitKey(33)
            if not ret:
                continue
            corner2 = cv2.cornerSubPix(gframe, corner, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            obj_pts.append(objp.copy())
            img_pts.append(corner2)
            draw = cv2.drawChessboardCorners(frame, (9, 6), corner2, ret)
            cv2.imshow('frame', draw)
            cv2.waitKey(1000)
            # corner

        ret, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, gframe.shape, None, None)
        f = cv2.FileStorage(store_path, cv2.FILE_STORAGE_WRITE)
        f.write("intrinsic", camMat)
        f.write("distoration", distCoeffs)
        f.release()
    
    fs = cv2.FileStorage(store_path, cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode('distortion').mat()

    return intrinsic, distortion

if __name__ == '__main__':
    drone = Tello()
    drone.connect()
    cal(drone)