import cv2
import math
import numpy as np
from pyimagesearch.pid import PID
from face_detect import *

x_pid, z_pid, y_pid, yaw_pid = [None, None, None, None]
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid = PID(kP=0.9, kI=0.0001, kD=0.1)
# yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

x_pid.initialize()
z_pid.initialize()
y_pid.initialize()
yaw_pid.initialize()

def get_theta(rot):
    z_unit = np.array([0, 0, 1])
    rmat = cv2.Rodrigues(rot)[0]
    zz = rmat.dot(z_unit)[0]
    theta = math.atan2(zz, 1) * 180 / math.pi
    return theta

MAX_SPEED = 35
def clamp(x): return max(min(x, MAX_SPEED), -MAX_SPEED)
def get_follow_update(dis, trans, rot):
    x_update = trans[0, 0]
    x_update = x_pid.update(x_update, sleep=0)
    #x_update = int(clamp(x_update) // 4)
    x_update = int(clamp(x_update) // 2)

    z_update = trans[0, 2] - dis
    z_update = z_pid.update(z_update, sleep=0)
    z_update = int(clamp(z_update) // 2)

    y_update = trans[0, 1] + 20
    y_update = y_pid.update(-y_update, sleep=0)
    y_update = int(clamp(y_update))

    theta = get_theta(rot)
    yaw_update = yaw_pid.update(-theta, sleep=0)
    yaw_update = int(yaw_update)

    return x_update, z_update, y_update, yaw_update

def follow_marker(drone, action, intrinsic, distortion, strict=False):
    current_marker = None
    while True:
        frame = drone.get_frame_read().frame
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        
        if markerIds is not None:
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 18.5, intrinsic, distortion)
            minmk = float("inf")
            # get min marker and its tvec, rvec
            for i in range(rvec.shape[0]):
                if markerIds[i] < minmk:
                    minmk = int(markerIds[i])
                    cur_t = tvec[i]
                    cur_r = rvec[i]
            x, y, z = cur_t[0]
            text = f"y+20: {str(y+20)[:6]}, x: {str(x)[:6]} y: {str(y)[:6]} z: {str(z)[:6]}, dist: {(x**2 + y**2 + z**2) ** 0.5}"
            cv2.putText(frame, text, (16, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        else:
            cv2.imshow("drone", frame)
            key = cv2.waitKey(100)
            if key != -1:
                drone.keyboard(key)
            continue

        print(f"detect marker: {minmk}")
        if current_marker is None:
            current_marker = minmk
        if minmk != current_marker:
            break
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        cv2.imshow("drone", frame)
        key = cv2.waitKey(100)
        if key != -1:
            drone.keyboard(key)
        x_update, z_update, y_update, yaw_update = get_follow_update(action[1], cur_t, cur_r)
        theta = get_theta(cur_r)
        print(f"current theta: {theta}")
        inst = action[0]
        x, y, z = cur_t[0]
        print(f"following marker: x: {x}, y: {y}, z: {z}")
        if (inst == 'keep_follow' or
            (inst == 'follow' and z > action[1] * 1.1 and ((not strict) or (abs(x) > 10 or abs(y) > 10))) or 
            (inst == 'follow_back' and z < action[1] * 0.9 and ((not strict) or (abs(x) > 10 or abs(y) > 10)))):
            print(f"x_update: {x_update}, z_update: {z_update}, y_update: {y_update}, yaw_update: {yaw_update}")
            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
        elif ((inst == 'follow' and z <= action[1] * 1.1 and ((not strict) or (abs(x) <= 10 or abs(y) <= 10))) or 
                (inst == 'follow_back' and z <= action[1] * 0.9 and ((not strict) or (abs(x) <= 10 or abs(y) <= 10)))):
            # when condition is not meet for follow or follow back
            return

#def follow_face(drone, follow_dist intrinsic, distortion, until):
def follow_face(drone, inst, follow_dist, intrinsic, distortion, until=None):
    not_seen = 0
    if until is None:
        until = lambda _, d: d == float("inf") and not_seen >= 10
    
    while True:
        frame = drone.get_frame_read().frame
        closet_dist, remains = detect_nearest_face(frame, intrinsic, distortion, draw_img=True)
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
        if key != -1:
            drone.keyboard(key)
        if until(frame, closet_dist):
            break
        if closet_dist == float("inf"):
            not_seen += 1
            continue
        not_seen = 0

        rvec, tvec = remains[:2]
        rvec = rvec.reshape(1, 3)
        tvec = tvec.reshape(1, 3)
        #print("rvec", rvec," tvec", tvec)
        #print(rvec.shape, tvec.shape)
        if (inst == 'keep_follow_face' or
            (inst == 'follow_face' and closet_dist > follow_dist * 1.1) or 
            (inst == 'follow_face_back' and closet_dist < follow_dist * 0.9)):
            print(f"inst: {inst}, closet dist: {closet_dist} follow_dist: {follow_dist}")
            x_update, z_update, y_update, yaw_update = get_follow_update(follow_dist, tvec, rvec)
            print(f"x_update: {x_update}, z_update: {z_update}, y_update: {y_update}, yaw_update: {yaw_update}")
            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
        elif inst == 'follow_face' or inst == 'follow_face_back':
            # when condition is not met for follow or follow back
            print(f"inst: {inst}, closet dist: {closet_dist} follow_dist: {follow_dist} finish!!!!!!!!!!!!!!!!!!!!!!!")
            return
