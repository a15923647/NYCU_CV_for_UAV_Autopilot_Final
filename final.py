import cv2
import numpy as np
from my_tello import Tello
from trace_line import *
from toy_detect import *
from face_detect import *
from follow import *
from cal import cal

drone = Tello()
drone.connect()
intrinsic, distortion = cal(drone, lambda drone: drone.get_frame_read().frame, store_path='cal_drone.xml')
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

print(drone)

def control_drone(drone, inst, dis):
    print(f"control drone: {inst}, {dis}")
    drone.move(inst, dis)
rise_speed = 15
def rise_until_see_face(drone):
    prev_closet = float("inf")
    while True:
        drone.streamon()
        frame = drone.get_frame_read().frame

        closet_dist, remains = detect_nearest_face(frame, intrinsic, distortion, draw_img=True)
        
        cv2.imshow('drone', frame)
        key = cv2.waitKey(100)
        if key != -1:
            drone.keyboard(key)
            continue
        if not drone.is_flying:
            continue
        
        # rise until closet_dist is decreasing or not see any face
        if closet_dist != float("inf"):# and closet_dist >= prev_closet:
            return
        print("rising")
        drone.send_rc_control(0, 0, rise_speed, 0)

def yaw(drone, speed):
    drone.send_rc_control(0, 0, 0, speed)

def detect_toy(drone):
    while detect_toy.result == '':
        drone.streamon()
        frame = drone.get_frame_read().frame
        res = detect_obj(network, class_names, frame, draw_img=True)
        cv2.imshow('drone', frame)
        key = cv2.waitKey(100)
        if key != -1:
            drone.keyboard(key)
        if len(res)  > 0:
            detect_toy.result = res[0][0]
    print(f"detect toy: {detect_toy.result}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def marker_in_frame(frame):
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    print("marker ids: ", markerIds)
    return not (markerIds is None)

def dynamic_trace_line(drone, det_toy_fn):
    if det_toy_fn.result == 'melody':
        #                                             fork    fork    fork                  fork
        pipeline = ['left', 'down', 'left', 'up', 'left', 'left', 'left', 'up', 'left', 'down', 'left']
        pipeline_transition = [(1, 3), (0, 2), (0, 3), (1, 2), (0, 2, 3), (0, 2, 3), (0, 2, 3), (1, 2), (1, 3), (0, 2, 3)]
    elif det_toy_fn.result == 'carna':
        #                                             fork                   fork    fork    fork
        pipeline = ['left', 'down', 'left', 'up', 'left', 'up', 'left', 'down', 'left', 'left', 'left']
        pipeline_transition = [(1, 3), (0, 2), (0, 3), (1, 2), (0, 2, 3), (1, 2), (1, 3), (0, 2, 3), (0, 2, 3), (0, 2, 3)]
    is_done = marker_in_frame
    trace_line(drone, pipeline, pipeline_transition, is_done, show_blue=True)

def keep_follow_face_until_marker(drone, dist):
    follow_face(drone, 'keep_follow_face', dist, intrinsic, distortion, until=lambda frame, _: marker_in_frame(frame))
    

def main():
    drone.is_flying = False
    drone.ready = False

    detect_toy.result = ''

    #rise_until_see_face(drone)
    
    action_pipeline = list(range(5))
    action_pipeline[0] = [('forward', 100, False), ('up', 80), ('follow_face', 130), ('right', 60), ('forward', 140)]
    action_pipeline[1] = [('follow_face', 120), ('left', 50), ('forward', 100), (drone.rotate_counter_clockwise, (90,))]
    action_pipeline[2] = [('follow', 300, False), (detect_toy, (drone,))]
    #action_pipeline[2] = [('follow', 30)]
    action_pipeline[3] = [('follow', 120, False), (dynamic_trace_line, (drone, detect_toy)), ('left', 60),  (drone.rotate_counter_clockwise, (180,))]
    
    action_pipeline[4] = [('follow_face', 70), (drone.rotate_counter_clockwise, (135,)), ('follow', 80, False), (drone.land,)]
    
    while not drone.is_flying or not drone.ready:
        print(drone.is_flying, drone.ready)
        drone.streamon()
        frame = drone.get_frame_read().frame
        cv2.imshow('drone', frame)
        key = cv2.waitKey(100)
        if key != -1:
            drone.keyboard(key)


    for action_idx in range(5):
        pipeline = action_pipeline[action_idx]
        print(f"action_idx: {action_idx}")
        for tk in pipeline:
            if isinstance(tk[0], str):
                print(f"task: {tk[0]}")
                if 'face' in tk[0]:
                    follow_face(drone, tk[0], tk[1], intrinsic, distortion)
                elif 'follow' in tk[0]:
                    follow_marker(drone, tk[:2], intrinsic, distortion, strict=tk[2])
                else:
                    frame = drone.get_frame_read().frame
                    cv2.imshow('drone', frame)
                    key = cv2.waitKey(100)
                    if key != -1:
                        drone.keyboard(key)
                    control_drone(drone, tk[0], tk[1])
            else:
                frame = drone.get_frame_read().frame
                cv2.imshow('drone', frame)
                key = cv2.waitKey(100)
                if key != -1:
                    drone.keyboard(key)
                if len(tk) == 2:
                    tk[0](*tk[1])
                elif len(tk) == 1:
                    tk[0]()



if __name__ == '__main__':
    main()