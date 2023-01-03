import cv2
import time
import math
import numpy as np
from djitellopy import Tello as _Tello
from pyimagesearch.pid import PID
from cal import cal
from lab1_1 import blue_only


DEALY = 0.5
lf_speed = 10
up_speed = 15
down_speed = 14
def drone_move(drone, action):
    action2speed = {
        'right': (lf_speed, 0, 0, 0),
        'up': (0, 0, up_speed, 0),
        'down': (0, 0, -down_speed, 0),
        'left': (-lf_speed, 0, 0, 0)
    }
    speed = action2speed[action]
    drone.send_rc_control(*speed)


kernel_size = 5#3
canny_threshold = (30, 150)
def img_preprocess(img):
    tmp = img.copy()
    tmp = blue_only(tmp)
    gray = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges_frame = cv2.Canny(blur_gray, *canny_threshold)
    return edges_frame#blur_gray#blur_gray#tmp

large_rect_ratio = 6
#large_rect_ratio = 5
#small_rect_ratio = 4
def detect_line(img):
    rows, cols = img.shape[:2]
    start = (cols // large_rect_ratio, rows // large_rect_ratio)
    end = (start[0] * (large_rect_ratio-1), start[1] * (large_rect_ratio-1))
    result = [False] * 4
    # detect up
    for i in range(start[0], end[0]):
        if img[start[1]][i] > 100:
            result[0] = True
            break
    # detect down
    for i in range(start[0], end[0]):
        if img[end[1]][i] > 100:
            result[1] = True
            break
    # detect left
    for i in range(start[1], end[1]):
        if img[i][start[0]] > 100:
            result[2] = True
            break
    # detect right
    for i in range(start[1], end[1]):
        if img[i][end[0]] > 100:
            result[3] = True
            break
    '''
    # smaller rect
    start = (rows // small_rect_ratio, cols // small_rect_ratio)
    end = (start[0] * (small_rect_ratio-1), start[1] * (small_rect_ratio-1))
    # detect up
    for i in range(start[0], end[0]):
        if img[i][start[1]] > 100:
            result[0] = True
            break
    # detect down
    for i in range(start[0], end[0]):
        if img[i][end[1]] > 100:
            result[1] = True
            break
    # detect left
    for i in range(start[1], end[1]):
        if img[start[0]][i] > 100:
            result[2] = True
            break
    # detect right
    for i in range(start[1], end[1]):
        if img[end[0]][i] > 100:
            result[3] = True
            break
    '''
    return result
    
def draw_windows(out_img):
    cols, rows = out_img.shape[:2]
    start = (rows // large_rect_ratio, cols // large_rect_ratio)
    end = (start[0] * (large_rect_ratio-1), start[1] * (large_rect_ratio-1))
    cv2.rectangle(out_img, (start[0], start[1]), (end[0], end[1]), 255, 3, cv2.LINE_AA)

# ['up', 'down', 'left', 'right']
def draw_detect_res(out_img, res:list):
    cols, rows = out_img.shape[:2]
    coords = [[(0, 0), (rows-1, 0)], [(0, cols-1), (rows-1, cols-1)], [(0, 0), (0, cols-1)], [(0, cols-1), (rows-1, cols-1)]]
    for r, two_p in zip(res, coords):
        if r:
            cv2.line(out_img, two_p[0], two_p[1], 255, 2)

#action_pipeline = ['right', 'up', 'right', 'up', 'left', 'down']
#action_transition = [(0, 2), (1, 3), (0, 2), (1, 2), (1, 3)]
last_shift = 0
def trace_line(drone, action_pipeline, action_transition, is_done, show_blue=True):
    global last_shift
    action_idx = 0
    def time2shift(res:list) -> bool:
        if action_idx >= len(action_transition):
            return False
        false_list = [i for i in range(4) if i not in action_transition[action_idx]]
        print("false list", false_list)
        for i in action_transition[action_idx]:
            if not res[i]:
                return False
        for i in false_list:
            if res[i]:
                return False
        return True
    done = False
    while not done:
        drone.streamon()
        frame = drone.get_frame_read().frame
        cv2.imshow('drone', frame)
        key = cv2.waitKey(100)
        if key != -1:
            drone.keyboard(key)
            continue
        blue_img = img_preprocess(frame)
        out_img = blue_img.copy()
        draw_windows(out_img)
        res = detect_line(blue_img)
        print(f"res: {res}")
        draw_detect_res(out_img, res)
        if show_blue:
            cv2.imshow('blue img', out_img)
        if time2shift(res):
            action_idx += 1
            last_shift = time.time()
            drone.send_rc_control(0, 0, 0, 0)
        if action_idx == len(action_pipeline) - 1:
            done = is_done(frame)
        if not done:
            action = action_pipeline[action_idx] if time.time() - last_shift >= DEALY else action_pipeline[max(0, action_idx-1)]
            print(f"action_idx: {action_idx}, action: {action}")
            if action == 'land':
                drone.land()
                return
            drone_move(drone, action)
            
    
