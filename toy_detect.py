import cv2
import time
import math
import darknet
import numpy as np

CONFIG_PATH = './yolov3-tiny-lab9.cfg'
DATA_PATH = './obj.data'
WEIGHTS = './yolov3-tiny-lab9_final_two.weights'
network, class_names, class_colors = darknet.load_network(CONFIG_PATH, DATA_PATH, WEIGHTS, batch_size=1)
darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def detect_obj(network, class_names, img, thresh=0.25, draw_img=False):
    prev_time = time.time()
    ################## prepare darknet image #############
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)

    darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    ####################### detect #######################
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    ### adjust detection result back to org img size ###
    detections_adjusted = []
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(img, bbox)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
    if draw_img:
        img = darknet.draw_boxes(detections_adjusted, img, class_colors)
    darknet.free_image(darknet_image)
    print(f"FPS: {int(1/(time.time() - prev_time))}")
    return detections_adjusted # list of (label, confidence, bbox)

    