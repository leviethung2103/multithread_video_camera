"""
    Run the webcam with Face Detection 
        Face Detection Model: Masked Face from AIZooTech
        Single Thread Reading Webcam
    Author : HungLV
    Date: 5 June, 2020
"""

import math
import numpy as np
import cv2
import time
import imutils
from videos.utils import *
from cutils.logger import Logger
from imutils.video import FPS
from maskDect.maskDetector import MaskDetector

# HYPER PARAMETERS
LOG = Logger(name='logger',set_level='DEBUG')

camera_params = {
    'camera_id': 0,
    'is_export': True,
    'output_folder': './',
    'width': None,
    'height': None,
    'camera_name': 'camera',
    'keep_original_size': True,
    'fps': 25
}

face_params = {
    'weight': '/mnt/49418012-cfa6-4af1-86d8-c0fb55ae6501/Gaze_Estimation/Gitlab_Code/src/graphs/face_mask_detection.params',
    'det_threshold': 0.5,
    'area_threshold': 0,
    'target_size': (260,260),
    'reverse_order': False,
    'color': (0,255,0)
}

# Camera & Video Writer
capture = cv2.VideoCapture(camera_params['camera_id'])

# open a pointer to the video stream and start the FPS timer
fps = FPS().start()

# Face Detector
face_detector = MaskDetector(face_params['weight'])

frame_cnt = 0 

# while hasFrame:
while fps._numFrames < 100:
    start_time = time.perf_counter()

    _, img = capture.read()

    if camera_params['keep_original_size']: 
        img_height, img_width = img.shape[:2]
    else:
        # Resize -> Faster processing
        img = imutils.resize(img, width= min(400,img.shape[1]))
        img_height, img_width = img.shape[:2]

    if frame_cnt == 0:
        LOG.info(f"Height x Width: {img_height,img_width}")
        if camera_params['is_export']:
            writer = prepare_export_video(camera_params['output_folder'], camera_params['camera_name'], camera_params['fps'], (img_width, img_height))

    # Main Process
    face_boxes, confs, labels = face_detector.detect_face(img,conf_thresh= face_params['det_threshold'],target_shape=face_params['target_size'],
                                                reverse_order=face_params['reverse_order'],area_thresh=face_params['area_threshold'])

    for box in face_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2],box[3]), face_params['color'], 1)

    # Show video frames
    cv2.imshow("DEBUG MODE", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s'): # stop
        cv2.waitKey(0)

    if camera_params['is_export']:
        writer.write(img)

    # update the fps counter
    fps.update()

    frame_cnt +=1

    end_time = time.perf_counter()
    print (f"Current FPS: {1/(end_time-start_time)}")

# stop the timer and display the information
fps.stop()
capture.release()
print ("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
print ("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()

# Export result videos
if camera_params['is_export']:
    writer.release()
