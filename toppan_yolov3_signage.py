"""
    Run the webcam with Face Detection 
        Face Detection Model: Masked Face from AIZooTech
        Included multithreading reading video method for faster speed
    Author : HungLV
    Date: 9 June, 2020
"""

import math
import time
import numpy as np
import cv2
import imutils
from videos.utils import *
from videos.utils import FPS
from videos.utils import WebCamVideoStream
from cutils.logger import Logger
from utils.utils import read_cfg
from maskDect.maskDetector import MaskDetector

# HYPER PARAMETERS
LOG = Logger(name='logger',set_level='DEBUG')

# Load file config
cfg = read_cfg('config/video.yaml')

# Camera & Video Writer
capture = WebCamVideoStream(src=cfg['input_video']).start()

# open a pointer to the video stream and start the FPS timer
fps = FPS().start()

frame_cnt = 0 

# Face Detector
face_detector = MaskDetector(cfg['face']['weight'])

while capture.stream.isOpened():
    start_time = time.perf_counter()
    # grab the frame from the stream
    img = capture.read()

    if img is None: break

    if cfg['camera']['keep_original_size']: 
        img_height, img_width = img.shape[:2]
    else:
        # Resize -> Faster processing
        img = imutils.resize(img, width= min(400,img.shape[1]))
        img_height, img_width = img.shape[:2]

    if frame_cnt == 0:
        LOG.info(f"Height x Width: {img_height,img_width}")
        if cfg['camera']['is_export']:
            writer = prepare_export_video(cfg['output_folder'], cfg['camera']['camera_name'], cfg['camera']['fps'], (img_width, img_height))

    # Main Process
    face_boxes, confs, labels = face_detector.detect_face(img,conf_thresh= cfg['face']['det_threshold'],target_shape=tuple(cfg['face']['target_size']),
                                                reverse_order=cfg['face']['reverse_order'],area_thresh=cfg['face']['area_threshold'])

    for box in face_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2],box[3]), tuple(cfg['face']['color']), 1)

    # Show video frames
    cv2.imshow("DEBUG MODE", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s'): # stop
        cv2.waitKey(0)

    if cfg['camera']['is_export']:
        writer.write(img)

    # update the fps counter
    fps.update()

    frame_cnt +=1

    end_time = time.perf_counter()
    print (f"Current FPS: {1/(end_time-start_time)}")

# stop the timer and display the information
fps.stop()
capture.stop()
# capture.release()
cv2.destroyAllWindows()

print ("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
print ("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Export result videos
if cfg['camera']['is_export']:
    writer.release()
