import os
import logging
import logging.handlers
import random


import numpy as np
import cv2
import matplotlib.pyplot as plt


import skvideo
skvideo.setFFmpegPath(r'D:\Backup 3-11-2020 OS ReInstall\Visual Studio Documents\TrafficCounter\manual_imports\ffmpeg\fftools')
cv2.ocl.setUseOpenCL(False)
random.seed(123)
import skvideo.io

import utils
from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter, 
    VehicleCounter
)

#==================
IMAGE_DIR = "./out"
VIDEO_SOURCE="input_Trimmed_30_SEC.mp4"
SHAPE = (720, 1280)
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])
#==================

num_frames = 100

def train_bg_subtractor(inst, capture, num=num_frames):
    print ('Training BG Subtractor...')
    i=0
    for frame in capture:
        inst.apply(frame, None, 0.001)
        i+=1
        if i>=num:
            return capture



        
def main():
    log = logging.getLogger("main")

    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=num_frames, detectShadows=True)


    # processing pipline for programming convenience

    pipeline = PipelineRunner(pipeline=[ ContourDetection(bg_subtractor=bg_subtractor, save_image=True, image_dir=IMAGE_DIR),
        
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),

        Visualizer(save_image=False, image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')

        ], log_level=logging.DEBUG)


    #image source
    capture = skvideo.io.vreader(VIDEO_SOURCE)

    train_bg_subtractor(bg_subtractor, capture, num=num_frames)


    frame_number = -1
    _frame_number = -1

    for frame in capture:
        if not frame.any():
            log.error("Frame capture failed, ending progress")
            break

        _frame_number += 1

        if _frame_number %2 !=0:
            continue

        frame_number += 1 

        #frame =next(capture)

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
 
        pipeline.run()

    image_folder = "out"
    vid_name = "all_out.avi"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    video = cv2.VideoWriter(vid_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

        
#==============================

if __name__ == "__main__":

    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)


    
    main()
