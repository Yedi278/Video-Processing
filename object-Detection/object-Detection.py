import cv2
import numpy as np

import ctypes as c
from multiprocessing import Process,RawArray

path_input  = "test.mp4"
path_output = "output.avi"

def detection(cap):

    pass

if __name__ == '__main__':

    a = 1
    if a == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path_input)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    out = cv2.VideoWriter(path_output,cv2.VideoWriter_fourcc(*'XVID'), cap.get(5), (frame_width,frame_height))