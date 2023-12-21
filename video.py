import cv2 as cv
import numpy as np
from numpy.ctypeslib import as_ctypes

import ctypes as c
from multiprocessing import Process,RawArray

import time

path_input  = r"/home/peppo/Documents/Video_processing/test.mp4"
path_output = r"/home/peppo/Documents/Video_processing/output.avi"

def render(cap,out,n_processes=2):

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    oldFrame = np.zeros((frame_height,frame_width))
    
    t = RawArray(c.c_uint8, (frame_height*frame_width*1))
    t_np = np.frombuffer(t,dtype=np.uint8).reshape((frame_height,frame_width))

    counter = 0

    while cap.isOpened():

        if counter >= 20:
            break

        if cv.waitKey(1) == ord('q'):   # exit the program
            break

        ret, frame = cap.read() 

        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        if not ret:             # if frame is read correctly ret is True
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        tmp = np.zeros(frame.shape,dtype=np.uint8)

        np.copyto(t_np,tmp)


        k = int(frame_height/n_processes)
        processes = list()
        for i in range(n_processes):

            p = Process(target=func, args=(frame,oldFrame,t_np, range(i*k,(i+1)*k),range(frame_width)))
            processes.append(p)
            p.start()

        for i in range(n_processes):
            processes[i].join()

        out.write(t_np)

        oldFrame[:] = frame

        # grey = cv.cvtColor(t_np, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', t_np)
        
        counter += 1

    cap.release()
    cv.destroyAllWindows()


def func(frame,oldFrame,tmp,xlim,ylim):

    for x in xlim:
            for y in ylim:

                # if abs(oldFrame[x,y]-frame[x,y]) > 3:

                tmp[x,y] = oldFrame[x,y]-frame[x,y]

                # for i in range(3):
                    
                #     tmp[x,y,i] = abs(oldFrame[x,y,i]-frame[x,y,i])



if __name__ == '__main__':

    cap = cv.VideoCapture(path_input)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    out = cv.VideoWriter(path_output,cv.VideoWriter_fourcc(*'MJPG'), 10, (frame_width,frame_height))

    n = 2

    a = time.time()
    render(cap,out,n)
    b = time.time()

    print(f"Number of processes:\t{n}")
    print("execution Time:\t",b-a)