import cv2
import numpy as np

import ctypes as c
from multiprocessing import Process,RawArray

path_input  = "test1.mp4"
path_output = "output.mp4"

def render(cap,out,n_processes=2,skip_frames=2):

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    oldFrame = np.zeros((frame_height,frame_width))
    
    t = RawArray(c.c_uint8, (frame_height*frame_width*3))
    t_np = np.frombuffer(t,dtype=np.uint8).reshape((frame_height,frame_width,3))

    counter = 0

    while 1:
        

        if cv2.waitKey(1) == ord('q'):   # exit the program
            break
        

        ret, frame = cap.read() 

        if counter < skip_frames:
            counter +=1
            continue
        
        if counter >= skip_frames:
            counter =0
        
        np.copyto(t_np,frame)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        

        if not ret:             # if frame is read correctly ret is True
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        tmp = np.zeros((frame.shape[0],frame.shape[1],3),dtype=np.uint8)



        k = int(frame_height/n_processes)
        processes = list()

        for i in range(n_processes):

            p = Process(target=func, args=(frame,oldFrame,t_np, range(i*k,(i+1)*k),range(frame_width)))
            processes.append(p)
            p.start()

        for i in range(n_processes):
            processes[i].join()

        # print(t_np)
        out.write(t_np)

        oldFrame[:] = frame

        # grey = cv2.cvtColor(t_np, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', t_np)
        
        counter += 1

    cap.release()
    cv2.destroyAllWindows()


def func(frame,oldFrame,tmp,xlim,ylim):

    for x in xlim:
            for y in ylim:

                if abs(oldFrame[x,y]-frame[x,y]) > 10:

                    tmp[x,y,0] = 0
                    tmp[x,y,1] = 0
                    tmp[x,y,2] = oldFrame[x,y]-frame[x,y]

                # for i in range(3):
                    
                #     tmp[x,y,i] = abs(oldFrame[x,y,i]-frame[x,y,i])



if __name__ == '__main__':

    cap = cv2.VideoCapture(path_input)
    # cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    out = cv2.VideoWriter(path_output,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), cap.get(5), (frame_width,frame_height))

    n_proces = 2
    skip_frames = 0

    render(cap,out,n_proces,skip_frames)