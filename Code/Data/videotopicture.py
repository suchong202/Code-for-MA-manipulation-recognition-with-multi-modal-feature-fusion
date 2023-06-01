#coding=utf-8
import cv2
import os

def videotoimg(v_name):
    vc=cv2.VideoCapture(v_name)
    c=1
    dir_name="nzb"

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False
        print("video is not opened")
    while rval:
        rval,frame=vc.read()
        if frame is not None:
            
            cv2.imwrite(dir_name+"/"+str(c)+'.jpg',frame)
            print(c)
            c=c+1
            cv2.waitKey(1)
    vc.release()
    
videotoimg("E:/wangchen/data/video2/nzb/nzb1.avi")