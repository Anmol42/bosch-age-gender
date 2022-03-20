#!/usr/bin/python3
import numpy as np
import cv2 as cv
import torch
import matplotlib
import tkinter
import matplotlib.pyplot as plt
import pandas as pd

# print(plt.get_backend())
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)    # this line changes matplotlib backend beware lol
# imgs = ['https://ultralytics.com/images/zidane.jpg']
imgs = []
cap = cv.VideoCapture('reservation.mp4')
# print(cap)
# if cap.isOpened()==False:
    # print("l lag gaye")
count = 0
# matplotlib.rcParams['backend'] = 'GTK3Agg' 
df_lists = []
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frameq
        # cv.imshow('Frame',frame)
        if count%5==0:
            imgs.append(frame)
    # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
    count += 1
    if count%50==0:
        results = model(imgs)
        # results.print()
        df_lists += results.pandas().xyxy  #list of lists here lol
        imgs = []
        # break
# print(result.bbox.x)
# print(imgs[0].shape)
# results = model(imgs)
df_lists += results.pandas().xyxy
print(len(df_lists))
agg_rslt = pd.concat(df_lists,axis=0)   #axis 0 for columns
temp = agg_rslt.loc[agg_rslt['class']==0]
print(agg_rslt.describe())
# agg_rslt = agg_rslt.drop(agg_rslt[agg_rslt['class']!=0].index)
print(temp.describe())
# matplotlib.use('GTK3Agg',force=True)
# plt.imshow(cv.cvtColor(imgs[25], cv.COLOR_BGR2RGB))
# plt.show()
# Results
# print(plt.get_backend())
# gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
# non_gui_backends = matplotlib.rcsetup.non_interactive_bk
# print ("Non Gui backends are:", non_gui_backends)
# print ("Gui backends I will test for", gui_env)
# results.save()  # or .show()
# print(results.xyxy[0])  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)
