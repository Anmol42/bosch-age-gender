import numpy as np
import cv2

cap = cv2.VideoCapture('reservation.mp4')

if(cap.isOpened() == False):
    print("Error")

while(cap.isOpened()):
    ret,frame = cap.read()
    if(ret== True):
        cv2.imshow('Frame',frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

cv2.destroyAllWindows()

