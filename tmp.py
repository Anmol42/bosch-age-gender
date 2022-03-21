from os import access
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
# from utils import *
from PIL import Image, ImageDraw, ImageFont
from mtcnn.mtcnn import MTCNN
import time

print(cv.__version__)
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

cap = cv.VideoCapture('cctv.mp4')

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes = [0]
face_mod = MTCNN()
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "dump.caffemodel"
# srgan_checkpoint = "./checkpoint_srgan.pth.tar"
# srgan_generator = torch.load(srgan_checkpoint)['generator']
# srgan_generator.eval()
color = (255, 0, 0)
thickness = 2
cnt = 0
genderNet = cv.dnn.readNet(genderModel, genderProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
padding = 10
while True:

    ret, frame = cap.read()
    if ret and cnt == 0:
        # frame = cv.resize(frame, dsize=(860, 480))
        # frame = cv.resize(frame,None ,fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        # cv.imshow('Frame', frame)
        faces = face_mod.detect_faces(frame)
        # print(faces)
        # print(len(faces))
        for f in faces:
            if f['confidence'] < 0.8:
                continue
            # print(f['box'])
            x, y, w, h = f['box']
            # face = frame[x:x+w, y:y+h]
            face = frame[max(y-padding,0):min(y+h+padding,frame.shape[0]-1), max(0,x-padding):min(x+w+padding,frame.shape[1]-1),:]
            face = cv.resize(face,None ,fx=2, fy=2, interpolation=cv.INTER_CUBIC)# print(face.shape)
            # print(face.shape)
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            blob = cv.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # print(blob.dtype)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            confidence = round(f['confidence'],4)
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
            cv.putText(frame, f'{gender}, {age},{confidence}', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv.LINE_AA)
        cv.imshow('/final', frame)

        

        #results = model(frame)
        #crops = results.crop(save=False)
        #detection = np.squeeze(results.render())
        #cv.imshow('BB', detection)
        #if len(crops):
        #    prs_1 = crops[0]['im']
        #    prs_1 = prs_1[:,:,::-1]
        #    fac_1 = face_mod.detect_faces(frame)
        #    #print(fac_1)
        #    if len(fac_1):
        #        print("lmao ded")
        #        x,y,width,height = fac_1[0]['box']
        #        frame2 = cv.rectangle(frame,(x,y),(x+width,y+height),color,thickness)
        #    cv.imshow('Person_1', frame2)
        #    print(prs_1.shape)
        #img = torch.reshape(torch.tensor(prs_1), prs_1.shape).to(torch.float32)
        #print(img)
    if not ret:
        break
    if cv.waitKey(25) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break
    cnt = (cnt+1) % 2
