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
import pandas as pd
from argparse import ArgumentParser
import tensorflow as tf

parser = ArgumentParser()
parser.add_argument('--file',type=str,default='indiancctv.mp4')
parser.add_argument('--fr_rate',type=int,default=5)

args = parser.parse_args()
#print(cv.__version__)
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

cap = cv.VideoCapture(args.file)

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
columns = ['fr_num','bb_xmin','bb_ymin','bb_height','bb_width','age_min','age_max','gender']
rslt = pd.DataFrame(columns=columns)


b2g = tf.keras.models.load_model('./body2gen/model.h5')
while True:

    ret, frame = cap.read()
    if ret and cnt%args.fr_rate == 0:
        # frame = cv.resize(frame, dsize=(860, 480))
        # frame = cv.resize(frame,None ,fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        # cv.imshow('Frame', frame)
        # faces = face_mod.detect_faces(frame)
        # print(faces)
        # print(len(faces))
        results = model(frame)
        crops = results.crop(save=False)
        for f in crops:
            x1, y1, x2, y2 = f['box']
            x1 = x1.detach().cpu().numpy().astype(int)
            y1 = y1.detach().cpu().numpy().astype(int)
            x2 = x2.detach().cpu().numpy().astype(int)
            y2 = y2.detach().cpu().numpy().astype(int)
            # print(f['box'])
            body = f['im']
            body = body[:, :, ::-1]
            faces = face_mod.detect_faces(body)
            for face in faces:
                x, y, w, h = face['box']
                # print(face.shape)
                frame = cv.rectangle(frame, (x + x1, y + y1), (x1 + x+w, y1 + y+h), color, thickness)
                tm = face

                face = frame[max(y - padding, 0):min(y+h+padding, frame.shape[0]-1), max(0, x-padding):min(x+w+padding, frame.shape[1] - 1), :]
                face = cv.resize(face,None ,fx=2, fy=2, interpolation=cv.INTER_CUBIC)# print(face.shape)
                
                blob = cv.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                # print(blob.dtype)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                confidence = round(tm['confidence'],4)
                # print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                # print(f'Age: {age[1:-1]} years')
                #cv.putText(frame, f'{gender}, {age},{confidence}', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv.LINE_AA)
                age_min = age[1:-1]
                mini,maxi = age_min.split('-')
                record = {'fr_num':cnt,'bb_xmin':x + x1,'bb_ymin':y + y1,'bb_height':h,'bb_width':w,'age_min':mini,'age_max':maxi,'gender':gender}
                #record = pd.DataFrame(record)
                rslt = rslt.append(record,ignore_index=True)
                # print(record,rslt.describe())

            if len(faces) == 0:
                body = cv.resize(body, (64, 128), interpolation=cv.INTER_AREA)
                result1 = b2g(tf.expand_dims(tf.convert_to_tensor(body), 0))
                result1 = result1.numpy()
                if result1 >= 0.5:
                    result1=1
                else:
                    result1=0
                # print(result1, type(result1))
                gen = ['Female', 'Male']
                w = x2 - x1
                h = y2 - y1
                # print(y)
                frame = cv.rectangle(frame, (x1, y1), (x1+w, y1+h), color, thickness)
                record = {'fr_num':cnt,'bb_xmin':x1,'bb_ymin':y1,'bb_height':h,'bb_width':w,'age_min':20,'age_max':50,'gender':gen[result1]}
                #record = pd.DataFrame(record)
                rslt = rslt.append(record,ignore_index=True)
                # print(record,rslt.describe())

        # cv.imshow('/final', frame)

        

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
    cnt = cnt+1
print(rslt)
rslt.to_csv('Outputs.csv')
