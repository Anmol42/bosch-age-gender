import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
# from utils import *
from PIL import Image, ImageDraw, ImageFont

cap = cv.VideoCapture('cam1.mp4')

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes = [0]

# srgan_checkpoint = "./checkpoint_srgan.pth.tar"
# srgan_generator = torch.load(srgan_checkpoint)['generator']
# srgan_generator.eval()

while True:
	ret, frame = cap.read()
	if ret:
		frame = cv.resize(frame, dsize=(860, 480))
		cv.imshow('Frame', frame)
		results = model(frame)
		crops = results.crop(save=False)
		detection = np.squeeze(results.render())
		cv.imshow('BB', detection)
		if len(crops):
			prs_1 = crops[0]['im']
			prs_1 = prs_1[:,:,::-1]
			cv.imshow('Person_1', prs_1)
			print(prs_1.shape)

		# img = torch.squeeze(torch.tensor(cv.imread('img.png')))
		# img = torch.reshape(img, (3, 1800, 2880)).to(torch.float32)
		# print(img)
		# out = srgan_generator(img)

	if cv.waitKey(25) & 0xFF == ord('q'):
		cv.destroyAllWindows()
		break;
