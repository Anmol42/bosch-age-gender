import torch
import cv2
from utils import *
import numpy as np
import models

srgan_checkpoint = "./checkpoint_srgan.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srgan_generator = torch.load(srgan_checkpoint, map_location=device)['generator'].to(device)
srgan_generator.eval()

cap = cv2.VideoCapture('cctv.mp4')

while True:
    ret, frame = cap.read()
    if ret:
        frm = torch.tensor(frame)
        result = srgan_generator(frame)
        print(result)
    else:
        break
