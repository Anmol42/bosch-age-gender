import torch
import cv2
from utils import *
import numpy as np

srgan_checkpoint = "./checkpoint_srgan.pth.tar"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

srgan_generator = torch.load(srgan_checkpoint)['generator']#.to(device)
srgan_generator.eval()

img = torch.squeeze(torch.tensor(cv2.imread('girl1.png')))
img = torch.reshape(img, (3, 480, 720)).to(torch.float32)
print(img)
out = srgan_generator(img)
