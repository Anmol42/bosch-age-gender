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
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from datasets import Get_Dataset

train_data, test_data, _, __ = Get_Dataset()

model = Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='model')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
model.fit(train_data, epochs=5, validation_data=train_data)
model.save('./model_1.h5')
