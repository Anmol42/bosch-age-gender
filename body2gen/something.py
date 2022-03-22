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

vgg19_new = tf.keras.applications.VGG19(include_top=False, input_shape = (128, 64, 3))

for layers in vgg19_new.layers:
    layers.trainable=False

model = Sequential([
    vgg19_new,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='model')

model = tf.keras.models.load_model('./model.h5')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model.fit(train_data, epochs=2, validation_data=test_data)
model.save('./model.h5')
