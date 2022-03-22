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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D

train_data, test_data, _, __ = Get_Dataset()

model_instance = MobileNetV2(input_tensor = Input(shape = (128, 64, 3)), weights = "imagenet", include_top = False)

headModel = model_instance.output
#headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)

model = Model(inputs = model_instance.input, outputs = headModel)

model = tf.keras.models.load_model('custom_model.h5')

for layer in model_instance.layers:
	layer.trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model.fit(train_data, epochs=5, validation_data=test_data)
model.save('custom_model.h5')
