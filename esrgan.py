import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import cv2

IMAGE_PATH = "original.png"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def preprocess_image(image):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.convert_to_tensor(image)

  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

cap = cv2.VideoCapture('indiancctv.mp4')
model = hub.load(SAVED_MODEL_PATH)

print(1)
while True:
    ret, frame = cap.read()
    if ret:
        #frame = preprocess_image(frame)
        result = cv2.resize(frame,None ,fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        #result = tf.squeeze(model(frame)).numpy()
        #print(result)
        #result = result.astype(np.uint8)
        cv2.imshow('frame', result)
        cv2.imshow('original',frame)
        cv2.waitKey(1)
