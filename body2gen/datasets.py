import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import torch.utils.data as data
import pandas as pd

description = ['img', 'Age16-30',
                        'Age31-45',
                        'Age46-60',
                        'AgeAbove61',
                        'Backpack',
                        'CarryingOther',
                        'Casual lower',
                        'Casual upper',
                        'Formal lower',
                        'Formal upper',
                        'Hat',
                        'Jacket',
                        'Jeans',
                        'Leather Shoes',
                        'Logo',
                        'Long hair',
                        'Male',
                        'Messenger Bag',
                        'Muffler',
                        'No accessory',
                        'No carrying',
                        'Plaid',
                        'PlasticBags',
                        'Sandals',
                        'Shoes',
                        'Shorts',
                        'Short Sleeve',
                        'Skirt',
                        'Sneaker',
                        'Stripes',
                        'Sunglasses',
                        'Trousers',
                        'Tshirt',
                        'UpperOther',
                        'V-Neck']
req_des = ['img',
            'Male']

class MultiLabelDataset(data.Dataset):
    def __init__(self, label):
        df = pd.read_csv(label,delimiter=' ', header=None)
        df = df.drop(36, axis=1)
        df.columns = description
        self.df= df[req_des]
        print(df)
    
    def __len__(self):
        return len(self.df)

attr_nums = 1

def Get_Dataset():

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True,
            rotation_range=10,
            horizontal_flip=True,
            rescale=1/255.0)

    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True,
            rotation_range=10,
            rescale=1/255.0)

    train_dataset = MultiLabelDataset(label='PETA_train_list.txt').df
    val_dataset = MultiLabelDataset(label='PETA_test_list.txt').df

    train_data = train_gen.flow_from_dataframe(
            train_dataset,
            x_col='img',
            y_col=req_des[1:],
            target_size=(128, 64),
            class_mode='multi_output')

    val_data = val_gen.flow_from_dataframe(
            val_dataset,
            x_col='img',
            y_col=req_des[1:],
            target_size=(128, 64),
            class_mode='multi_output')

    return train_data, val_data, attr_nums, description
