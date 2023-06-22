import sys
import os
import pathlib
import platform
import random

import tensorflow as tf
import tensorflow.keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv2D, MaxPool2D,Flatten, Dense,Activation
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.utils import to_categorical

import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
sys.path.insert(1, '/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow')
import helper_functions

#Walk thorugh 10 percent data directory and list number of files
for dirpath,dirnames,filenames in os.walk("/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/101_food_classes_10_percent"):
  print(f"there are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


#Set up data inputs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helper_functions import plot_loss_curves,create_tensorboard_callback,compare_historys,make_confusion_matrix

import tensorflow_datasets as tensorflow_datasets
#Load in the data (take 5-6 minutes)
(train_data,test_data),ds_info = tfds.load(name="food101",
                                           split=["train","validation"],
                                           shuffle_files = True,
                                           as_supervised=True, #data gets returned in tuple format (data,label)
                                           with_info = True)


#Get the class names
class_names = ds_info.features["label"].names
#Tamke one sample of trained data
train_one_sample = train_data.take(1) #samples are in format (imge_tensor,label)

# Important things to look for:
# class names
# shape of our input data
# the datatype of our input data (image tensors)
# the datatype of our input data
# what the labels look like (one hot encoded or label encoded)
# do the labels match up

