import sys
import os
import tensorflow.keras
import pathlib
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.utils import to_categorical
import matplotlib.image as mpimg
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

#Walk through pizza_steak directory and list number of files
for dirpath,dirnames,filenames in os.walk("/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/pizza_steak"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

data_dir = pathlib.Path("pizza_steak/train")
#Create a list of class_names from the subdirectories
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)

def view_random_image(target_dir,target_class):
  #Set up the targey directory (we'll view images from here)
  target_folder = target_dir+target_class

  #Get a random image path
  random_image = random.sample(os.listdir(target_folder),1)


  #Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder+"/"+random_image[0])
  plt.imshow(img)
  #plt.show()
  plt.title(target_class)
  plt.axis("off")
  print(f"Imgae shape: {img.shape}")
  return img

img=view_random_image(target_dir="/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/pizza_steak/train/",target_class="pizza")

#set the seed
tf.random.set_seed(42)

#Preprocess data (get all of the pixel values between 0 and 1 also called)
#Generate batches of tensor image data with real-time data augmentation.

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

#Set up paths to our data directories
train_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/pizza_steak/train"
test_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/pizza_steak/test"


#Import data from directories and turn into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,batch_size=32,target_size=(224,224),class_mode="binary",seed=42)

valid_data = valid_datagen.flow_from_directory(directory=test_dir,batch_size=32,target_size=(224,224),class_mode="binary",seed=42)

#Build a CNN model
model = tf.keras.models.Sequential([
 #input layer
  tf.keras.layers.Conv2D(filters=10,
                        kernel_size=3,
                        activation="relu",
                        input_shape=[224,224,3]),
 #same layer as above
  tf.keras.layers.Conv2D(10,3,activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2,padding="valid"),
  tf.keras.layers.Conv2D(10,3,activation="relu"),
  tf.keras.layers.Conv2D(10,3,activation="relu"),
   tf.keras.layers.MaxPool2D(2),
   tf.keras.layers.Flatten(),
   #you can seperate the activation functions from your layer
   tf.keras.layers.Dense(1,activation="sigmoid")
])

#Compile our CNN
model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

#Fit the model
history= model.fit(train_data,
                       epochs=5,
                       #computer has limited amount of memory, dividing into batches allows chip to learn patterns
                       #in smaller sizes (47 batches of 32 images)
                       steps_per_epoch=len(train_data),
                       validation_data=valid_data,
                       validation_steps=len(valid_data))

