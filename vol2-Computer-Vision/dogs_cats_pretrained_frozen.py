import sys
import os,shutil
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
from keras.applications import VGG16
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


#STEP 1: orgnaize all the directories needed for training,validatoin and testing for both cats and dogs
original_dataset_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/dogs_cats_original"

base_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision"



train_dir = os.path.join(base_dir,'dogs_cats_train')
shutil.rmtree(train_dir)
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'dogs_cats_validation')
shutil.rmtree(validation_dir)
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'dogs_cats_test')
shutil.rmtree(test_dir)
os.mkdir(test_dir)


train_cats_dir = os.path.join(train_dir,'cats_train')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs_train')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cats_validation')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir,'dogs_validation')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats_test')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'dogs_test')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(3000,4000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)


fnames = ['cat.{}.jpg'.format(i) for i in range(4000,5000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(3000,4000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(4000,5000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)


#STEP 2: data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'    
)


conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape = (150,150,3)
)

model = tf.keras.models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))



model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                metrics = ["acc"])
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps=50
)
model.save("dogs_cats_model.h5")

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history['loss']
val_loss = history.history["val_loss"]

model.evaluate(test_generator)

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label="Training acc")
plt.plot(epochs,val_acc,'b',label="Training acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Training loss")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()

#Takes too long to train, need to figure out GPU usage on mac before training