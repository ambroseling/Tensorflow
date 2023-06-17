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

import requests
requests.packages.urllib3.disable_warnings()
import ssl
from keras.applications import VGG16
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

original_dataset_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/dogs_cats_original"

base_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision"

train_dir = os.path.join(base_dir,'dogs_cats_train')

validation_dir = os.path.join(base_dir,'dogs_cats_validation')

test_dir = os.path.join(base_dir,'dogs_cats_test')

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape = (150,150,3)
)

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory,sample_count):
    features = np.zeros(shape=(sample_count,4,4,512)) #output shape of VGG16 ImageNet
    labels= np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size = batch_size,
        class_mode='binary'
    )
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size >= sample_count:
            break
    return features,labels

train_features,train_labels = extract_features(train_dir, 2000)
validation_features,validation_labels = extract_features(validation_dir, 1000)
test_features,test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features,(2000,4*4*512))
validation_features = np.reshape(validation_features,(1000,4*4*512))
test_features = np.reshape(test_features,(1000,4*4*512))

model = tf.keras.models.Sequential()
model.add(layers.Dense(256,activation="relu",input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                metrics = ["acc"])

history = model.fit(train_features,train_labels,
epochs=30,
batch_size = 20,
validation_data=(validation_features,validation_labels))


acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history['loss']
val_loss = history.history["val_loss"]

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label="Training acc")
plt.plot(epochs,val_acc,'b',label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()

