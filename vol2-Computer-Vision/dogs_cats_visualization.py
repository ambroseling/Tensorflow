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

from keras.models import load_model
from keras import models
from keras.preprocessing import image
from keras import backend as K
tf.compat.v1.disable_eager_execution()

# model = load_model('dogs_cats_model.h5')

# model.summary()

# img_path = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/dogs_cats_original/cat.7251.jpg"

# img = image.load_img(img_path,target_size = (150,150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor,axis=0)
# img_tensor /= 255.

# layer_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = models.Model(inputs = model.input,outputs = layer_outputs)

# activations = activation_model.predict(img_tensor)

# first_layer_activation = activations[0]

# print(first_layer_activation.shape)
# print(first_layer_activation)
# plt.matshow(first_layer_activation[0,:,:,3],cmap='viridis')
# plt.show()

model = VGG16(weights = 'imagenet',
include_top = False)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std()+1e-5)
    x*=0.1
    x+=0.5  
    x = np.clip(x,0,1)
    x*=255
    x = np.clip(x,0,255).astype('uint8')
    return x

def generate_pattern(layer_name,filter_index,size = 150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    grads = K.gradients(loss,model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
    iterate = K.function([model.input],[loss,grads])
    input_img_data = np.random.random((1,size,size,3))*20+128
    step = 1
    for i in range(40):
        loss_value , grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

layer_name = 'block1_conv1'
size = 64
margin = 5
results = np.zeros((8*size+7*margin,8*size+7*margin,3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i+(j*8),size = size)
        horizontal_start = i*size + i*margin
        horizontal_end = horizontal_start + size
        vertical_start = j*size + j*margin
        vertical_end = vertical_start +size
        results[horizontal_start:horizontal_end,vertical_start:vertical_end,:] = filter_img
plt.figure(figsize=(20,20))
plt.imshow(results)
plt.show()

model = VGG16(weights = 'imagenet')
