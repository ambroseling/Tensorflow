import sys
import tensorflow.keras
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol1-Fundamentals/breast-cancer.csv")

X = data.drop(["happiness"],axis=1)

y = data["happiness"]
