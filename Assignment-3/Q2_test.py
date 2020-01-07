'''
*** 1D Convolutional Autoencoder  ***
Author :
Pranath Reddy
2016B5A30572H
'''

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, Conv1D, AveragePooling1D, UpSampling1D, Conv2DTranspose
from keras.utils import *
from keras.optimizers import *
import keras.backend as K
from mat4py import loadmat
from sklearn import preprocessing
from keras.models import load_model

x = loadmat('data_for_cnn.mat')
x = pd.DataFrame(x)
x = np.asarray(x)

x_temp = []
for i in range(len(x)):
    x_temp.append(x[i][0])
x_temp = np.asarray(x_temp)
x = x_temp
x = preprocessing.normalize(x)
x = x.reshape(x.shape[0],x.shape[1],1)
x_train = x[:900,:,:]
x_test = x[900:,:,:]

model = load_model('A3Q2.h5')
model.save("A3Q2.h5")

x_pred= model.predict(x_test)
n=10
for i in range(2,10):
    plt.subplot(n,2,2*i-1)
    plt.plot(x_pred[i].reshape(-1,1),color='blue')
    plt.subplot(n,2,2*i)
    plt.plot(x_test[i],color='green')
plt.show()



