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

input   =  Input(shape=(1000,1))
encoder =  Conv1D(10, 11 , strides=1,input_shape=(1000,1))(input)
encoder =  AveragePooling1D(pool_size=10)(encoder)
encoded =  (Dense(16))(encoder)
decoder =  UpSampling1D(size=10)(encoded)
x       =  keras.layers.Lambda(lambda x: K.expand_dims(x, axis=2))(decoder)
x       =  Conv2DTranspose(filters=1, kernel_size=(11, 1), strides=(1, 1))(x)
decoded =  keras.layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)

model = Model(inputs=input, outputs=decoded)
model.compile(optimizer=Adam(lr=0.01),loss='mean_squared_error')
model_history=model.fit(x_train,x_train,epochs=500,batch_size=100)
model.summary()

model.save("model.h5")

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('loss.png')







