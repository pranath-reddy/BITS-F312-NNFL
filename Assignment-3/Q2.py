'''
*** 1D Convolutional Autoencoder  ***

Author :
Pranath Reddy
2016B5A30572H
'''

from mat4py import loadmat
import numpy as np
import pandas as pd
from keras import Sequential
from keras import optimizers
from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPooling1D,UpSampling1D,Reshape,Flatten
import matplotlib.pyplot as plt
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

#input-convolution layer-pooling layer-FC-upsampling layer-transpose convolution layer
input = Input(shape=(1000,1))
encoder = Conv1D(32, 5, activation= 'relu' , padding= 'same')(input)
encoder = MaxPooling1D(4, padding= 'same')(encoder)
encoder = Flatten()(encoder)
encoded = Dense(500, activation='softmax')(encoder)
decoder = UpSampling1D(2)(encoded)
decoder = Reshape((1000,1))(decoder)
decoded = Conv1D(1, 5, activation='sigmoid', padding='same')(decoder)
autoencoder = Model(input, decoded)
autoencoder.summary()

opt = optimizers.Adam(lr=0.01)
autoencoder.compile(optimizer= opt, loss='mse')
#autoencoder.compile(optimizer= opt, loss='binary_crossentropy')
history = autoencoder.fit(x, x, epochs=2000, batch_size=512, shuffle=True)

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('loss.png')






