'''
*** Stacked autoencoder ***
Binary Classification

Author :
Pranath Reddy
2016B5A30572H
'''

import pandas as pd
import math 
import numpy as np
from mat4py import loadmat
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras import optimizers
from sklearn.model_selection import train_test_split

def set(y):
    for i in range(len(y)):
        if(y[i]>0.5):
            y[i] = 1.0
        if(y[i]<=0.5):
            y[i] = 0.0
    return y

data = loadmat('./data5.mat')
data = pd.DataFrame(data)
data = np.asarray(data)

data_temp = []
for i in range(len(data)):
    data_temp.append(data[i][0])

data_temp = np.asarray(data_temp)
data = data_temp
y = data[:,-1]
x = data[:,:-1]
x = (x - np.mean(x,axis=0))/np.std(x,axis=0)
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.3)
m = x_tr.shape[0]
n = x_tr.shape[1]

batch_size = 128
epochs = 5000

# Pretraining Models
# Layer 1
input1 = Input(shape = (n, ))
encoder1 = Dense(120, activation = 'sigmoid')(input1)
decoder1 = Dense(n, activation = 'sigmoid')(encoder1)

autoencoder1 = Model(input = input1, output = decoder1)
encoder_layer1 = Model(input = input1, output = encoder1)

# Layer 2
input2 = Input(shape = (120,))
encoder2 = Dense(100, activation = 'sigmoid')(input2)
decoder2 = Dense(120, activation = 'sigmoid')(encoder2)

autoencoder2 = Model(input = input2, output = decoder2)
encoder_layer2 = Model(input = input2, output = encoder2)

# Layer 3 
input3 = Input(shape = (100,))
encoder3 = Dense(80, activation = 'sigmoid')(input3)
decoder3 = Dense(100, activation = 'sigmoid')(encoder3)

autoencoder3 = Model(input = input3, output = decoder3)

sgd = optimizers.SGD(lr=0.1)

autoencoder1.compile(loss='binary_crossentropy', optimizer = sgd)
autoencoder2.compile(loss='binary_crossentropy', optimizer = sgd)
autoencoder3.compile(loss='binary_crossentropy', optimizer = sgd)

encoder_layer1.compile(loss='binary_crossentropy', optimizer = sgd)
encoder_layer2.compile(loss='binary_crossentropy', optimizer = sgd)

autoencoder1.fit(x_tr, x_tr, epochs= 1000, batch_size = 512, verbose=0)
layer2_input = encoder_layer1.predict(x_tr)
autoencoder2.fit(layer2_input, layer2_input, epochs= 1000, batch_size = 512, verbose=0)
layer3_input = encoder_layer2.predict(layer2_input)
autoencoder3.fit(layer3_input, layer3_input, epochs= 1000, batch_size = 512, verbose=0)

# Main Model
# stacked Autoencoder

encoder1_sa = Dense(120, activation = 'sigmoid')(input1)
encoder2_sa = Dense(100, activation = 'sigmoid')(encoder1_sa)
encoder3_sa = Dense(80, activation = 'sigmoid')(encoder2_sa)
final_output = Dense(1, activation = 'sigmoid')(encoder3_sa)

stack_autoencoder = Model(input = input1, output = final_output)

stack_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd)

stack_autoencoder.layers[1].set_weights(autoencoder1.layers[1].get_weights()) 
stack_autoencoder.layers[2].set_weights(autoencoder2.layers[1].get_weights()) 
stack_autoencoder.layers[3].set_weights(autoencoder3.layers[1].get_weights()) 

stack_autoencoder.fit(x_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=1)

yp = stack_autoencoder.predict(x_ts, batch_size=batch_size)

yp_temp = []
for i in range(len(yp)):
    yp_temp.append(yp[i][0])
yp = yp_temp
yp = set(yp)

y_actual = pd.Series(y_ts, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
confmat = pd.crosstab(y_actual, y_pred)

print(confmat)
confmat = np.asarray(confmat)
tp = confmat[1][1]
tn = confmat[0][0]
fp = confmat[0][1]
fn = confmat[1][0]

Acc = float(tp+tn)/float(tp+tn+fp+fn)
SE = float(tp)/float(tp+fn)
SP = float(tn)/float(tn+fp)

print('Accuracy : ' + str(Acc))
print('sensitivity : ' + str(SE))
print('specificity : ' + str(SP))



















































