'''
*** RBFNN ***
Binary Classification

Author :
Pranath Reddy
2016B5A30572H
'''

import pandas as pd
import math 
import numpy as np
from mat4py import loadmat
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

# p - no of neurons in hidden layer
p = 125

cval = np.zeros((x_tr.shape[0],1))
indexes = np.random.randint(0,x_tr.shape[0],p)
centers = x_tr[indexes]
l2_norm = np.zeros((x_tr.shape[0],1))
beta = np.zeros((p,1))

def compute_distance(feature_centers,datapoint):
    return np.sum(np.power(datapoint-feature_centers,2),axis=1)

def kmeans(data,num_cluster_centers,epochs=1000):
    cluster = np.zeros((data.shape[0],1))
    center_indexes = np.random.random_integers(0,data.shape[0],num_cluster_centers)
    feature_centers = data[center_indexes]

    for epoch in range(epochs):
        distances = np.zeros((num_cluster_centers,1))
        for datapoint in range(data.shape[0]):
            distances = compute_distance(feature_centers,data[datapoint,:])
            cluster_index = np.argmin(distances)
            cluster[datapoint,0] = cluster_index

        for i in range(num_cluster_centers):
            cluster_points_indices = np.argwhere(cluster == i)
            cluster_points = data[cluster_points_indices[:,0]]
            if cluster_points.shape[0] != 0:
                feature_centers[i] = np.mean(cluster_points,axis=0)

    return feature_centers

centers = kmeans(x_tr,p)

H= np.zeros((x_tr.shape[0],p))
for i in range(x_tr.shape[0]):
    for j in range(p):
        H[i][j] = np.linalg.norm(x_tr[i]-centers[j])

H_test = np.empty((x_ts.shape[0],p), dtype= float)
for i in range(x_ts.shape[0]):
    for j in range(p):
        H_test[i][j] = np.linalg.norm(x_ts[i]-centers[j])

H = np.matrix(H)
print(H.shape)
print(y_tr.shape)
W= np.dot(H.I,y_tr)
print(W.shape)
y_pred = np.dot(H_test,W.T)

y_pred_temp = []
y_pred = np.asarray(y_pred)
for i in range(y_pred.shape[0]):
    y_pred_temp.append(y_pred[i][0])
yp = set(y_pred_temp)

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









