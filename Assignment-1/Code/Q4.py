'''
***Multivariate Linear Regression***
Vector implementation

Author :
Pranath Reddy
2016B5A30572H
'''
import pandas as pd
import math 
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# A function to return the column at specified index
def getcol(data,c):
    col = []
    for i in range(len(data)):
        col.append(data[i][c])
    return col

# A function to implement min-max normalization
def norm(data):
    ndata = data
    for i in range(2):
        maxval = max(getcol(data,i))
        minval = min(getcol(data,i))
        for j in range(len(data)):
            ndata[j][i] = (data[j][i]-minval)/((maxval-minval)+0.05)
    return ndata

# import the data
data = pd.read_excel('data.xlsx',header=None)
# normalize the data
data = np.asarray(data)
data = norm(data)

# split into dependent and independent variables
x = data[:,:-1]
len = len(x)
x_temp = np.ones((len,1))
x = np.append(x_temp, x, axis=1)
y = data[:,-1]

w = 0
w = inv(np.dot(np.transpose(x),x))
w = np.dot(w,np.transpose(x))
w = np.dot(w,y)

print("The weight vector is : " + str(w))


  



