'''
*** Max likelihood multiclass Classification ***

Author :
Pranath Reddy
2016B5A30572H
'''
import pandas as pd
import math 
import numpy as np
from sklearn.model_selection import train_test_split

# A function to return a column of the data at the specified index
def col(array, i):
    return [row[i] for row in array]

# A function to calculate the mean of an array
def mean(array): 
    m = []
    for i in range(7):
        m.append(sum(col(array,i))/len(col(array,i)))
    return m

# a function to implement LRT
def rule(x_ts,x,y):
    x1 = np.array([x[i] for (i, val) in enumerate(y) if val == 1])
    x2 = np.array([x[i] for (i, val) in enumerate(y) if val == 2])
    x3 = np.array([x[i] for (i, val) in enumerate(y) if val == 3])
    m1 = mean(x1)
    m2 = mean(x2)
    m3 = mean(x3)
    cov1 = np.cov(x1.T)
    cov2 = np.cov(x2.T)
    cov3 = np.cov(x3.T)
    coeff1 = 1/(((2*3.14)**2)*np.linalg.det(cov1)**0.5)
    coeff2 = 1/(((2*3.14)**2)*np.linalg.det(cov2)**0.5)
    coeff3 = 1/(((2*3.14)**2)*np.linalg.det(cov3)**0.5)
    # likelihoods P(x|y)
    l1 = coeff1*np.exp(-0.5*np.dot(np.dot((x_ts - m1),np.linalg.inv(cov1)),(x_ts - m1).T))
    l2 = coeff2*np.exp(-0.5*np.dot(np.dot((x_ts - m2),np.linalg.inv(cov2)),(x_ts - m2).T))
    l3 = coeff3*np.exp(-0.5*np.dot(np.dot((x_ts - m3),np.linalg.inv(cov3)),(x_ts - m3).T))
    if max(l1,l2,l3) == l1:
        return 1
    elif max(l1,l2,l3) == l2:
        return 2
    else:
        return 3
 
    
def confmat(y_pred,y_ts):
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(y_ts)):
        if y_ts[i] == 1:
            if y_pred[i] == 1:
                a = a + 1
            if y_pred[i] == 2:
                b = b + 1
        if y_ts[i] == 2:
            if y_pred[i] == 1:
                c = c + 1
            if y_pred[i] == 2:
                d = d + 1
    return a, b, c, d

# input the data csv
data = pd.read_excel('data4.xlsx',header=None)
data = np.asarray(data)

x = data[:,:-1]
y = data[:,-1]
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.3)

y_pred = []
for i in range(len(x_ts)):
    y_pred.append(rule(x_ts[i],x_tr,y_tr))

y_actual = pd.Series(y_ts, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')
confmat = pd.crosstab(y_actual, y_pred)
print(confmat)
confmat = np.asarray(confmat)
Acc = (confmat[0][0] + confmat[1][1] + confmat[2][2])/sum(sum(confmat))
Acc1 = confmat[0][0]/sum(confmat[0])
Acc2 = confmat[1][1]/sum(confmat[1])
Acc3 = confmat[2][2]/sum(confmat[2])
print('Overall Accuracy : ' + str(Acc))
print('Accuracy of class 1 : ' + str(Acc1))
print('Accuracy of class 2 : ' + str(Acc2))
print('Accuracy of class 3 : ' + str(Acc3))




