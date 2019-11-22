'''
*** LRT Binary Classification ***

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
    for i in range(4):
        m.append(sum(col(array,i))/len(col(array,i)))
    return m

# a function to implement LRT
def rule(x_ts,x,y):
    p1 = len([i for (i, val) in enumerate(y) if val == 1])
    p2 = len([i for (i, val) in enumerate(y) if val == 2])
    p1, p2 = p1/(len(y)), p2/(len(y))
    x1 = np.array([x[i] for (i, val) in enumerate(y) if val == 1])
    x2 = np.array([x[i] for (i, val) in enumerate(y) if val == 2])
    m1 = mean(x1)
    m2 = mean(x2)
    cov1 = np.cov(x1.T)
    cov2 = np.cov(x2.T)
    coeff1 = 1/(((2*3.14)**2)*np.linalg.det(cov1)**0.5)
    coeff2 = 1/(((2*3.14)**2)*np.linalg.det(cov2)**0.5)
    l1 = coeff1*np.exp(-0.5*np.dot(np.dot((x_ts - m1),np.linalg.inv(cov1)),(x_ts - m1).T))
    l2 = coeff2*np.exp(-0.5*np.dot(np.dot((x_ts - m2),np.linalg.inv(cov2)),(x_ts - m2).T))
    if (l1/p2) > (l2/p1):
        return 1
    else:
        return 2
    
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
data = pd.read_excel('data3.xlsx',header=None)
data = np.asarray(data)

x = data[:,:-1]
y = data[:,-1]
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.4)

y_pred = []
for i in range(len(x_ts)):
    y_pred.append(rule(x_ts[i],x_tr,y_tr))

a, b, c, d = confmat(y_pred,y_ts)
acc = (a+d)/(a+b+c+d)
sens = (a)/(a+b)
spec = (d)/(d+c)

print('we are assuming class 1 to be positive and class2 to be negative')
print('tp: ',a,'fp: ',c,'tn: ',d,'fn: ',b)
print('accuracy: ',acc,'sensitivity: ',sens,'specificity: ',spec)




