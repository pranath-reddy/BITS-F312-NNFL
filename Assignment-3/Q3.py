'''
*** ANFIS ***
Multiclass Classification

Author :
Pranath Reddy
2016B5A30572H
'''

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

# Function to set the class labels to predictions
def set(y):
    for i in range(len(y)):
        if(0.0<y[i]<=1.5):
            y[i] = 1.0
        if(1.5<y[i]<=2.5):
            y[i] = 2.0
        if(y[i]>=2.5):
            y[i] = 3.0
    return y

# Function to normalize the data
def norm(x):
    return (x - x.mean(axis=0))/x.std(axis=0)

# Adaptive neuro-fuzzy inference system implementation
class ANFIS:

    def __init__(self, n_inputs, n_rules, learning_rate=1e-2):
        self.n = n_inputs
        self.m = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, n_inputs))  # Input
        self.targets = tf.placeholder(tf.float32, shape=None)  # Desired output
        mu = tf.get_variable("mu", [n_rules * n_inputs],
                             initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        sigma = tf.get_variable("sigma", [n_rules * n_inputs],
                                initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS
        y = tf.get_variable("y", [1, n_rules], initializer=tf.random_normal_initializer(0, 1))  # Sequent centers

        self.params = tf.trainable_variables()

        self.rul = tf.reduce_prod(
            tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), mu)) / tf.square(sigma)),
                       (-1, n_rules, n_inputs)), axis=2)  # Rule activations
        # Fuzzy base expansion function:
        num = tf.reduce_sum(tf.multiply(self.rul, y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12)
        self.out = tf.divide(num, den)

        self.loss = tf.losses.huber_loss(self.targets, self.out)  # Loss function computation
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)  # Optimization step
        self.init_variables = tf.global_variables_initializer()  # Variable initializer

    # Function to get predictions from test samples
    def infer(self, sess, x, targets=None):
        if targets is None:
            return sess.run(self.out, feed_dict={self.inputs: x})
        else:
            return sess.run([self.out, self.loss], feed_dict={self.inputs: x, self.targets: targets})

    # Function to initiate and train the graph
    def train(self, sess, x, targets):
        yp, l, _ = sess.run([self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        return l, yp
    
# Importing the data
data = pd.read_excel('data4.xlsx')
data = pd.DataFrame(data)
data = np.asarray(data)
y = data[:,-1]
x = data[:,:-1]
x = norm(x)

# Split train and test set
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.3)
m = x_tr.shape[0]
n = x_tr.shape[1]

# Hyperparameters
m = 16  # number of rules
alpha = 0.01  # learning rate
epochs= 2000
fis = ANFIS(n_inputs=7, n_rules=m, learning_rate=alpha)

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(epochs):
        # Train the model
        trn_loss, train_pred = fis.train(sess, x_tr, y_tr)
        # Evaluate on test set
        test_pred, val_loss = fis.infer(sess, x_ts, y_ts)
        # Print the training cost
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
        if epoch == epochs - 1:
            time_end = time.time()

yp = test_pred # Get the predictions
yp = set(yp)

# Confusion matrix and accuracy
y_actual = pd.Series(y_ts, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
confmat = pd.crosstab(y_actual, y_pred)
print(confmat)

confmat = np.asarray(confmat)
Accuracy = float(confmat[0][0]+confmat[1][1]+confmat[2][2])/float(yp.shape[0])
print('Accuracy :' + ' ' + str(Accuracy))

