"""
Logical operators DNN training
"""

#tlearn logical
#http://tflearn.org

#https://medium.com/@ilblackdragon
# https://raw.githubusercontent.com/tflearn/tflearn/master/examples/basics/logical.py
# http://www.jorditorres.org/introduccion-practica-al-deep-learning-con-tensorflow-de-google-parte-5/
# http://www.jorditorres.org/libro-hello-world-en-tensorflow/
# http://www.jorditorres.org/wp-content/uploads/2016/02/FirstContactWithTensorFlow.part1_.pdf

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn

def train_dnn(X, Y, input_num, first_layer, second_layer):
    # Graph definition
    with tf.Graph().as_default():
        g = tflearn.input_data(shape=[None, input_num])
        g = tflearn.fully_connected(g, first_layer, activation='linear')
        g = tflearn.fully_connected(g, second_layer, activation='linear')
        g = tflearn.fully_connected(g, 1, activation='sigmoid')
        g = tflearn.regression(g, optimizer='sgd', learning_rate=2., loss='mean_square')
	
        #Model training
        m = tflearn.DNN(g)
        m.fit(X, Y, n_epoch=100, snapshot_epoch=False)

    return m

# Logical NOT operator
X = [[0.], [1.]]
Y = [[1.], [0.]]

m = train_dnn(X, Y, 1, 128, 128)

# Test model
print("Testing NOT operator")
print("NOT 0:", m.predict([[0.]]))
print("NOT 1:", m.predict([[1.]]))

# Logical OR operator
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [1.], [1.], [1.]]

m = train_dnn(X, Y, 2, 128, 128)

#Test model
print("Testing OR operator")
print("0 or 0:", m.predict([[0., 0.]]))
print("0 or 1:", m.predict([[0., 1.]]))
print("1 or 0:", m.predict([[1., 0.]]))
print("1 or 1:", m.predict([[1., 1.]]))

# Logical AND operator
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [0.], [0.], [1.]]

m = train_dnn(X, Y, 2, 128, 128)

print("Testing AND operator")
print("0 and 0:", m.predict([[0., 0.]]))
print("0 and 1:", m.predict([[0., 1.]]))
print("1 and 0:", m.predict([[1., 0.]]))
print("1 and 1:", m.predict([[1., 1.]]))

# Data
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_nand = [[1.], [1.], [1.], [0.]]
Y_or = [[0.], [1.], [1.], [1.]]

m = train_dnn(
# m = train_dnn(X, Y, 2, 32, 32)


