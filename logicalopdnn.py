# -*- coding: utf-8 -*-
"""
Logical operators DNN training
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn

# Graph definition
def define_dnn_topology(input_num, first_layer, second_layer):
    with tf.Graph().as_default():
        g = tflearn.input_data(shape=[None, input_num])
        g = tflearn.fully_connected(g, first_layer, activation='linear')
        g = tflearn.fully_connected(g, second_layer, activation='linear')
        g = tflearn.fully_connected(g, 1, activation='sigmoid')
        g = tflearn.regression(g, optimizer='sgd', learning_rate=2., loss='mean_square')

    return g 

#Model training
def train_dnn(X, Y, g, n_epoch):
    m = tflearn.DNN(g)
    m.fit(X, Y, n_epoch=n_epoch, snapshot_epoch=False)

    return m

# Logical NOT operator
X = [[0.], [1.]]
Y = [[1.], [0.]]

g = define_dnn_topology(1, 128, 128)
m = train_dnn(X, Y, g, 100)

# Test model
print("Testing NOT operator")
print("NOT 0:", m.predict([[0.]]))
print("NOT 1:", m.predict([[1.]]))

# Logical OR operator
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [1.], [1.], [1.]]

g = define_dnn_topology(2, 128, 128)
m = train_dnn(X, Y, g, 100)

#Test model
print("Testing OR operator")
print("0 or 0:", m.predict([[0., 0.]]))
print("0 or 1:", m.predict([[0., 1.]]))
print("1 or 0:", m.predict([[1., 0.]]))
print("1 or 1:", m.predict([[1., 1.]]))

# Logical AND operator
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [0.], [0.], [1.]]

g = define_dnn_topology(2, 128, 128)
m = train_dnn(X, Y, g, 100)

print("Testing AND operator")
print("0 and 0:", m.predict([[0., 0.]]))
print("0 and 1:", m.predict([[0., 1.]]))
print("1 and 0:", m.predict([[1., 0.]]))
print("1 and 1:", m.predict([[1., 1.]]))

# Data
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_nand = [[1.], [1.], [1.], [0.]]
Y_or = [[0.], [1.], [1.], [1.]]

g_nand = define_dnn_topology(2, 32, 32) 
g_or = define_dnn_topology(2, 32, 32)

# XOR Merging Nand and OR operators 
g_xor = tflearn([g_nand, g_or], mode='elewise_mul')
m = train_dnn(X, [Y_nand, Y_or], g_xor, 400)

# Testing
print("Testing XOR operator")
print("0 xor 0:", m.predict([[0., 0.]]))
print("0 xor 1:", m.predict([[0., 1.]]))
print("1 xor 0:", m.predict([[1., 0.]]))
print("1 xor 1:", m.predict([[1., 1.]]))

