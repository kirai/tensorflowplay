import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Input data (Spiral with three types of points)
 
np.random.seed(0)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])

#fig.savefig('spiral_raw.png')

# Train a linear classifier

with tf.Graph().as_default():
    net = tflearn.input_data([None, 2])
    net = tflearn.fully_connected(net, 100, activation='relu', weights_init='normal',
                                  regularizer='L2', weight_decay=0.001)
    net = tflearn.fully_connected(net, 3, activation='softmax')
    sgd = tflearn.SGD(learning_rate=1.0, lr_decay=0.96, decay_step=500)
    net = tflearn.regression(net, optimizer=sgd, loss='categorical_crossentropy')

    Y = to_categorical(y, 3)
    model = tflearn.DNN(net)
    model.fit(X, Y, show_metric=True, batch_size=len(X), n_epoch=30000, snapshot_epoch=False)

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net.png')
print("Accuracy: {}%".format(100 * np.mean(y == np.argmax(model.predict(X), axis=1))))

