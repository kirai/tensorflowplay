import json
from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_test = X_test.astype('float32')
X_test /= 255
nb_classes = 10
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Load trained model
json_string = open('model.json').read()  
model_config = model_from_json(json_string).get_config()
model = Sequential.from_config(model_config)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics=['accuracy'])
model.load_weights('modelweights.hdf5')

# Evaluate with previously trained model
score = model.evaluate(X_test, Y_test, verbose = 0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
