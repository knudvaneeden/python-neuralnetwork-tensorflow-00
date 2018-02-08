import numpy as np
np.random.seed(42)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print( "X train.shape = ", X_train.shape )
print( "y_train.shape = ", y_train.shape )
print( "y_train[0:99] = ", y_train[0:99] )
print( "X_train[0] = ", X_train[0] )
print( "y_test.shape = ", y_test.shape )
print( "X_test.shape = ", X_test.shape )
X_test = X_test.reshape(10000,784).astype('float32')
X_train = X_train.reshape(60000,784).astype('float32')
X_test = X_test.reshape(10000,784).astype('float32')
X_train /= 255
X_test /= 255
print( "X_train[0] = ", X_train[0] )
n_classes = 10
y_train = keras.utils.to_categorical( y_train, n_classes)
y_test = keras.utils.to_categorical( y_test, n_classes)
print( "y_test[0] = ", y_test[0] )
print( "y_train[0] = ", y_train[0] )
model = Sequential()
model.add(Dense((64), activation='sigmoid', input_shape=(784,)))
model.add(Dense((10), activation='softmax'))
model.summary()
model.compile( loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit( X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
