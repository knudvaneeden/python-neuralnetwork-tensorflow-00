#
import keras
import numpy as np
np.random.seed(42)
#
 #
from time import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
#
from keras.callbacks import TensorBoard
#
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#

X_train = X_train.reshape(60000,784).astype('float32')
#

#---------------
# debug begin
print ( X_train )
print( type( X_train ) )
print( type( X_train ).__name__ )
exit()
# debug end
#---------------


X_train /= 255
#
X_test = X_test.reshape(10000,784).astype('float32')
X_test /= 255
#
n_classes = 10
y_train = keras.utils.to_categorical( y_train, n_classes)
y_test = keras.utils.to_categorical( y_test, n_classes)
#
model = Sequential()
#
# model.add(Dense((64), activation='tanh', input_shape=(784,)))
# model.add(Dense((64), activation='relu', input_shape=(784,)))
model.add(Dense((64), activation='sigmoid', input_shape=(784,)))
#
model.add(Dense((10), activation='softmax'))
#
model.summary()
#
# model.compile( loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.compile( loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
#
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#
model.fit( X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test), callbacks=[tensorboard])
#
classes = model.predict( X_test, batch_size = 128 )
#
def FNArrayGetIndexMaximumI( list ):
 maximum = 0
 value = 0
 I = 0
 indexI = 0
 for I, value in enumerate( list ):
  if ( value > maximum ):
   maximum = value
   indexI = I
 return( indexI )
#
# check the first predicted values
#
print( " " )
print( classes[0] )
print( y_test[0] )
print( np.amax( classes[0] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 0 ] ) )

print( " " )
print( classes[1] )
print( y_test[1] )
print( np.amax( classes[1] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 1 ] ) )

print( " " )
print( classes[2] )
print( y_test[2] )
print( np.amax( classes[2] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 2 ] ) )
#
# should be 12
#
print( " " )
print( classes[31] )
print( y_test[31] )
print( np.amax( classes[31] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 31 ] ) )
#
# for I in range( 0, 40 ):
#  print ( I )
#  print( " " )
#  print( classes[ I ] )
#  print( y_test[ I ] )
#  print( np.amax( classes[ I ] ) )
#  print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ I ] ) )
