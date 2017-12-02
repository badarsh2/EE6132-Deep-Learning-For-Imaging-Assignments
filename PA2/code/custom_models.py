import keras
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.regularizers import l2


def baseline_model(input_shape, num_classes):
    """ Creates a baseline CNN with the following architecture:
      input - conv (32 3x3 filters, stride 1, zero padding of 1) - 2x2 maxpool with
      stride 2 - fully connected (10 outputs) - softmax classifier

      Params:
        input_shape: Shape of input array 
        num_classes: Number of classes in the output classification

      Returns:
        Sequential: keras model
  """
    model = Sequential()
    model.add(ZeroPadding2D(padding=1, input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(num_classes, kernel_regularizer=l2(0.01), name="logits"))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])
    return model


def two_conv_layer_model(input_shape, num_classes):
    """ Creates a two conv layer CNN with the following architecture:
      input - conv1 (32 3x3 filters, stride 1, zero padding of 1) - 2x2 maxpool
      with stride 2 - conv2 (32, 3x3 filters,stride 1, zero padding of 1) - 2x2 maxpool with stride 2 - fully
      connected (10 outputs) - softmax classifier

      Params:
        input_shape: Shape of input array 
        num_classes: Number of classes in the output classification

      Returns:
        Sequential: keras model
  """
    model = Sequential()
    model.add(ZeroPadding2D(padding=1, input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(num_classes, kernel_regularizer=l2(0.01), name="logits"))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])
    return model


def two_conv_one_dense_layer_model(input_shape, num_classes):
    """ Creates a two conv layer + 1 hidden dense layer CNN with the following architecture:
      input - conv1 (32 3x3 filters, stride 1, zero padding of 1) - 2x2 maxpool
      with stride 2 - conv2 (32 3x3 filters, stride 1, zero padding of 1) - 2x2 maxpool with stride 2 - fully
      connected (500 outputs)- fully connected (10 outputs) - softmax classifier

      Params:
        input_shape: Shape of input array 
        num_classes: Number of classes in the output classification

      Returns:
        Sequential: keras model
  """
    model = Sequential()
    model.add(ZeroPadding2D(padding=1, input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name="maxpooling2"))
    model.add(Flatten())
    model.add(Dense(500, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(num_classes, kernel_regularizer=l2(0.01), name="logits"))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])
    return model
