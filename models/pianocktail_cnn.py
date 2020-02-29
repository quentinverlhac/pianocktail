import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Reshape, Conv2D, Flatten, Dense, AveragePooling2D, MaxPool2D

import config


class PianocktailCNN(Model):
    def __init__(self, name=config.ModelEnum.PIANOCKTAIL_CNN.value, subspectrogram_points = config.SUBSPECTROGRAM_POINTS, mel_bins = config.MEL_BINS, number_of_emotions = config.NUMBER_OF_EMOTIONS):
        super(PianocktailCNN,self).__init__(name=name)

        self.input_layer = InputLayer((subspectrogram_points, mel_bins), name=f"{name}_input")

        self.reshape = Reshape((subspectrogram_points, mel_bins, 1), name=f"{name}_reshape") 

        self.conv1 = Conv2D(filters=6,kernel_size=(5,7),activation=tf.nn.relu,name=f"{name}_conv1")

        self.pooling1 = AveragePooling2D(pool_size=2,name=f"{name}_pool1")

        self.conv2 = Conv2D(filters=16,kernel_size=5,activation=tf.nn.relu,name=f"{name}_conv2")

        self.pooling2 = AveragePooling2D(pool_size=2,name=f"{name}_pool2")

        self.flatten = Flatten(name=f"{name}_flatten")

        self.dense1 = Dense(200,activation=tf.nn.relu,name=f"{name}_dense1")

        self.dense2 = Dense(config.NUMBER_OF_EMOTIONS, activation=tf.nn.sigmoid, name=f"{name}_output")

    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.reshape(net)
        net = self.conv1(net)
        net = self.pooling1(net)
        net = self.conv2(net)
        net = self.pooling2(net)
        net = self.flatten(net)
        net = self.dense1(net)
        net = self.dense2(net)
        return net