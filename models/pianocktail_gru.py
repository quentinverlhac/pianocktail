import tensorflow as tf 

from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, GRU, GRUCell, Dense, InputLayer

import config

class PianocktailGRU(Model):
    def __init__(self, name='pianocktail_gru', subspectrogram_points = config.SUBSPECTROGRAM_POINTS, mel_bins = config.MEL_BINS, number_of_emotions = config.NUMBER_OF_EMOTIONS, **kwargs):
        super(PianocktailGRU, self).__init__(name=name, **kwargs)

        self.input_layer = InputLayer((subspectrogram_points, mel_bins,1), name=f"{name}_input")

        self.reshape = Reshape((subspectrogram_points, mel_bins), name=f"{name}_reshape") 
        
        self.gru = GRU(subspectrogram_points, name=f"{name}_gru")

        self.dense = Dense(number_of_emotions, activation=tf.nn.sigmoid, name=f"{name}_output")

    def call(self, inputs, training=True):
        net = self.input_layer(inputs)
        net = self.reshape(net)
        net = self.gru(net, training=training)
        net = self.dense(net)
        return net
