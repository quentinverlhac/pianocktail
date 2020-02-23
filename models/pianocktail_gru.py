import tensorflow as tf 

from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, GRU, GRUCell, Dense, InputLayer

import config

class PianocktailGRU(Model):
    def __init__(self, name='pianocktail_gru', **kwargs):
        super(PianocktailGRU, self).__init__(name=name, **kwargs)

        self.input_layer = InputLayer((config.SUBSPECTROGRAM_POINTS,config.MEL_POINTS,1), name=f"{name}_input")

        self.reshape = Reshape([config.SUBSPECTROGRAM_POINTS,config.MEL_POINTS], name=f"{name}_reshape") 
        
        self.gru = GRU([config.SUBSPECTROGRAM_POINTS,config.MEL_POINTS],name=f"{name}_gru")

        self.dense = Dense(10, activation=tf.nn.sigmoid, name=f"{name}_output")

    def call(self, inputs, training=True):
        net = self.input_layer(inputs)
        net = self.reshape(net)
        net = self.gru(net, training=training)
        net = self.dense(net)
        return net
