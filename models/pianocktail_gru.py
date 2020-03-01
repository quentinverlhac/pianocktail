import tensorflow as tf 

from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, GRU, GRUCell, Dense, InputLayer, GlobalAveragePooling1D, Dropout

import config


class PianocktailGRU(Model):
    def __init__(self, name=config.ModelEnum.PIANOCKTAIL_GRU.value, subspectrogram_points = config.SUBSPECTROGRAM_POINTS, mel_bins = config.MEL_BINS, number_of_emotions = config.NUMBER_OF_EMOTIONS, **kwargs):
        super(PianocktailGRU, self).__init__(name=name, **kwargs)

        self.input_layer = InputLayer((subspectrogram_points, mel_bins), name=f"{name}_input")
        
        self.gru1 = GRU(100, name=f"{name}_gru1", return_sequences=True)

        self.dropout1 = Dropout(0.3, name=f"{name}_dropout1")

        self.gru2 = GRU(50,name=f"{name}_gru2", return_sequences=True)

        self.dropout2 = Dropout(0.3, name=f"{name}_dropout2")

        self.pool = GlobalAveragePooling1D(name=f"{name}_average")

        self.dropout3 = Dropout(0.3, name=f"{name}_dropout3")

        self.dense = Dense(number_of_emotions, activation=tf.nn.sigmoid, name=f"{name}_output")

    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.gru1(net, training=training)
        net = self.dropout1(net, training=training)
        net = self.gru2(net, training=training)
        net = self.dropout2(net, training=training)
        net = self.pool(net)
        net = self.dropout3(net, training=training)
        net = self.dense(net)
        return net
