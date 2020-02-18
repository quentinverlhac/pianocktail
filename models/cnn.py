from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, AveragePooling2D

class ConvModel(Model):
    def __init__(self,name="conv_basic"):
        super(ConvModel,self).__init__()

        self.input_layer = InputLayer((128,430,1), name=f"{name}_input")

        self.conv1 = Conv2D(filters=6,kernel_size=(5,7),activation=tf.nn.relu,name=f"{name}_conv1")

        self.pooling1 = AveragePooling2D(pool_size=2,name=f"{name}_pool1")

        self.conv2 = Conv2D(filters=16,kernel_size=5,activation=tf.nn.relu,name=f"{name}_conv2")

        self.pooling2 = AveragePooling2D(pool_size=2,name=f"{name}_pool2")

        self.flatten = Flatten(name=f"{name}_flatten")

        self.dense1 = Dense(200,activation=tf.nn.relu,name=f"{name}_dense1")

        self.output = Dense(9,activation=tf.nn.softmax,name=f"{name}_output")

    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.conv1(net)
        net = self.pooling1(net)
        net = self.conv2(net)
        net = self.pooling2(net)
        net = self.flatten(net)
        net = self.dense1(net)
        return self.output(net)



