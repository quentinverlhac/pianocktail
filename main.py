import tensorflow as tf
import pickle as pkl
import config
import numpy as np
from utils import draw_subspectrogram

def main() :

    # import data
    with open(config.EMOTIFY_SPECTROGRAM_DUMP_PATH,"rb") as f:
        spectrograms = pkl.load(f)

    # creating random labels for testing purpose
    train_labels = np.zeros((config.DEV_MODE_SAMPLE_NUMBER,9))
    for i in range(8):
        category = np.random.randint(0,9)
        train_labels[i,category] = 1

    # building the model
    from models.cnn import ConvModel as Model
    model = Model()
    model.build(input_shape=(1,128,430,1))
    model.summary()
    optimizer = tf.optimizers.Adam(config.LEARNING_RATE)

    #declaring forward pass and gradient descent
    @tf.function
    def forward_pass(inputs,labels):
        print("tracing forward graph")
        predictions = model.call(inputs)
        loss = tf.losses.categorical_crossentropy(
            y_true = labels,
            y_pred = predictions
        )
        return predictions, loss

    @tf.function
    def train_step(inputs,labels):
        print("tracing train graph")
        with tf.GradientTape() as tape:
            predictions, loss = forward_pass(inputs,labels)
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    
    # for test we iterate over samples one by one
    epochs = 3
    for epoch in range(epochs) :
        print("epoch n_",epoch)
        for step in range(config.DEV_MODE_SAMPLE_NUMBER) :
            input_tensor = tf.convert_to_tensor(draw_subspectrogram(spectrograms[step],5))
            layer_input = tf.reshape(input_tensor,(1,128,430,1))
            train_step(layer_input,train_labels[step])

if __name__ == "__main__" :
    main()




