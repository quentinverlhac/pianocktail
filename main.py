import tensorflow as tf
import pickle as pkl
import config
import numpy as np
from utils import draw_subspectrogram, load_dump

def main() :

    # import data
    spectrograms = load_dump(config.EMOTIFY_SPECTROGRAM_DUMP_PATH)

    # import labels
    train_labels = load_dump(config.EMOTIFY_LABELS_DUMP_PATH)

    # generate dataset
    def generate_subspectrogram():
        for i in range(len(train_labels)) :
            sub_spectro = draw_subspectrogram(spectrograms[i],5)
            tensor_spectro = tf.convert_to_tensor(sub_spectro)
            tensor_spectro = tf.reshape(tensor_spectro,(128,430,1))
            tensor_label = tf.convert_to_tensor(train_labels[i])
            yield tensor_spectro, tensor_label

    train_dataset = tf.data.Dataset.from_generator(generate_subspectrogram,(tf.float32,tf.float32))
    train_dataset = train_dataset.batch(config.BATCH_SIZE)

    # building the model
    from models.cnn import ConvModel as Model
    model = Model()
    model.build(input_shape=(config.BATCH_SIZE,128,430,1))
    model.summary()
    optimizer = tf.optimizers.Adam(config.LEARNING_RATE)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

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

        # compute metrics
        train_loss.update_state(loss)
        train_accuracy.update_state(predictions, labels)
    
    # for test we iterate over samples one by one
    for epoch in range(config.NB_EPOCHS) :

        for iteration, (spectro, label) in enumerate(train_dataset) :
            train_step(spectro,label)

            # display metrics
            if iteration % 10 == 0:
                template = 'iteration {} - loss: {:4.2f} - accuracy: {:5.2%}'
                print(template.format(iteration, train_loss.result(), train_accuracy.result()))
                train_loss.reset_states()
                train_accuracy.reset_states()

        

if __name__ == "__main__" :
    main()




