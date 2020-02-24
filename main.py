import tensorflow as tf

import config
from utils import draw_subspectrogram, load_dump, save_model, setup_checkpoints
from models.basic_cnn import BasicCNN
from models.pianocktail_gru import PianocktailGRU


def main():
    # import data
    train_spectrograms = load_dump(config.DEV_DATA_PATH if config.IS_DEV_MODE else config.TRAIN_DATA_PATH)

    # import labels
    train_labels = load_dump(config.DEV_LABELS_PATH if config.IS_DEV_MODE else config.TRAIN_LABELS_PATH)

    # generate dataset
    def generate_subspectrogram(duration_s = config.SUBSPECTROGRAM_DURATION_S, fft_rate = config.FFT_RATE, mel_bins = config.MEL_BINS):
        for i in range(len(train_labels)):
            sub_spectro = draw_subspectrogram(train_spectrograms[i], duration_s, fft_rate)
            tensor_spectro = tf.convert_to_tensor(sub_spectro)
            tensor_spectro = tf.transpose(tensor_spectro)
            tensor_label = tf.convert_to_tensor(train_labels[i])
            yield tensor_spectro, tensor_label

    train_dataset = tf.data.Dataset.from_generator(generate_subspectrogram, (tf.float32, tf.float32))
    train_dataset = train_dataset.batch(config.BATCH_SIZE)

    # building the model
    if config.MODEL == config.ModelEnum.PIANOCKTAIL_GRU:
        Model = PianocktailGRU
    elif config.MODEL == config.ModelEnum.BASIC_CNN:
        Model = BasicCNN
    model = Model()
    model.build(input_shape=(config.BATCH_SIZE, config.SUBSPECTROGRAM_POINTS, config.MEL_BINS))
    model.summary()
    optimizer = tf.optimizers.Adam(config.LEARNING_RATE)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    checkpoint, checkpoint_manager = setup_checkpoints(model, optimizer)

    # declaring forward pass and gradient descent
    @tf.function
    def forward_pass(inputs, labels):
        print("tracing forward graph")
        predictions = model.call(inputs)
        loss = tf.losses.categorical_crossentropy(
            y_true=labels,
            y_pred=predictions
        )
        return predictions, loss

    @tf.function
    def train_step(inputs, labels):
        print("tracing train graph")
        with tf.GradientTape() as tape:
            predictions, loss = forward_pass(inputs, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # compute metrics
        train_loss.update_state(loss)
        train_accuracy.update_state(predictions, labels)

        return predictions, labels

    # ============================ train the model =============================
    # restore checkpoint
    if config.RESTORE_CHECKPOINT:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(
            f"Restored checkpoint. Model {checkpoint.model.name} - epoch {checkpoint.epoch.value()} - iteration {checkpoint.iteration.value()}")
    # for test we iterate over samples one by one
    for epoch in range(config.NB_EPOCHS):

        print(f"============================ epoch {epoch} =============================")
        checkpoint.epoch.assign(epoch)

        for iteration, (spectro, label) in enumerate(train_dataset):
            predictions, labels = train_step(spectro, label)

            # display metrics
            if iteration % config.ITERATION_PRINT_PERIOD == 0:
                template = 'iteration {} - loss: {:4.2f} - accuracy: {:5.2%}'
                print(template.format(iteration, train_loss.result(), train_accuracy.result()))
                if config.IS_VERBOSE:
                    emotion_template = 'Emotion category: {:>17} - prediction: {:10f} - label: {:10f} - difference: {:10f}'
                    for i in range(len(config.EMOTIFY_EMOTIONS_ORDERED_LIST)):
                        prediction = (predictions.numpy())[0][i]
                        label = (labels.numpy())[0][i]
                        print(emotion_template.format(config.EMOTIFY_EMOTIONS_ORDERED_LIST[i], prediction, label, prediction - label))
                train_loss.reset_states()
                train_accuracy.reset_states()

            # manage checkpoint
            checkpoint.iteration.assign(iteration)
            if iteration % config.SAVE_PERIOD == 0:
                checkpoint_manager.save()

    save_model(model, epoch)


if __name__ == "__main__":
    main()
