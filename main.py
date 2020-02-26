import tensorflow as tf
from enum import Enum

import config
from utils import draw_subspectrogram, load_dump, save_model, setup_checkpoints
from models.basic_cnn import BasicCNN
from models.pianocktail_gru import PianocktailGRU


def main():
    class DatasetEnum(Enum):
        TRAIN = "TRAIN"
        TEST = "TEST"
    
    # import data
    train_spectrograms = load_dump(config.DEV_DATA_PATH if config.IS_DEV_MODE else config.TRAIN_DATA_PATH)
    test_spectrograms = load_dump(config.TEST_DATA_PATH)

    # import labels
    train_labels = load_dump(config.DEV_LABELS_PATH if config.IS_DEV_MODE else config.TRAIN_LABELS_PATH)
    test_labels = load_dump(config.TEST_LABELS_PATH)

    # generate datasets
    def generate_subspectrogram(use_test_dataset = False, duration_s = config.SUBSPECTROGRAM_DURATION_S, fft_rate = config.FFT_RATE, mel_bins = config.MEL_BINS):
        spectrograms = train_spectrograms
        labels = train_labels
        if use_test_dataset:
            spectrograms = test_spectrograms
            labels = test_labels

        for i in range(len(labels)):
            sub_spectro = draw_subspectrogram(spectrograms[i], duration_s, fft_rate, random_pick=config.RANDOM_PICK)
            tensor_spectro = tf.convert_to_tensor(sub_spectro)
            tensor_spectro = tf.transpose(tensor_spectro)
            tensor_label = tf.convert_to_tensor(labels[i])
            yield tensor_spectro, tensor_label
    
    def generate_randomized_batched_dataset(use_test_dataset = False):
        dataset = tf.data.Dataset.from_generator(generate_subspectrogram, (tf.float32, tf.float32), args=[use_test_dataset])
        return dataset.batch(config.BATCH_SIZE)

    train_dataset = generate_randomized_batched_dataset()
    test_dataset = generate_randomized_batched_dataset(use_test_dataset=True)

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
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    # display_and_reset_metrics is displaying loss and accuracy values with a nice format. It resets the metrics once it is done.
    # is_test allows to display the final performances of the model.
    def display_and_reset_metrics(loss, accuracy, predictions, labels, iteration = None, is_test = False):
        iteration_header = 'Results on test dataset' if is_test else f'iteration {iteration}'
        template = '{} - loss: {:4.2f} - accuracy: {:5.2%}'
        print(template.format(iteration_header, loss.result(), accuracy.result()))
        if config.IS_VERBOSE and not is_test:
            emotion_template = 'Emotion category: {:>17} - prediction: {:10f} - label: {:10f} - difference: {:10f}'
            for i in range(len(config.EMOTIFY_EMOTIONS_ORDERED_LIST)):
                prediction = (predictions.numpy())[0][i]
                label = (labels.numpy())[0][i]
                print(emotion_template.format(config.EMOTIFY_EMOTIONS_ORDERED_LIST[i], prediction, label, prediction - label))
        loss.reset_states()
        accuracy.reset_states()

    checkpoint, checkpoint_manager = setup_checkpoints(model, optimizer)

    # declaring forward pass and gradient descent
    @tf.function
    def forward_pass(inputs, labels):
        print("tracing forward graph")
        predictions = model.call(inputs)
        loss = tf.keras.losses.binary_crossentropy(
            y_true = labels, 
            y_pred = predictions
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
        train_accuracy.update_state(labels, predictions)

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
            if iteration % 10 == 0:
                display_and_reset_metrics(train_loss, train_accuracy, predictions, labels, iteration=iteration)

            # manage checkpoint
            checkpoint.iteration.assign(iteration)
            if iteration % config.SAVE_PERIOD == 0:
                checkpoint_manager.save()

    save_model(model, epoch)

    # ============================ test the model ==============================

    print(f"======================== evaluation on test data =========================")
    for iteration, (spectro, label) in enumerate(test_dataset):
        predictions, loss = forward_pass(spectro, label)

        # compute metrics
        test_loss.update_state(loss)
        test_accuracy.update_state(label, predictions)

    display_and_reset_metrics(test_loss, test_accuracy, predictions, label, is_test=True)

if __name__ == "__main__":
    main()
