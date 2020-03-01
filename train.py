import tensorflow as tf
import argparse

import config
from models.pianocktail_cnn import PianocktailCNN
from models.pianocktail_gru import PianocktailGRU
import utils


def train(model_name=config.MODEL.value):
    # import data and labels
    train_spectrograms = utils.load_dump(config.DEV_DATA_PATH if config.IS_DEV_MODE else config.TRAIN_DATA_PATH)
    train_labels = utils.load_dump(config.DEV_LABELS_PATH if config.IS_DEV_MODE else config.TRAIN_LABELS_PATH)

    val_spectrograms = utils.load_dump(config.VAL_DATA_PATH)
    val_labels = utils.load_dump(config.VAL_LABELS_PATH)

    # generate and batch datasets
    def generate_subspectrogram(duration_s=config.SUBSPECTROGRAM_DURATION_S, fft_rate=config.FFT_RATE):
        spectrograms = train_spectrograms
        labels_ = train_labels

        for i in range(len(labels_)):
            sub_spectro = utils.draw_subspectrogram(spectrograms[i], duration_s, fft_rate,
                                                    random_pick=config.RANDOM_PICK)
            tensor_spectro = tf.convert_to_tensor(sub_spectro)
            tensor_spectro = tf.transpose(tensor_spectro)
            tensor_label = tf.convert_to_tensor(labels_[i])
            yield tensor_spectro, tensor_label

    if config.SEQUENTIAL_TRAINING:
        train_spectrograms = [spectrogram.T for spectrogram in train_spectrograms]
        train_spectrograms = tf.convert_to_tensor(train_spectrograms)
        train_labels = tf.convert_to_tensor(train_labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_spectrograms, train_labels))
    else:
        train_dataset = tf.data.Dataset.from_generator(generate_subspectrogram, (tf.float32, tf.float32))
    train_dataset = train_dataset.batch(config.BATCH_SIZE)

    # building the model
    model = utils.initialize_model(model_name)
    model.build(input_shape=(config.BATCH_SIZE, config.SUBSPECTROGRAM_POINTS, config.MEL_BINS))
    model.summary()
    optimizer = tf.optimizers.Adam(config.LEARNING_RATE)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    # Val metrics
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
    val_accuracies = []
    val_losses = []

    checkpoint, checkpoint_manager = utils.setup_checkpoints(model, optimizer)

    # restore checkpoint
    if config.RESTORE_CHECKPOINT:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Restored checkpoint. Model {checkpoint.model.name} - epoch {checkpoint.epoch.value()}")

    loss_history = []
    epoch_range = range(checkpoint.epoch.value(), config.NB_EPOCHS)

    # for test we iterate over samples one by one
    for epoch in epoch_range:

        for iteration, (spectro, label) in enumerate(train_dataset):
            predictions, labels = train_step(spectro, label, model, optimizer, train_loss, train_accuracy)

        # display metrics
        loss_history.append(train_loss.result())
        utils.display_and_reset_metrics(train_loss, train_accuracy, predictions, labels, epoch=epoch)

        # save checkpoint
        checkpoint.epoch.assign_add(1)
        checkpoint_manager.save()

        if epoch % config.VALIDATION_EPOCH_GAP == 0 or epoch == config.NB_EPOCHS - 1:
            print("======================== evaluation on validation data =========================")
            # test model on validation set
            this_val_accuracy, this_val_loss = utils.test_model(
                model, val_spectrograms, val_labels, val_loss, val_accuracy, epoch=epoch)
            val_accuracies.append(this_val_accuracy)
            val_losses.append(this_val_loss)
            print("============================ returning to training =============================")

    utils.save_model(model, epoch)
    utils.save_and_display_metric_through_epochs(epoch_range, loss_history, model.name, "training loss")

    # Plot val accuracies
    val_range = [i for i in epoch_range if i % config.VALIDATION_EPOCH_GAP == 0 or i == config.NB_EPOCHS - 1]
    utils.save_and_display_metric_through_epochs(val_range, val_accuracies, model.name, "validation accuracy")
    utils.save_and_display_metric_through_epochs(val_range, val_losses, model.name, "validation loss")
    best_epoch = (len(val_accuracies) - list(reversed(val_accuracies)).index(
        min(val_accuracies)) - 1) * config.VALIDATION_EPOCH_GAP
    print(f"Best validation loss was epoch {best_epoch}")


# declaring forward pass and gradient descent
@tf.function
def forward_pass(inputs, labels, model):
    print("tracing forward graph")
    predictions = model.call(inputs, training=True)
    if config.LABEL_ENCODING == config.LabelEncodingEnum.MAJORITY:
        loss_func = tf.keras.categorical_crossentropy
    if config.LABEL_ENCODING == config.LabelEncodingEnum.THRESHOLD:
        loss_func = tf.keras.losses.binary_crossentropy
    loss = loss_func(
        y_true=labels,
        y_pred=predictions
    )
    return predictions, loss


@tf.function
def train_step(inputs, labels, model, optimizer, train_loss, train_accuracy):
    print("tracing train graph")
    with tf.GradientTape() as tape:
        predictions_, loss = forward_pass(inputs, labels, model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # compute metrics
    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions_)

    return predictions_, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="name of the model to train",
                        choices=[model.value for model in config.ModelEnum])
    args = parser.parse_args()
    train(args.model_name)
