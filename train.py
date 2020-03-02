import argparse

import numpy as np
import pandas as pd
from scipy.special import softmax
import tensorflow as tf

import config
from models.pianocktail_cnn import PianocktailCNN
from models.pianocktail_gru import PianocktailGRU
import utils


def train(model_name=config.MODEL.value, epochs=config.NB_EPOCHS, validate=False, balance=False):
    # import data and labels
    train_spectrograms = utils.load_dump(config.DEV_DATA_PATH if config.IS_DEV_MODE else config.TRAIN_DATA_PATH)
    train_labels = utils.load_dump(config.DEV_LABELS_PATH if config.IS_DEV_MODE else config.TRAIN_LABELS_PATH)

    if balance:
        # Balance train dataset
        train_spectrograms, train_labels = balance_dataset(train_spectrograms, train_labels)

    if validate:
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

    if validate:
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
    epoch_range = range(checkpoint.epoch.value(), epochs)

    # for test we iterate over samples one by one
    for epoch in epoch_range:

        for iteration, (spectro, label) in enumerate(train_dataset):
            predictions, labels = train_step(spectro, label, model, optimizer, train_loss, train_accuracy)

        # display metrics
        loss_history.append(train_loss.result())
        utils.display_and_reset_metrics(train_loss, train_accuracy, predictions, labels, epoch=epoch)

        if epoch % config.VALIDATION_EPOCH_GAP == 0 or epoch == epochs - 1:
            # save checkpoint
            checkpoint_manager.save()
            checkpoint.epoch.assign_add(10)

        if (epoch % config.VALIDATION_EPOCH_GAP == 0 or epoch == epochs - 1) and validate:
            print("======================== evaluation on validation data =========================")
            # test model on validation set
            this_val_accuracy, this_val_loss = utils.test_model(
                model, val_spectrograms, val_labels, val_loss, val_accuracy, epoch=epoch)
            val_accuracies.append(this_val_accuracy)
            val_losses.append(this_val_loss)
            print("============================ returning to training =============================")

    utils.save_model(model, epoch)
    utils.save_and_display_metric_through_epochs(epoch_range, loss_history, model.name, "training loss")

    if validate:
        # Plot val accuracies
        val_range = [i for i in epoch_range if i % config.VALIDATION_EPOCH_GAP == 0 or i == epochs - 1]
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


def balance_dataset(data, labels):
    """
    Balance dataset by duplicating interesting examples, uses some randomness
    :param data: pandas DataFrame or array-like, data before balancing
    :param labels: pandas DataFrame or array-like, labels before balancing
    :return: tuple, list of indexes to use from original data (will have some duplicate indexes) and duplicated labels
    """
    labels = pd.DataFrame(labels)
    data = pd.DataFrame(data)
    labels["index"] = list(labels.index)
    iteration = 0
    print("========== label proportions before balancing (major class / minor class) ==========")
    print(_get_proportions(labels.drop("index", axis=1)))
    while iteration < 200:
        id_ = _get_proportions(labels.drop("index", axis=1)).idxmax()
        minor_class = labels.iloc[:, id_].value_counts().idxmin()
        duplicable_labels = labels.loc[:, id_] == minor_class
        temp_df = labels[duplicable_labels].drop_duplicates(subset=["index"]).drop("index", axis=1)
        scores = temp_df.apply(_score_sample, axis=1, args=[labels.drop("index", axis=1)])
        duplicate_probability = softmax(scores)
        chosen_id = np.random.choice(list(duplicate_probability.index), p=list(duplicate_probability.values))
        labels = labels.append(labels.iloc[chosen_id, :], ignore_index=True)
        data = data.append(data.iloc[chosen_id, :], ignore_index=True)
        iteration += 1
    print("========== label proportions before balancing (major class / minor class) ==========")
    print(_get_proportions(labels.drop("index", axis=1)))
    print("========== value counts of most duplicated examples after balancing ==========")
    print(labels["index"].value_counts()[:10])
    return [spectrogram[0] for spectrogram in data.to_numpy()], labels.drop("index", axis=1).to_numpy()


def _get_proportions(labels):
    value_counts = labels.apply(lambda x: x.value_counts())
    proportions = value_counts.max() / value_counts.min()
    return proportions


def _score_sample(sample, labels):
    major_classes = labels.apply(lambda x: x.value_counts().idxmax())
    return (~(sample == major_classes)).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="name of the model to train",
                        choices=[model.value for model in config.ModelEnum])
    parser.add_argument("--balance", action="store_true", help="balance train dataset")
    parser.add_argument("--epochs", dest="epochs", help="number of epochs", type=int, required=False, default=config.NB_EPOCHS)
    parser.add_argument("--validate", action="store_true", help="tests of validation set if True")
    args = parser.parse_args()
    train(args.model_name, args.epochs, args.validate, args.balance)
