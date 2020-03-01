import os
import pickle as pkl
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import config
from models.pianocktail_cnn import PianocktailCNN
from models.pianocktail_gru import PianocktailGRU


def create_directory_if_doesnt_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def draw_subspectrogram(spectrogram, duration_s, fft_rate, random_pick=False):
    """
    Draw a random subspectrogram of given time length from the given spectrogram
    """
    if not random_pick: np.random.seed(42)
    offset = int(np.random.random() * (spectrogram.shape[1] - duration_s * fft_rate))
    return spectrogram[:, offset:offset + int(duration_s * fft_rate)]


def segment_spectrogram(spectrogram, duration_s, fft_rate):
    """
    Segment the spectrogram into successive subspectrograms of given length and returns
    the list of all subspectrograms
    """
    spectrograms = []
    sub_len = int(duration_s*fft_rate)
    n_subspectros = int(spectrogram.shape[1]/sub_len)
    for i in range(n_subspectros):
        spectrograms.append(spectrogram[:,sub_len*i:sub_len*(i+1)])
    return spectrograms

def segment_dataset(all_spectrograms, labels, duration_s, fft_rate):
    """
    Segment all spectrograms in the dataset in snippets of the given duration, and update
    the labels accordingly
    """
    new_spectrograms = []
    new_labels = []
    for i in range(len(all_spectrograms)):
        segments = segment_spectrogram(all_spectrograms[i], duration_s, fft_rate)
        new_spectrograms += segments
        new_labels += [labels[i] for spectro in segments]
    return new_spectrograms, new_labels
    

def dump_elements(elements, dump_path):
    """
    Dumps all elements in the list to a binary file on the dump path
    """
    with open(dump_path, 'wb') as f:
        pkl.dump(elements, f)


def load_dump(dump_path):
    """
    Load the content of the file at dump path
    """
    with open(dump_path, "rb") as file:
        return pkl.load(file)


def load_labels(path):
    """
    Load a pickle file containing list of labels and return as pandas DataFrame
    """
    with open(path, "rb") as f:
        labels = pkl.load(f)

    # Transform labels to pandas DataFrame
    labels = pd.DataFrame(labels)
    labels.columns = config.EMOTIFY_EMOTIONS_ORDERED_LIST
    return labels


def initialize_model(model_name):
    if model_name == config.ModelEnum.PIANOCKTAIL_CNN.value:
        return PianocktailCNN()
    elif model_name == config.ModelEnum.PIANOCKTAIL_GRU.value:
        return PianocktailGRU()
    else:
        raise Exception(f"The name of the saved model doesn't match any model type. It should be one of the following: {[model.value for model in config.ModelEnum]}")


def get_save_file_name(model_name, epoch):
    dev_mode_string = "_dev" if config.IS_DEV_MODE else ""
    return f"{model_name}_{epoch:06d}{dev_mode_string}"


def save_model(model, epoch):
    create_directory_if_doesnt_exist(config.SAVED_MODELS_PATH)
    save_path = os.path.join(config.SAVED_MODELS_PATH, get_save_file_name(model.name, epoch) + ".h5")
    model.save_weights(save_path)
    print(f"Saved model {model.name} at {save_path}")


def load_model(file_path, batch_size=1):
    model_name = os.path.split(file_path)[-1].split("_")[0]
    model = initialize_model(model_name)
    model.build(input_shape=(batch_size, config.SUBSPECTROGRAM_POINTS, config.MEL_BINS))
    model.load_weights(file_path)
    return model


def setup_checkpoints(model, optimizer):
    create_directory_if_doesnt_exist(config.CHECKPOINTS_PATH)
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0),
        model=model,
        optimizer=optimizer
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        config.CHECKPOINTS_PATH,
        max_to_keep=1
    )
    return checkpoint, checkpoint_manager


def normalise_by_max(spectrogram):
    return spectrogram / np.max(np.abs(spectrogram))


# display_and_reset_metrics is displaying loss and accuracy values with a nice format. It resets the metrics once it is
# done.
# is_test allows to display the final performances of the model.
def display_and_reset_metrics(loss, accuracy, predictions, labels, epoch=None, is_test=False):
    epoch_header = 'Results on test dataset' if is_test else f'epoch {epoch}'
    template = '{} - loss: {:4.2f} - accuracy: {:5.2%}'
    print(template.format(epoch_header, loss.result(), accuracy.result()))
    if config.IS_VERBOSE and not is_test:
        emotion_template = 'Emotion category: {:>17} - prediction: {:10f} - label: {:10f} - difference: {:10f}'
        for i in range(len(config.EMOTIFY_EMOTIONS_ORDERED_LIST)):
            prediction = (predictions.numpy())[0][i]
            label = (labels.numpy())[0][i]
            print(emotion_template.format(config.EMOTIFY_EMOTIONS_ORDERED_LIST[i], prediction, label, prediction - label))
    loss.reset_states()
    accuracy.reset_states()


def test_model(model, data, labels, test_loss, test_accuracy, epoch=None, is_test=False):
    if is_test:
        per_emotion_score = {}
        for emotion in config.EMOTIFY_EMOTIONS_ORDERED_LIST:
            per_emotion_score[emotion] = tf.keras.metrics.BinaryAccuracy(name=f'{emotion}_accuracy')
    for i, label in enumerate(labels):
        predictions = average_predictions(data[i], model)
        if config.IS_VERBOSE:
            print(f'Labels: {label} - predictions: {predictions}')
        # compute metrics
        loss = tf.keras.metrics.binary_crossentropy(y_pred=predictions, y_true=label)
        test_loss.update_state(loss)
        test_accuracy.update_state(label, predictions)
        if is_test:
            for i in range(len(config.EMOTIFY_EMOTIONS_ORDERED_LIST)):
                per_emotion_score[config.EMOTIFY_EMOTIONS_ORDERED_LIST[i]].update_state([label[i]], [predictions.numpy()[0][i]])
    this_test_loss = test_loss.result()
    this_test_accuracy = test_accuracy.result()
    display_and_reset_metrics(test_loss, test_accuracy, predictions, label, epoch=epoch, is_test=is_test)
    if is_test:
        print("Accuracy per emotion on whole test set:")
        for emotion in config.EMOTIFY_EMOTIONS_ORDERED_LIST:
                print("{}: {:5.2%}".format(emotion, per_emotion_score[emotion].result()))
    return this_test_accuracy, this_test_loss


def average_predictions(full_spectrogram, model):
    """
    Average predictions of the model over all segments in the spectrogram
    """
    segmented_spectro = segment_spectrogram(full_spectrogram, config.SUBSPECTROGRAM_DURATION_S, config.FFT_RATE)
    all_predictions = []
    for spectro in segmented_spectro:
        tensor_spectro = tf.convert_to_tensor(spectro)
        tensor_spectro = tf.transpose(tensor_spectro)
        tensor_spectro = tf.reshape(tensor_spectro,[1,tensor_spectro.shape[0],tensor_spectro.shape[1]])
        all_predictions.append(model.call(tensor_spectro))
    return tf.add_n(all_predictions)/len(segmented_spectro)


def save_and_display_metric_through_epochs(epoch_range, loss, model_name, y_label):
    create_directory_if_doesnt_exist(config.SAVED_LOSS_GRAPHS_PATH)
    file_name = get_save_file_name(model_name, epoch_range[-1]) + ".png"
    plt.plot(epoch_range, loss)
    plt.xlabel('epochs')
    plt.ylabel(f'{y_label}')
    plt.title(f'Evolution of {y_label} through epochs')
    plt.savefig(os.path.join(config.SAVED_LOSS_GRAPHS_PATH, file_name))
    plt.show()


def shuffle_data_and_labels(data, labels):
    """
    Shuffles data and labels the same way
    :param data: list: input data
    :param labels: list: corresponding labels
    :return: tuple, shuffled data and labels
    """
    temp = list(zip(data, labels))
    random.shuffle(temp)
    return zip(*temp)
