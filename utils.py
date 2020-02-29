import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from models.basic_cnn import BasicCNN
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
    if model_name == config.ModelEnum.BASIC_CNN.value:
        return BasicCNN()
    elif model_name == config.ModelEnum.PIANOCKTAIL_GRU.value:
        return PianocktailGRU()
    else:
        raise Exception(f"The name of the saved model doesn't match any model type. It should be one of the following: {[model.value for model in config.ModelEnum]}")

def save_model(model, epoch):
    create_directory_if_doesnt_exist(config.SAVED_MODELS_PATH)
    dev_mode_string = "_dev" if config.IS_DEV_MODE else ""
    save_path = os.path.join(config.SAVED_MODELS_PATH, f"{model.name}_{epoch:06d}{dev_mode_string}.h5")
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
