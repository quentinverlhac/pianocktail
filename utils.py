import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config


def create_directory_if_doesnt_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def draw_subspectrogram(spectrogram):
    """
    Draw a random subspectrogram of given time length from the given spectrogram
    """
    offset = int(np.random.random() * (spectrogram.shape[1] - config.SUBSPECTROGRAM_DURATION_S * config.FFT_RATE))
    return spectrogram[:, offset:offset + config.SUBSPECTROGRAM_POINTS]


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


def save_model(model, epoch):
    create_directory_if_doesnt_exist(config.SAVED_MODELS_PATH)
    dev_mode_string = "_dev" if config.IS_DEV_MODE else ""
    model.save_weights(os.path.join(
        config.SAVED_MODELS_PATH, f"{model.name}_{epoch:06d}{dev_mode_string}.h5"))


def setup_checkpoints(model, optimizer):
    create_directory_if_doesnt_exist(config.CHECKPOINTS_PATH)
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0),
        iteration=tf.Variable(0),
        model=model,
        optimizer=optimizer
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        config.CHECKPOINTS_PATH,
        max_to_keep=1
    )
    return checkpoint, checkpoint_manager
