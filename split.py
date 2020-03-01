import argparse

from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
import numpy as np
import pandas as pd
import random

import config
import utils

def split(validate):
    # Create split_data directory if it doesn’t exist
    utils.create_directory_if_doesnt_exist(config.SPLIT_DATA_DIRECTORY_PATH)

    # Load processed data
    all_spectro = utils.load_dump(config.EMOTIFY_SPECTROGRAM_DUMP_PATH)

    index = np.array([[i+1] for i in range(len(all_spectro))])

    labels = utils.load_dump(config.EMOTIFY_LABELS_DUMP_PATH)

    if validate:
        # Train/val+test split, use iterative stratification
        train_x_index, train_y, test_val_x_index, test_val_y = iterative_train_test_split(
            index, np.asarray(labels, dtype=np.int), test_size=(1-config.TRAIN_SIZE))

        # Val/test split, use iterative stratification
        val_x_index, val_y, test_x_index, test_y = iterative_train_test_split(
            test_val_x_index, test_val_y, test_size=(1-config.VAL_SIZE/(1-config.TRAIN_SIZE)))

        val_x = [all_spectro[index[0]-1] for index in val_x_index]
        utils.dump_elements(val_x, config.VAL_DATA_PATH)
        utils.dump_elements(val_y, config.VAL_LABELS_PATH)

    else:
        # Train/test split, use iterative stratification
        train_x_index, train_y, test_x_index, test_y = iterative_train_test_split(
            index, np.asarray(labels, dtype=np.int), test_size=(1 - (config.TRAIN_SIZE+config.VAL_SIZE)))

    test_x = [all_spectro[index[0] - 1] for index in test_x_index]
    train_x = [all_spectro[index[0] - 1] for index in train_x_index]

    print(len(test_x), len(train_x))

    # Shuffle train data and labels
    if config.SEQUENTIAL_TRAINING:
        train_x, train_y = utils.segment_dataset(train_x, train_y, config.SUBSPECTROGRAM_DURATION_S, config.FFT_RATE)

    train_x, train_y = utils.shuffle_data_and_labels(train_x, train_y)

    # Dump split data
    utils.dump_elements(train_x, config.TRAIN_DATA_PATH)
    utils.dump_elements(train_y, config.TRAIN_LABELS_PATH)
    utils.dump_elements(test_x, config.TEST_DATA_PATH)
    utils.dump_elements(test_y, config.TEST_LABELS_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="tuple, creates a validation set if True")
    args = parser.parse_args()
    print(args)
    split(args.validate)
