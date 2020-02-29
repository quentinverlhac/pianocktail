from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
import numpy as np
import pandas as pd
import random

import config
import utils

# Create split_data directory if it doesn’t exist
utils.create_directory_if_doesnt_exist(config.SPLIT_DATA_DIRECTORY_PATH)

# Load processed data
all_spectro = utils.load_dump(config.EMOTIFY_SPECTROGRAM_DUMP_PATH)

index = np.array([[i+1] for i in range(len(all_spectro))])

labels = utils.load_dump(config.EMOTIFY_LABELS_DUMP_PATH)

# Train/test split 80/20, stratify on highest ranked emotion
train_x_index, train_y, test_x_index, test_y = iterative_train_test_split(
    index, np.asarray(labels, dtype=np.int), test_size=(1-config.TRAIN_SIZE))

train_x = [all_spectro[index[0]-1] for index in train_x_index]
test_x = [all_spectro[index[0]-1] for index in test_x_index]

# Shuffle train data and labels
temp = list(zip(train_x, train_y))
random.shuffle(temp)
train_x, train_y = zip(*temp)

if config.SEQUENTIAL_TRAINING:
    train_x, train_y = utils.segment_dataset(train_x, train_y, config.SUBSPECTROGRAM_DURATION_S, config.FFT_RATE)

# Dump split data
utils.dump_elements(train_x, config.TRAIN_DATA_PATH)
utils.dump_elements(train_y, config.TRAIN_LABELS_PATH)
utils.dump_elements(test_x, config.TEST_DATA_PATH)
utils.dump_elements(test_y, config.TEST_LABELS_PATH)
