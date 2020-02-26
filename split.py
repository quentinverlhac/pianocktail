from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
import numpy as np
import pandas as pd

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

df_x = pd.DataFrame(all_spectro)
train_x = list(df_x.iloc[list(map(lambda x: int(x[0]-1), train_x_index)), :].apply(lambda x: x[0], axis=1).values)
test_x = list(df_x.iloc[list(map(lambda x: int(x[0]-1), test_x_index)), :].apply(lambda x: x[0], axis=1).values)


# Dump split data
utils.dump_elements(train_x, config.TRAIN_DATA_PATH)
utils.dump_elements(train_y, config.TRAIN_LABELS_PATH)
utils.dump_elements(test_x, config.TEST_DATA_PATH)
utils.dump_elements(test_y, config.TEST_LABELS_PATH)
