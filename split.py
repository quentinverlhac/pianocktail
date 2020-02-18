from sklearn.model_selection import train_test_split
import pickle as pkl
import pandas as pd
from pathlib import Path

import config
import utils

# Create split_data directory if it doesn’t exist
Path(config.SPLIT_DATA_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)

# Load processed data
with open(config.EMOTIFY_SPECTROGRAM_DUMP_PATH, "rb") as f:
    all_spectro = pkl.load(f)

with open(config.EMOTIFY_LABELS_DUMP_PATH, "rb") as f:
    labels = pkl.load(f)

# Transform labels to pandas DataFrame (mainly to use idxmax)
labels = pd.DataFrame(labels)
labels.columns = config.EMOTIFY_EMOTIONS_ORDERED_LIST

# Train/test split 80/20, stratify on highest ranked emotion
train_x, test_x, train_df_y, test_df_y = train_test_split(
    all_spectro, labels, train_size=0.8,  stratify=labels.idxmax(axis=1))

# Dump split data
utils.dump_elements(train_x, config.TRAIN_DATA_PATH)
train_df_y.to_pickle(config.TRAIN_LABELS_PATH)
utils.dump_elements(test_x, config.TEST_DATA_PATH)
test_df_y.to_pickle(config.TEST_LABELS_PATH)
