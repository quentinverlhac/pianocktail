from sklearn.model_selection import train_test_split

import config
import utils

# Create split_data directory if it doesn’t exist
utils.create_directory_if_doesnt_exist(config.SPLIT_DATA_DIRECTORY_PATH)

# Load processed data
all_spectro = utils.load_dump(config.EMOTIFY_SPECTROGRAM_DUMP_PATH)

labels = utils.load_labels(config.EMOTIFY_LABELS_DUMP_PATH)

# Train/test split 80/20, stratify on highest ranked emotion
train_x, test_x, train_df_y, test_df_y = train_test_split(
    all_spectro, labels, train_size=config.TRAIN_SIZE,  stratify=labels.idxmax(axis=1), random_state=42)

# Transform back to list
train_y = list(train_df_y.values)
test_y = list(test_df_y.values)

# Dump split data
utils.dump_elements(train_x, config.TRAIN_DATA_PATH)
utils.dump_elements(train_y, config.TRAIN_LABELS_PATH)
utils.dump_elements(test_x, config.TEST_DATA_PATH)
utils.dump_elements(test_y, config.TEST_LABELS_PATH)
