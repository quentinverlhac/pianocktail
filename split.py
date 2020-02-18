from sklearn.model_selection import train_test_split
import pickle as pkl
from pathlib import Path

import config


Path(config.SPLIT_DATA_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)

with open(config.EMOTIFY_SPECTROGRAM_PATH, "rb") as f:
    all_spectro = pkl.load(f)


train_df_x, test_df_x, train_df_y, test_df_y = train_test_split(
    df.drop("fake_email", axis=1), df.fake_email, train_size=0.9,  stratify=df.fake_email)
train_df_x, validation_df_x, train_df_y, validation_df_y = train_test_split(
    train_df_x, train_df_y, train_size=0.83333, stratify=train_df_y)

train_df = train_df_x.join(train_df_y)
validation_df = validation_df_x.join(validation_df_y)
test_df = test_df_x.join(test_df_y)

train_df.to_csv(config.TRAIN_DATA_OUTPUT_PATH, index=False)
validation_df.to_csv(config.VALIDATION_DATA_OUTPUT_PATH, index=False)
test_df.to_csv(config.TEST_DATA_OUTPUT_PATH, index=False)
