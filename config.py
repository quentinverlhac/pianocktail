import os

# Config variables

# Set this to test code on limited sample number. Useful to test faster.
IS_DEV_MODE = False
DEV_MODE_SAMPLE_NUMBER = 10

# Paths
DIRECTORY_PATH = os.getcwd()
# Data paths
DATA_DIRECTORY_PATH = os.path.join(DIRECTORY_PATH, "data")
# Emotify
EMOTIFY_DATA_PATH = os.path.join(DATA_DIRECTORY_PATH, "emotifymusic")
EMOTIFY_SAMPLES_PATH = os.path.join(EMOTIFY_DATA_PATH, "samples")
EMOTIFY_LABELS_CSV_PATH = os.path.join(EMOTIFY_DATA_PATH, "labels.csv")
EMOTIFY_SAMPLES_DUMP_PATH = os.path.join(EMOTIFY_DATA_PATH, "data.pkl")
EMOTIFY_SPECTROGRAM_DUMP_PATH = os.path.join(EMOTIFY_DATA_PATH, "spectro.pkl")
EMOTIFY_LABELS_DUMP_PATH = os.path.join(EMOTIFY_DATA_PATH, "labels.pkl")
# Emotify split data
SPLIT_DATA_DIRECTORY_PATH = os.path.join(DATA_DIRECTORY_PATH, "split_data")
TRAIN_DATA_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "train.pkl")
TRAIN_LABELS_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "train_labels.pkl")
TEST_DATA_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "test.pkl")
TEST_LABELS_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "test_labels.pkl")
DEV_DATA_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "dev.pkl")
DEV_LABELS_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "dev_labels.pkl")
# Saved models
CHECKPOINTS_PATH = os.path.join(DIRECTORY_PATH, 'checkpoints')
SAVED_MODELS_PATH = os.path.join(DIRECTORY_PATH, 'saved_models')

# The list of ordered emotions.
EMOTIFY_EMOTIONS_ORDERED_LIST = ["amazement", "solemnity", "tenderness", "nostalgia", "calmness", "power",
                                 "joyful_activation", "tension", "sadness"]
NUMBER_OF_EMOTIONS = len(EMOTIFY_EMOTIONS_ORDERED_LIST)

# Sampling and spectrogram variables
SAMPLING_RATE = 44100
FFT_HOP = 512
FFT_RATE = SAMPLING_RATE / FFT_HOP
SUBSPECTROGRAM_DURATION_S = 5
SUBSPECTROGRAM_POINTS = int(SUBSPECTROGRAM_DURATION_S * FFT_RATE)
MEL_POINTS = 128


# Train size (0-1)
TRAIN_SIZE = 0.8

# Training variables
LEARNING_RATE = 10
BATCH_SIZE = 1
NB_EPOCHS = 10

# Manage checkpoints
RESTORE_CHECKPOINT = False
SAVE_PERIOD = 100
