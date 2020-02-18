import os
# Config variables

# Set this to test code on limited sample number. Useful to test faster.
IS_DEV_MODE = False
DEV_MODE_SAMPLE_NUMBER = 10

# Data paths
DIRECTORY_PATH = os.getcwd()
DATA_DIRECTORY_PATH = os.path.join(DIRECTORY_PATH, "data")
EMOTIFY_DATA_PATH = os.path.join(DATA_DIRECTORY_PATH, "emotifymusic")
EMOTIFY_SAMPLES_PATH = os.path.join(EMOTIFY_DATA_PATH, "samples")
EMOTIFY_LABELS_CSV_PATH = os.path.join(EMOTIFY_DATA_PATH, "labels.csv")
EMOTIFY_SAMPLES_DUMP_PATH = os.path.join(EMOTIFY_DATA_PATH, "data.pkl")
EMOTIFY_SPECTROGRAM_DUMP_PATH = os.path.join(EMOTIFY_DATA_PATH, "spectro.pkl")
EMOTIFY_LABELS_DUMP_PATH = os.path.join(EMOTIFY_DATA_PATH, "labels.pkl")
SPLIT_DATA_DIRECTORY_PATH = os.path.join(DATA_DIRECTORY_PATH, "split_data")
TRAIN_DATA_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "train.pkl")
TRAIN_LABELS_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "train_labels.pkl")
TEST_DATA_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "test.pkl")
TEST_LABELS_PATH = os.path.join(SPLIT_DATA_DIRECTORY_PATH, "test_labels.pkl")

# The list of ordered emotions.
EMOTIFY_EMOTIONS_ORDERED_LIST = ["amazement", "solemnity", "tenderness", "nostalgia", "calmness", "power",
                                 "joyful_activation", "tension", "sadness"]

# Sampling and spectrogram variables
SAMPLING_RATE = 44100
FFT_HOP = 512
