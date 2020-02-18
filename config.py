import os
# Config variables

# Set this to test code on limited sample number. Useful to test faster.
IS_DEV_MODE = True
DEV_MODE_SAMPLE_NUMBER = 10

DIRECTORY_PATH = os.getcwd()
DATA_DIRECTORY_PATH = os.path.join(DIRECTORY_PATH, "data")
EMOTIFY_DATA_PATH = os.path.join(DATA_DIRECTORY_PATH, "emotifymusic")
EMOTIFY_SAMPLES_PATH = os.path.join(EMOTIFY_DATA_PATH, "samples")
EMOTIFY_LABELS_PATH = os.path.join(EMOTIFY_DATA_PATH, "labels.csv")
EMOTIFY_DUMP_PATH = os.path.join(EMOTIFY_DATA_PATH, "data.pkl")
EMOTIFY_SPECTROGRAM_PATH = os.path.join(EMOTIFY_DATA_PATH, "spectro.pkl")
EMOTIFY_LABELS_PATH = os.path.join(EMOTIFY_DATA_PATH, "labels.pkl")
