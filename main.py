import librosa
import librosa.display
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import audiosegment as extended_audiosegment
from pydub import AudioSegment
from tqdm import tqdm_notebook
import pickle as pkl

# Config variables

# Set this to test code on limited sample number. Useful to test faster.
IS_DEV_MODE = False
DEV_MODE_SAMPLE_NUMBER = 10

EMOTIFY_DATA_PATH = "/content/drive/My Drive/Deep Fried Learning/Datasets/emotifymusic"
EMOTIFY_LABELS_PATH = "/content/drive/My Drive/Deep Fried Learning/Datasets/emotifymusic/data.csv"
EMOTIFY_DUMP_PATH = "/content/drive/My Drive/Deep Fried Learning/Datasets/emotifymusic/data.pkl"