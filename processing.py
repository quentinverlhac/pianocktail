import librosa.display
import os
import pickle as pkl

import librosa.display
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm_notebook

import config


def get_processed_labels(path):
    df = pd.read_csv(path)
    # remove leading spaces
    df.rename(columns=lambda s: s.strip(), inplace=True)
    # drop unwanted columns
    df.drop(columns=["mood", "liked", "disliked", "age", "gender", "mother tongue"], inplace=True)
    # get the average score on all user answers for each song
    return df.groupby(['track id', 'genre']).mean()


def get_raw_data(path):
    """Convert to mono and return an array of samples"""
    segment = AudioSegment.from_mp3(path)
    left, right = segment.split_to_mono()
    segment = left.overlay(right)
    return segment.get_array_of_samples()


def dump_all_songs(songs_paths, dump_paths):
    """
    Dumps all songs with path in the list to a binary file on the dump path
    """
    song_list = []
    for i in tqdm_notebook(range(config.DEV_MODE_SAMPLE_NUMBER if config.IS_DEV_MODE else len(path_list))):
        song_list.append(get_raw_data(path_list[i]))
    with open(config.emotify_dump_path, 'wb') as f:
        pkl.dump(song_list, f)


# Load labels and process them.
# 9 columns between 0 and 1 represent the percentage of respondants who tagged the song with the corresponding emotion.

processed_labels_df = get_processed_labels(config.EMOTIFY_LABELS_PATH)

# Dumping songs from the emotify dataset in the emotify dump file
path_list = []
for outer_path in os.listdir(config.EMOTIFY_DATA_PATH):
    if os.path.isdir(os.path.join(config.EMOTIFY_DATA_PATH, outer_path)):
        for inner_path in os.listdir(os.path.join(config.EMOTIFY_DATA_PATH, outer_path)):
            path_list.append(os.path.join(config.EMOTIFY_DATA_PATH, outer_path, inner_path))

dump_all_songs(path_list, config.EMOTIFY_DUMP_PATH)

with open(config.EMOTIFY_DUMP_PATH, "rb") as file:
    all_time_series = pkl.load(file)
    all_mel_spectrogram = []
    for time_series in tqdm_notebook(all_time_series):
        all_mel_spectrogram.append(librosa.feature.melspectrogram(y=np.array(time_series, dtype=np.float)))
