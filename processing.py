import librosa.display
import os
import pickle as pkl

import librosa.display
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

import config


def get_raw_labels(path):
    df = pd.read_csv(path)
    # remove leading spaces
    df.rename(columns=lambda s: s.strip(), inplace=True)
    # drop unwanted columns
    df.drop(columns=["mood", "liked", "disliked", "age", "gender", "mother tongue"], inplace=True)
    # get the average score on all user answers for each song
    grouped_df = df.groupby(['genre', 'track id']).mean().reset_index()
    # Song ids are split between genre and range between 1 and 100 in data folders, but it ranges between 1 and 400 in labels.csv
    # This line changes ids from labels.csv to match data file names
    grouped_df['track id'] = ((grouped_df['track id'] - 1) % 100) + 1
    return grouped_df


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
    for i in tqdm(range(config.DEV_MODE_SAMPLE_NUMBER if config.IS_DEV_MODE else len(path_list))):
        song_list.append(get_raw_data(path_list[i]))
    with open(config.EMOTIFY_DUMP_PATH, 'wb') as f:
        pkl.dump(song_list, f)


# Load labels and process them.
# 9 columns between 0 and 1 represent the percentage of respondants who tagged the song with the corresponding emotion.

raw_labels_df = get_raw_labels(config.EMOTIFY_LABELS_CSV_PATH)
labels = []

# Dumping songs from the emotify dataset in the emotify dump file
path_list = []
for outer_path in os.listdir(config.EMOTIFY_SAMPLES_PATH):
    if os.path.isdir(os.path.join(config.EMOTIFY_SAMPLES_PATH, outer_path)):
        for inner_path in os.listdir(os.path.join(config.EMOTIFY_SAMPLES_PATH, outer_path)):
            path_list.append(os.path.join(config.EMOTIFY_SAMPLES_PATH, outer_path, inner_path))
            # Get the label row(s) that match the song (genre and id)
            matching_label_rows = raw_labels_df.loc[(raw_labels_df["genre"] == outer_path) & (raw_labels_df["track id"] == int(inner_path.split(".")[0]))]
            # Select only emotion columns and convert the row as list 
            matching_label_lists = matching_label_rows[config.EMOTIFY_EMOTIONS_ORDERED_LIST].values.tolist()
            # Check that there is only one matching song in labels
            if (len(matching_label_lists) > 1):
                raise ValueError("matching_label_lists has more than one matching song: {}".format(matching_label_lists))
            labels.append(matching_label_lists[0])

dump_all_songs(path_list, config.EMOTIFY_DUMP_PATH)

with open(config.EMOTIFY_DUMP_PATH, "rb") as file:
    all_time_series = pkl.load(file)

all_mel_spectrogram = []
for time_series in tqdm(all_time_series):
    all_mel_spectrogram.append(librosa.feature.melspectrogram(y=np.array(time_series, dtype=np.float)))

# Dumping spectrograms in another file
with open(config.EMOTIFY_SPECTROGRAM_PATH, "wb") as f:
    pkl.dump(all_mel_spectrogram, f)
