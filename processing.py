import os

import librosa.display
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import argparse

import config
import utils


def normalise(spectrograms):
    normalised_spectrograms = []
    for spectrogram in spectrograms:
        normalised_spectrograms.append(utils.normalise_by_max(spectrogram))
    return normalised_spectrograms


def get_raw_labels(path):
    """
    Load labels and process them.
    9 columns between 0 and 1 represent the percentage of respondants who tagged the song with the corresponding emotion.
    """
    df = pd.read_csv(path)
    # remove leading spaces
    df.rename(columns=lambda s: s.strip(), inplace=True)
    # drop unwanted columns
    df.drop(columns=["mood", "liked", "disliked", "age", "gender", "mother tongue"], inplace=True)
    # get the average score on all user answers for each song
    grouped_df = df.groupby(['genre', 'track id']).mean().reset_index()
    # Song ids are split between genre and range between 1 and 100 in data folders,
    # but range between 1 and 400 in labels.csv
    # This line changes ids from labels.csv to match data file names
    grouped_df['track id'] = ((grouped_df['track id'] - 1) % 100) + 1
    return grouped_df


def get_label_emotion_scores_for_track(raw_labels_df, genre, track_id):
    """
    Extract emotion scores matching song from raw labels dataframe. 
    Songs are uniquely defined by genre and track_id.
    Raises an exception when multiple rows are found for a unique song.
    """
    # Get the label row(s) that match the song (genre and id)
    matching_label_rows = raw_labels_df.loc[(raw_labels_df["genre"] == genre) & (raw_labels_df["track id"] == track_id)]
    # Select only emotion columns and convert the row as list 
    matching_label_lists = matching_label_rows[config.EMOTIFY_EMOTIONS_ORDERED_LIST].values.tolist()
    # Check that there is only one matching song in labels
    if len(matching_label_lists) > 1:
        raise ValueError("matching_label_lists has more than one matching song: {}".format(matching_label_lists))
    return matching_label_lists[0]

def thresh_label(label_list,threshold):
    """
    Encode label as 0 or 1 depending on the given threshold
    """
    return [0 if label < threshold else 1 for label in label_list]

def major_label(label_list):
    """
    Encode label by setting the label with the highest probability to one
    """
    return [1 if np.argmax(label_list) == ind else 0 for ind in range(len(label_list))]


def get_raw_data(path):
    """Convert to mono and return an array of samples"""
    segment = AudioSegment.from_mp3(path)
    left, right = segment.split_to_mono()
    segment = left.overlay(right)
    return segment.get_array_of_samples()


def processing_main(args,label_encoding=config.LABEL_ENCODING):
    """
    Import labels and samples from original dataset. 
    Process them and dump the output as pickle files.
    """
    raw_labels_df = get_raw_labels(config.EMOTIFY_LABELS_CSV_PATH)
    labels = []

    songs_paths = []
    for outer_path in os.listdir(config.EMOTIFY_SAMPLES_PATH):
        if os.path.isdir(os.path.join(config.EMOTIFY_SAMPLES_PATH, outer_path)):
            for inner_path in os.listdir(os.path.join(config.EMOTIFY_SAMPLES_PATH, outer_path)):
                # samples
                songs_paths.append(os.path.join(config.EMOTIFY_SAMPLES_PATH, outer_path, inner_path))
                # labels
                genre = outer_path
                track_id = int(inner_path.split(".")[0])
                label = get_label_emotion_scores_for_track(raw_labels_df, genre, track_id)
                if label_encoding == config.LabelEncodingEnum.MAJORITY:
                    labels.append(major_label(label))
                elif label_encoding == config.LabelEncodingEnum.THRESHOLD:
                    labels.append(thresh_label(label,config.EMOTION_THRESH))
                else:
                    labels.append(label)

    # Dumping labels from the emotify dataset in the emotify label dump file
    label_path = config.DEV_LABELS_PATH if config.IS_DEV_MODE else config.EMOTIFY_LABELS_DUMP_PATH
    label_list = labels[:config.DEV_MODE_SAMPLE_NUMBER] if config.IS_DEV_MODE else labels
    utils.dump_elements(label_list, label_path)

    if not args.label_only:
        song_list = []
        for i in tqdm(range(config.DEV_MODE_SAMPLE_NUMBER if config.IS_DEV_MODE else len(songs_paths))):
            song_list.append(get_raw_data(songs_paths[i]))

        all_mel_spectrogram = []
        for time_series in tqdm(song_list):
            all_mel_spectrogram.append(librosa.feature.melspectrogram(
                y=np.array(time_series, dtype=np.float), sr=config.SAMPLING_RATE, hop_length=config.FFT_HOP))
    
        all_mel_spectrogram = normalise(all_mel_spectrogram)
        # Dumping song spectrograms
        utils.dump_elements(all_mel_spectrogram, config.DEV_DATA_PATH if config.IS_DEV_MODE else config.EMOTIFY_SPECTROGRAM_DUMP_PATH)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-only", action="store_true", help="only compute labels")
    args = parser.parse_args()
    processing_main(args)
