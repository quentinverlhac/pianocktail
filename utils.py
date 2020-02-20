import numpy as np
import config
import pandas as pd
import pickle as pkl

def draw_subspectrogram(spectrogram,sub_len) :
    """
    Draw a random subspectrogram of given time length from the given spectrogram
    """
    fft_rate = config.SAMPLING_RATE/config.FFT_HOP
    n_points = int(sub_len*fft_rate)
    offset = int(np.random.random()*(spectrogram.shape[1] - sub_len*fft_rate))
    return spectrogram[:,offset:offset+n_points]


def dump_elements(elements, dump_path):
    """
    Dumps all elements in the list to a binary file on the dump path
    """
    with open(dump_path, 'wb') as f:
        pkl.dump(elements, f)

def load_dump(dump_path):
    """
    Load the content of the file at dump path
    """
    with open(dump_path, "rb") as file:
        return pkl.load(file)

def load_labels(path):
    """
    Load a pickle file containing list of labels and return as pandas DataFrame
    """
    with open(path, "rb") as f:
        labels = pkl.load(f)

    # Transform labels to pandas DataFrame
    labels = pd.DataFrame(labels)
    labels.columns = config.EMOTIFY_EMOTIONS_ORDERED_LIST
    return labels
