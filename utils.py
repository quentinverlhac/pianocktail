import numpy as np
import config
import pickle as pkl


def draw_subspectrogram(spectrogram,sub_len,seed=42) :
    """
    Draw a random subspectrogram of given time length from the given spectrogram
    """
    fft_rate = config.SAMPLING_RATE/config.FFT_HOP
    n_points = int(sub_len*fft_rate)
    print(n_points)
    np.random.seed(seed)
    offset = int(np.random.random()*(spectrogram.shape[1]/fft_rate-sub_len))
    return spectrogram[:,offset:offset+n_points]


def dump_elements(elements, dump_path):
    """
    Dumps all elements in the list to a binary file on the dump path
    """
    with open(dump_path, 'wb') as f:
        pkl.dump(elements, f)
