import numpy as np
import config
import pickle as pkl

def draw_subspectrogram(spectrogram,sub_len) :
    """
    Draw a random subspectrogram of given time length from the given spectrogram
    """
    fft_rate = config.SAMPLING_RATE/config.FFT_HOP
    n_points = int(sub_len*fft_rate)
    offset = int(np.random.random()*(spectrogram.shape[1] - sub_len*fft_rate))
    return spectrogram[:,offset:offset+n_points]
