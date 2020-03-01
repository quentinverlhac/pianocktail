import utils
import processing
import librosa
import config
import numpy as np
import argparse

def detect_emotions(model, song_path):
    """
    Score the given song using one of the trained models
    """
    raw_song = processing.get_raw_data(song_path)
    spectrogram = librosa.feature.melspectrogram(
        y=np.array(raw_song, dtype=np.float), sr=config.SAMPLING_RATE, hop_length=config.FFT_HOP)
    
    spectrogram = utils.normalise_by_max(spectrogram)
    predictions = utils.average_predictions(spectrogram, model)
    return predictions.numpy()

def display_emotions(scores):
    """
    display score for each emotion
    """
    emotions = config.EMOTIFY_EMOTIONS_ORDERED_LIST
    for i in range(len(emotions)):
        template = "{} : {:1.2f}"
        print(template.format(emotions[i], scores[0][i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model used to predict data")
    parser.add_argument("song_path", help="path to songs to predict scores from")
    args = parser.parse_args()
    model = utils.load_model(args.model_path)
    scores = detect_emotions(model, args.song_path)
    display_emotions(scores)