from utils import display_and_reset_metrics, load_dump, segment_spectrogram
import tensorflow as tf
import argparse

import config

def test_model(model):
    # import data and labels
    test_spectrograms = load_dump(config.TEST_DATA_PATH)
    test_labels = load_dump(config.TEST_LABELS_PATH)

    # define metrics
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    print(f"======================== evaluation on test data =========================")
    for iteration in range(len(test_labels)):
        label = test_labels[iteration]
        predictions = average_predictions(test_spectrograms[iteration], model)

        # compute metrics
        test_loss.update_state(label, predictions)
        test_accuracy.update_state(label, predictions)

    display_and_reset_metrics(test_loss, test_accuracy, predictions, label, is_test=True)

def average_predictions(full_spectrogram, model):
    segmented_spectro = segment_spectrogram(full_spectrogram, config.SUBSPECTROGRAM_DURATION_S, config.FFT_RATE)
    mean_predictions = tf.metrics.Mean()
    for spectro in segmented_spectro:
        tensor_spectro = tf.convert_to_tensor(spectro)
        tensor_spectro = tf.transpose(tensor_spectro)
        predictions = model.call(tensor_spectro)
        mean_predictions.update_state(predictions)
    return mean_predictions.result()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path",help="path of the model to evaluate")
    args = parser.parse_args()
    model = tf.keras.models.load_model(args.model_path)
    test_model(model)