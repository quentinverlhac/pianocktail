from utils import display_and_reset_metrics, load_dump, segment_spectrogram, load_model
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
        print(label)
        predictions = average_predictions(test_spectrograms[iteration], model)
        print(predictions)
        # compute metrics
        loss = tf.keras.metrics.binary_crossentropy(y_pred=predictions, y_true=label)
        test_loss.update_state(loss)
        test_accuracy.update_state(label, predictions)

    display_and_reset_metrics(test_loss, test_accuracy, predictions, label, is_test=True)

def average_predictions(full_spectrogram, model):
    """
    Average predictions of the model over all segments in the spectrogram
    """
    segmented_spectro = segment_spectrogram(full_spectrogram, config.SUBSPECTROGRAM_DURATION_S, config.FFT_RATE)
    all_predictions = []
    for spectro in segmented_spectro:
        tensor_spectro = tf.convert_to_tensor(spectro)
        tensor_spectro = tf.transpose(tensor_spectro)
        tensor_spectro = tf.reshape(tensor_spectro,[1,tensor_spectro.shape[0],tensor_spectro.shape[1]])
        all_predictions.append(model.call(tensor_spectro))
    return tf.add_n(all_predictions)/len(segmented_spectro)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path",help="path of the model to evaluate")
    args = parser.parse_args()
    model = load_model(args.model_path)
    test_model(model)