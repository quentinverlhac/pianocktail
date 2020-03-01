from utils import load_model, load_dump, test_model
import tensorflow as tf
import argparse

import config


def test(model):
    print(f"======================== evaluation on test data =========================")
    # import data and labels
    test_spectrograms = load_dump(config.TEST_DATA_PATH)
    test_labels = load_dump(config.TEST_LABELS_PATH)

    # define metrics
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    test_model(model, test_spectrograms, test_labels, test_loss, test_accuracy, is_test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path of the model to evaluate")
    args = parser.parse_args()
    model = load_model(args.model_path)
    test(model)
