from utils import display_and_reset_metrics
from pipeline import *

def test_model():
    print(f"======================== evaluation on test data =========================")
    for iteration, (spectro, label) in enumerate(test_dataset):
        predictions, loss = forward_pass(spectro, label)

        # compute metrics
        test_loss.update_state(loss)
        test_accuracy.update_state(label, predictions)

    display_and_reset_metrics(test_loss, test_accuracy, predictions, label, is_test=True)

if __name__ == '__main__':
    test_model()