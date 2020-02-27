import config
from utils import display_and_reset_metrics
from pipeline import initialize_pipeline
from models.basic_cnn import BasicCNN
from models.pianocktail_gru import PianocktailGRU

def test_model(test_dataset, forward_pass, test_loss, test_accuracy):
    print(f"======================== evaluation on test data =========================")
    for iteration, (spectro, label) in enumerate(test_dataset):
        predictions, loss = forward_pass(spectro, label)

        # compute metrics
        test_loss.update_state(loss)
        test_accuracy.update_state(label, predictions)

    display_and_reset_metrics(test_loss, test_accuracy, predictions, label, is_test=True)

def initialize_and_test_model(model, batch_size, random_pick, learning_rate, should_restore_checkpoint = False):
    forward_pass, _, _, test_dataset, _, _, test_loss, test_accuracy, _, _ = initialize_pipeline(model, batch_size, random_pick, learning_rate, should_restore_checkpoint = False)
    test_model(test_dataset, forward_pass, test_loss, test_accuracy)

if __name__ == '__main__':
    model = PianocktailGRU()
    batch_size = config.BATCH_SIZE
    random_pick = config.RANDOM_PICK
    learning_rate = config.LEARNING_RATE
    should_restore_checkpoint = config.RESTORE_CHECKPOINT
    
    initialize_and_test_model(model, batch_size, random_pick, learning_rate, should_restore_checkpoint = False)