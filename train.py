import tensorflow as tf

import config
from utils import save_model, display_and_reset_metrics
from pipeline import initialize_pipeline
from models.basic_cnn import BasicCNN
from models.pianocktail_gru import PianocktailGRU


def train_model(model, train_dataset, train_step, train_loss, train_accuracy, checkpoint, checkpoint_manager):
    # for test we iterate over samples one by one
    for epoch in range(checkpoint.epoch.value(), config.NB_EPOCHS):

        print(f"============================ epoch {epoch} =============================")

        for iteration, (spectro, label) in enumerate(train_dataset):
            predictions, labels = train_step(spectro, label)

            # display metrics
            if iteration % 10 == 0:
                display_and_reset_metrics(train_loss, train_accuracy, predictions, labels, iteration=iteration)

        # save checkpoint
        checkpoint.epoch.assign_add(1)
        checkpoint_manager.save()

    save_model(model, epoch)

def initialize_and_train_model(model, batch_size, random_pick, learning_rate, should_restore_checkpoint = False):
    _, train_step, train_dataset, _, train_loss, train_accuracy, _, _, checkpoint, checkpoint_manager = initialize_pipeline(model, batch_size, random_pick, learning_rate, should_restore_checkpoint = False)
    train_model(model, train_dataset, train_step, train_loss, train_accuracy, checkpoint, checkpoint_manager)

if __name__ == '__main__':
    model = PianocktailGRU()
    batch_size = config.BATCH_SIZE
    random_pick = config.RANDOM_PICK
    learning_rate = config.LEARNING_RATE
    should_restore_checkpoint = config.RESTORE_CHECKPOINT
    
    initialize_and_train_model(model, batch_size, random_pick, learning_rate, should_restore_checkpoint = False)