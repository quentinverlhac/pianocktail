import tensorflow as tf

import config
from utils import save_model, display_and_reset_metrics

from pipeline import *

def train_model():
    # for test we iterate over samples one by one
    for epoch in range(config.NB_EPOCHS):

        print(f"============================ epoch {epoch} =============================")
        checkpoint.epoch.assign(epoch)

        for iteration, (spectro, label) in enumerate(train_dataset):
            predictions, labels = train_step(spectro, label)

            # display metrics
            if iteration % 10 == 0:
                display_and_reset_metrics(train_loss, train_accuracy, predictions, labels, iteration=iteration)

            # manage checkpoint
            checkpoint.iteration.assign(iteration)
            if iteration % config.SAVE_PERIOD == 0:
                checkpoint_manager.save()

    save_model(model, epoch)

if __name__ == '__main__':
    train_model()