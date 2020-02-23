import tensorflow as tf

import config
from utils import draw_subspectrogram, load_dump, save_model, setup_checkpoints


def main():
    # import data
    spectrograms = load_dump(config.EMOTIFY_SPECTROGRAM_DUMP_PATH)

    # import labels
    train_labels = load_dump(config.EMOTIFY_LABELS_DUMP_PATH)

    # generate dataset
    def generate_subspectrogram():
        for i in range(len(train_labels)):
            sub_spectro = draw_subspectrogram(spectrograms[i])
            tensor_spectro = tf.convert_to_tensor(sub_spectro)
            tensor_spectro = tf.reshape(tensor_spectro, (config.MEL_POINTS, config.SUBSPECTROGRAM_POINTS, 1))
            tensor_label = tf.convert_to_tensor(train_labels[i])
            yield tensor_spectro, tensor_label

    train_dataset = tf.data.Dataset.from_generator(generate_subspectrogram, (tf.float32, tf.float32))
    train_dataset = train_dataset.batch(config.BATCH_SIZE)

    # building the model
    from models.cnn import ConvModel as Model
    model = Model()
    model.build(input_shape=(config.BATCH_SIZE, config.MEL_POINTS, config.SUBSPECTROGRAM_POINTS, 1))
    model.summary()
    optimizer = tf.optimizers.Adam(config.LEARNING_RATE)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

    checkpoint, checkpoint_manager = setup_checkpoints(model, optimizer)

    # declaring forward pass and gradient descent
    @tf.function
    def forward_pass(inputs, labels):
        print("tracing forward graph")
        predictions = model.call(inputs)
        loss = tf.losses.categorical_crossentropy(
            y_true=labels,
            y_pred=predictions
        )
        return predictions, loss

    @tf.function
    def train_step(inputs, labels):
        print("tracing train graph")
        with tf.GradientTape() as tape:
            predictions, loss = forward_pass(inputs, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # compute metrics
        train_loss.update_state(loss)
        train_accuracy.update_state(predictions, labels)

    # ============================ train the model =============================
    # restore checkpoint
    if config.RESTORE_CHECKPOINT:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(
            f"Restored checkpoint. Model {checkpoint.model.name} - epoch {checkpoint.epoch.value()} - iteration {checkpoint.iteration.value()}")
    # for test we iterate over samples one by one
    for epoch in range(config.NB_EPOCHS):

        print(f"============================ epoch {epoch} =============================")
        checkpoint.epoch.assign(epoch)

        for iteration, (spectro, label) in enumerate(train_dataset):
            train_step(spectro, label)

            # display metrics
            if iteration % 10 == 0:
                template = 'iteration {} - loss: {:4.2f} - accuracy: {:5.2%}'
                print(template.format(iteration, train_loss.result(), train_accuracy.result()))
                train_loss.reset_states()
                train_accuracy.reset_states()

            # manage checkpoint
            checkpoint.iteration.assign(iteration)
            if iteration % config.SAVE_PERIOD == 0:
                checkpoint_manager.save()

    save_model(model, epoch)


if __name__ == "__main__":
    main()
