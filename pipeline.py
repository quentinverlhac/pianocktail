import tensorflow as tf

import config
from utils import draw_subspectrogram, load_dump, setup_checkpoints

def get_datasets(batch_size, random_pick):
    # import data
    train_spectrograms = load_dump(config.DEV_DATA_PATH if config.IS_DEV_MODE else config.TRAIN_DATA_PATH)
    test_spectrograms = load_dump(config.TEST_DATA_PATH)

    # import labels
    train_labels = load_dump(config.DEV_LABELS_PATH if config.IS_DEV_MODE else config.TRAIN_LABELS_PATH)
    test_labels = load_dump(config.TEST_LABELS_PATH)

    # generate datasets
    def generate_subspectrogram(random_pick = True, use_test_dataset = False, duration_s = config.SUBSPECTROGRAM_DURATION_S, fft_rate = config.FFT_RATE, mel_bins = config.MEL_BINS):
        spectrograms = train_spectrograms
        labels = train_labels
        if use_test_dataset:
            spectrograms = test_spectrograms
            labels = test_labels

        for i in range(len(labels)):
            sub_spectro = draw_subspectrogram(spectrograms[i], duration_s, fft_rate, random_pick = True)
            tensor_spectro = tf.convert_to_tensor(sub_spectro)
            tensor_spectro = tf.transpose(tensor_spectro)
            tensor_label = tf.convert_to_tensor(labels[i])
            yield tensor_spectro, tensor_label

    def generate_randomized_batched_dataset(batch_size = 1, random_pick = True, use_test_dataset = False, duration_s = config.SUBSPECTROGRAM_DURATION_S, fft_rate = config.FFT_RATE, mel_bins = config.MEL_BINS):
        dataset = tf.data.Dataset.from_generator(generate_subspectrogram, (tf.float32, tf.float32), args=[use_test_dataset, random_pick, duration_s, fft_rate, mel_bins])
        return dataset.batch(batch_size)

    train_dataset = generate_randomized_batched_dataset(batch_size, random_pick)
    test_dataset = generate_randomized_batched_dataset(batch_size, random_pick, use_test_dataset=True)

    return train_dataset, test_dataset

def build_model(model, batch_size):
    model.build(input_shape=(batch_size, config.SUBSPECTROGRAM_POINTS, config.MEL_BINS))
    model.summary()

def set_up_optimizer(learning_rate):
    return tf.optimizers.Adam(learning_rate)

def define_metrics():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    return train_loss, train_accuracy, test_loss, test_accuracy

def initialize_checkpoint(model, optimizer):
    checkpoint, checkpoint_manager = setup_checkpoints(model, optimizer)
    return checkpoint, checkpoint_manager

def get_forward_and_train_functions(model, optimizer, train_loss, train_accuracy):
    # declaring forward pass and gradient descent
    @tf.function
    def forward_pass(inputs, labels):
        print("tracing forward graph")
        predictions = model.call(inputs)
        loss = tf.keras.losses.binary_crossentropy(
            y_true = labels, 
            y_pred = predictions
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
        train_accuracy.update_state(labels, predictions)

        return predictions, labels

    return forward_pass, train_step

def restore_checkpoint(checkpoint, checkpoint_manager):
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f"Restored checkpoint. Model {checkpoint.model.name} - epoch {checkpoint.epoch.value()}")

def initialize_pipeline(model, batch_size, random_pick, learning_rate, should_restore_checkpoint = False):
    train_dataset, test_dataset = get_datasets(batch_size, random_pick)
    build_model(model, batch_size)
    optimizer = set_up_optimizer(learning_rate)
    train_loss, train_accuracy, test_loss, test_accuracy = define_metrics()
    checkpoint, checkpoint_manager = initialize_checkpoint(model, optimizer)
    forward_pass, train_step = get_forward_and_train_functions(model, optimizer, train_loss, train_accuracy)
    if should_restore_checkpoint:
        restore_checkpoint(checkpoint, checkpoint_manager)
    return forward_pass, train_step, train_dataset, test_dataset, train_loss, train_accuracy, test_loss, test_accuracy, checkpoint, checkpoint_manager
