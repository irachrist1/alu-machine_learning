#!/usr/bin/env python3
"""Mini-batch gradient descent training module."""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Train a loaded neural network using mini-batch gradient descent.

    Args:
        X_train: training input data of shape (m, 784).
        Y_train: one-hot training labels of shape (m, 10).
        X_valid: validation input data of shape (m, 784).
        Y_valid: one-hot validation labels of shape (m, 10).
        batch_size: number of data points in a batch.
        epochs: number of passes through the dataset.
        load_path: path from which to load the model.
        save_path: path where the model should be saved.

    Returns:
        The path where the model was saved.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        train_data = {x: X_train, y: Y_train}
        valid_data = {x: X_valid, y: Y_valid}
        metrics = (loss, accuracy)

        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run(metrics, train_data)
            valid_cost, valid_accuracy = sess.run(metrics, valid_data)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                batch_start = 0
                batch_end = batch_size
                step = 1
                while True:
                    mini_batch = {
                        x: X_shuffled[batch_start:batch_end],
                        y: Y_shuffled[batch_start:batch_end]
                    }
                    sess.run(train_op, mini_batch)
                    if step % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            metrics, mini_batch)
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))
                    if batch_end >= X_shuffled.shape[0]:
                        break
                    batch_start += batch_size
                    batch_end += batch_size
                    step += 1

        return saver.save(sess, save_path)
