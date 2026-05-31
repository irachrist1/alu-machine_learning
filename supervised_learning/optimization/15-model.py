#!/usr/bin/env python3
"""Full neural network model with optimization techniques."""
import numpy as np
import tensorflow as tf

create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
shuffle_data = __import__('2-shuffle_data').shuffle_data


def forward_prop(x, layers, activations):
    """Build the forward propagation graph with batch normalization.

    Args:
        x: input placeholder tensor.
        layers: list of node counts for each layer.
        activations: list of activation functions for each layer.

    Returns:
        The output tensor of the network.
    """
    prev = x
    for nodes, activation in zip(layers[:-1], activations[:-1]):
        prev = create_batch_norm_layer(prev, nodes, activation)
    return tf.layers.Dense(units=layers[-1], activation=activations[-1])(prev)


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Build, train, and save a neural network with Adam optimization.

    Args:
        Data_train: tuple of training inputs and labels.
        Data_valid: tuple of validation inputs and labels.
        layers: list of node counts for each layer.
        activations: list of activation functions for each layer.
        alpha: learning rate.
        beta1: Adam first moment weight.
        beta2: Adam second moment weight.
        epsilon: small number to avoid division by zero.
        decay_rate: inverse time decay rate.
        batch_size: mini-batch size.
        epochs: number of training passes through the dataset.
        save_path: path where the model should be saved.

    Returns:
        The path where the model was saved.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)), tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    decay_step = X_train.shape[0] / batch_size
    alpha_decay = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(
        alpha_decay, beta1, beta2, epsilon).minimize(loss, global_step)
    tf.add_to_collection('train_op', train_op)

    train_data = {x: X_train, y: Y_train}
    valid_data = {x: X_valid, y: Y_valid}
    metrics = (loss, accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
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
                step = 0
                for batch_start in range(0, X_train.shape[0], batch_size):
                    batch_end = batch_start + batch_size
                    mini_batch = {
                        x: X_shuffled[batch_start:batch_end],
                        y: Y_shuffled[batch_start:batch_end]
                    }
                    sess.run(train_op, mini_batch)
                    step += 1
                    if step % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            metrics, mini_batch)
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy {}".format(step_accuracy))

        return saver.save(sess, save_path)
