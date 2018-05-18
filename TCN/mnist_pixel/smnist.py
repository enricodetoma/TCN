from pathlib import Path
import random 
from datetime import datetime
import sys

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("data/mnist/raw", one_hot=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Network Parameters
num_input = 1 # MNIST data input (img shape: 28*28)
timesteps = 28*28 # timesteps
num_hidden = 130 # hidden layer num of features as in TCN paper
num_classes = 10 # MNIST total classes (0-9 digits)
training_steps = 10000
display_step = 100

# tf Graph input
X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


def RNN(x):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output_layer = tf.contrib.layers.fully_connected(outputs[-1],
        num_classes, activation_fn=None)

    # Linear activation, using rnn inner loop last output
    return output_layer

if __name__ == '__main__':
    # Training Parameters
    learning_rate = 1e-3
    batch_size = 32
    if len(sys.argv) > 1:
        learning_rate = float(sys.argv[1])
        batch_size = int(sys.argv[2])

    logits = RNN(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))

    # Change optimizer to RMSProp and add gradient crop
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(loss_op)
    capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # format test data
    test_data = mnist.test.images.reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels
    f = open('lstm_smnist_784x1_%.4f_%d.txt' % (
        learning_rate, batch_size), 'w')

    # Start training
    with tf.Session(config=config) as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, training_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                test_acc = sess.run(accuracy, feed_dict={X: test_data,Y: test_label})

                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.4f}".format(acc)+ \
                      ", Testing Accuracy= " + "{:.4f}".format(test_acc))

                f.write("%d, %f\n" % (step, test_acc))
        print("Optimization Finished!")

        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    f.close()