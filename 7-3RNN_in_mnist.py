import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

MAX_TIMES = 28
ENCODE_LENGTH = 28
LSTM_SIZE = 100
OUTPUT_NODE = 10
STEP = 2000
BATCH_SIZE = 64

def forward(x):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    """
    final_state[0]: cell state
    final_state[1]: output state
    """
    
    weight = tf.Variable(tf.truncated_normal([LSTM_SIZE, OUTPUT_NODE], stddev=0.01), name="weight")
    bias = tf.Variable(tf.zeros([OUTPUT_NODE]), name="bias")
    result = tf.matmul(final_state[1], weight) + bias
    return result
    

def backward(mnist):
    x = tf.placeholder(tf.float32, shape=[None, MAX_TIMES, ENCODE_LENGTH])
    y = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE])
    
    y_hat = forward(x)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)
    
    correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs_reshape = np.reshape(xs, [-1, MAX_TIMES, ENCODE_LENGTH])
            sess.run(train, feed_dict={x: xs_reshape, y: ys})
            if i % 200 == 0:
                train_x_reshape = np.reshape(mnist.train.images, [-1, MAX_TIMES, ENCODE_LENGTH])
                train_accuracy = sess.run(accuracy, feed_dict={x: train_x_reshape, y: mnist.train.labels})
                test_x_reshape = np.reshape(mnist.test.images, [-1, MAX_TIMES, ENCODE_LENGTH])
                test_accuracy = sess.run(accuracy, feed_dict={x: test_x_reshape, y: mnist.test.labels})
                print("After %5d steps, train accuracy is %.6f, test accuracy is %.6f" % (i, train_accuracy, test_accuracy))
                

if __name__ == "__main__":
    mnist = input_data.read_data_sets("./data", one_hot=True)
    backward(mnist)
    # After  1800 steps, train accuracy is 0.976382, test accuracy is 0.974800