import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("../tensorflow_mooc/data", one_hot=True)
tf.logging.set_verbosity(old_v)

BATCH_SIZE = 512
STEP = 2000

def forward(x, keep_prob):
    W1 = get_weight([784, 1000])
    b1 = get_bias([1000])
    A1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W1), b1))
    A1_dropout = tf.nn.dropout(A1, keep_prob)
    W2 = get_weight([1000, 500])
    b2 = get_bias([500])
    A2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(A1_dropout, W2), b2))
    A2_dropout = tf.nn.dropout(A2, keep_prob)
    W3 = get_weight([500, 200])
    b3 = get_bias([200])
    A3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(A2_dropout, W3), b3))
    A3_dropout = tf.nn.dropout(A3, keep_prob)
    W4 = get_weight([200, 10])
    b4 = get_bias([10])
    y_hat = tf.nn.bias_add(tf.matmul(A3_dropout, W4), b4)
    return y_hat
    
def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))

def get_bias(shape):
    return tf.Variable(tf.zeros(shape))

def backward(mnist, pro_value):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    y_hat = forward(x, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
    equal_op = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equal_op, tf.float32))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train, feed_dict={x: xs, y: ys, keep_prob: pro_value})
            if i % 200 == 0:
                loss_val = sess.run(loss, feed_dict={x: xs, y: ys, keep_prob: 1.0})
                train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})
                print("After %6d steps, loss is %.6f, train accuracy %.6f, test accuracy %.6f" % (i, loss_val, train_acc, test_acc))

                
if __name__ == "__main__":
    print("without dropout")
    print("-" * 50)
    backward(mnist, 1.0)
    print("with dropout, keep_prob=0.5")
    print("-" * 50)
    backward(mnist, 0.5)