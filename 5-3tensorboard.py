import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("../tensorflow_mooc/data", one_hot=True)
tf.logging.set_verbosity(old_v)

BATCH_SIZE = 512
STEP = 2000

def variable_summary(var):
    with tf.name_scope("summary"):
        tf.summary.scalar("mean", tf.reduce_mean(var))
        tf.summary.scalar("stddin", tf.square(tf.reduce_mean(tf.square(var - tf.reduce_mean(var)))))
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

def forward(x, keep_prob):
    with tf.name_scope("forward"):
        with tf.name_scope("layer-1"):
            with tf.name_scope("weight-1"):
                W1 = get_weight([784, 1000])
                variable_summary(W1)
            with tf.name_scope("bias-1"):
                b1 = get_bias([1000])
                variable_summary(b1)
            with tf.name_scope("XW_plus_b"):
                xw_plus_b = tf.nn.bias_add(tf.matmul(x, W1), b1)
            with tf.name_scope("relu"):
                A1 = tf.nn.relu(xw_plus_b)
            with tf.name_scope("dropout"):
                A1_dropout = tf.nn.dropout(A1, keep_prob)
        with tf.name_scope("layer-2"):
            with tf.name_scope("weight-2"):
                W2 = get_weight([1000, 500])
                variable_summary(W2)
            with tf.name_scope("bias-2"):
                b2 = get_bias([500])
                variable_summary(b2)
            with tf.name_scope("XW_plus_b"):
                xw_plus_b = tf.nn.bias_add(tf.matmul(A1_dropout, W2), b2)
            with tf.name_scope("relu"):
                A2 = tf.nn.relu(xw_plus_b)
            with tf.name_scope("dropout"):
                A2_dropout = tf.nn.dropout(A2, keep_prob)
        with tf.name_scope("layer-3"):
            with tf.name_scope("weight-3"):
                W3 = get_weight([500, 200])
                variable_summary(W3)
            with tf.name_scope("bias-3"):
                b3 = get_bias([200])
                variable_summary(b3)
            with tf.name_scope("XW_plus_b"):
                xw_plus_b = tf.nn.bias_add(tf.matmul(A2_dropout, W3), b3)
            with tf.name_scope("relu"):
                A3 = tf.nn.relu(xw_plus_b)
            with tf.name_scope("dropout"):
                A3_dropout = tf.nn.dropout(A3, keep_prob)
        with tf.name_scope("layer-4"):
            with tf.name_scope("weight-4"):
                W4 = get_weight([200, 10])
                variable_summary(W4)
            with tf.name_scope("bias-4"):
                b4 = get_bias([10])
                variable_summary(b4)
            y_hat = tf.nn.bias_add(tf.matmul(A3_dropout, W4), b4)
    return y_hat
    
def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))

def get_bias(shape):
    return tf.Variable(tf.zeros(shape))

def backward(mnist):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 784], name="X_input")
        y = tf.placeholder(tf.float32, shape=[None, 10], name="Y_input")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    y_hat = forward(x, keep_prob)
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
        tf.summary.scalar("loss", loss)
    with tf.name_scope("accuracy"):
        equal_op = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_op, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(0.01).minimize(loss)
        
    merged = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./log/", sess.graph)
        for i in range(STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, summary = sess.run([train, merged], feed_dict={x: xs, y: ys, keep_prob: 1.0})
            writer.add_summary(summary, i)
            if i % 200 == 0:
                loss_val = sess.run(loss, feed_dict={x: xs, y: ys, keep_prob: 1.0})
                train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})
                print("After %6d steps, loss is %.6f, train accuracy %.6f, test accuracy %.6f" % (i, loss_val, train_acc, test_acc))

                
if __name__ == "__main__":
    backward(mnist)