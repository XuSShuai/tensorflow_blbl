import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_NUM = 32
CONV2_SIZE = 5
CONV2_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10

def forward(x, train, regularizer):
    with tf.name_scope("forward"):
        with tf.name_scope("conv_1"):
            w1 = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_NUM], regularizer)
            b1 = get_bias([CONV1_NUM])
            a1 = tf.nn.relu(tf.nn.bias_add(conv2d(x, w1), b1))
            with tf.name_scope("pool_1"):
                p1 = max_pool_2x2(a1)
        with tf.name_scope("conv_2"):
            w2 = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_NUM, CONV2_NUM], regularizer)
            b2 = get_bias([CONV2_NUM])
            a2 = tf.nn.relu(tf.nn.bias_add(conv2d(p1, w2), b2))
            with tf.name_scope("pool_2"):
                p2 = max_pool_2x2(a2)
        with tf.name_scope("conv_3"):
            p2_shape = p2.get_shape().as_list()
            nodes = p2_shape[1] * p2_shape[2] * p2_shape[3]
            reshaped_p2 = tf.reshape(p2, [-1, nodes])
        with tf.name_scope("fc_1"):
            fc1_w = get_weight([nodes, FC_SIZE], regularizer)
            fc1_b = get_bias([FC_SIZE])
            a1 = tf.nn.relu(tf.matmul(reshaped_p2, fc1_w) + fc1_b)
            with tf.name_scope("dropout"):
                if train:
                    a1 = tf.nn.dropout(a1, 0.5)
        with tf.name_scope("fc_2"):
            fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
            fc2_b = get_bias([OUTPUT_NODE])
            y = tf.matmul(a1, fc2_w) + fc2_b
            return y

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")