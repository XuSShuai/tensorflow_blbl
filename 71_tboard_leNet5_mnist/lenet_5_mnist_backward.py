import tensorflow as tf
import lenet_5_mnist_forward
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

STEPS = 20000
BATCH_SIZE = 32
REGULARIZER = 0.001
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
MOVEING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_SAVE_NAME = "mnist_model"
train_num_examples = 60000

def backward(mnist):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 
                                              lenet_5_mnist_forward.IMAGE_SIZE, 
                                             lenet_5_mnist_forward.IMAGE_SIZE, 
                                             lenet_5_mnist_forward.NUM_CHANNELS])
        y = tf.placeholder(tf.float32, shape=[None, lenet_5_mnist_forward.OUTPUT_NODE])
        
    y_hat = lenet_5_mnist_forward.forward(x, True, REGULARIZER)
    
    with tf.name_scope("global_step"):
        global_step = tf.Variable(0, trainable=False)
        
    with tf.name_scope("loss"):
        loss_cem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_hat))
        loss = loss_cem + tf.add_n(tf.get_collection("losses"))
        tf.summary.scalar("loss", loss)
    
    with tf.name_scope("learning_rate"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
                                                   global_step, 
                                                   train_num_examples/BATCH_SIZE, 
                                                   LEARNING_RATE_DECAY, 
                                                   staircase=True)
    
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    
    with tf.name_scope("ema"):
        ema = tf.train.ExponentialMovingAverage(MOVEING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name="train")
            
    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar("train_accuracy", accuracy)
        
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        writer = tf.summary.FileWriter("./log/train", sess.graph)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs, 
                                    [BATCH_SIZE, 
                                     lenet_5_mnist_forward.IMAGE_SIZE, 
                                     lenet_5_mnist_forward.IMAGE_SIZE, 
                                     lenet_5_mnist_forward.NUM_CHANNELS])
            _, global_step_value, loss_value = sess.run([train_op, global_step, loss], feed_dict={x:reshape_xs, y:ys})
            if i % 1000 == 0:
                print("After %5d steps, the loss is %f" % (global_step_value, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_SAVE_NAME), global_step)
                
                train_reshape = np.reshape(mnist.train.images, [mnist.train.num_examples, 
                                                                lenet_5_mnist_forward.IMAGE_SIZE, 
                                                                lenet_5_mnist_forward.IMAGE_SIZE, 
                                                                lenet_5_mnist_forward.NUM_CHANNELS])
                summary = sess.run(merged, feed_dict={x: train_reshape, y: mnist.train.labels})
                writer.add_summary(summary, i)
                
def main():
    mnist = input_data.read_data_sets("../data", one_hot=True)
    backward(mnist)
    
if __name__ == "__main__":
    main()