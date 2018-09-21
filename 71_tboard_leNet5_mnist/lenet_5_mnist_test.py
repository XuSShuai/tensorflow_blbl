import lenet_5_mnist_backward
import lenet_5_mnist_forward
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np

TEST_INTERVEL_SEC = 10
test_num_examples = 10000

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [mnist.test.num_examples, 
                                        lenet_5_mnist_forward.IMAGE_SIZE, 
                                        lenet_5_mnist_forward.IMAGE_SIZE, 
                                        lenet_5_mnist_forward.NUM_CHANNELS])
        y = tf.placeholder(tf.float32, [None, lenet_5_mnist_forward.OUTPUT_NODE])
        
        y_hat = lenet_5_mnist_forward.forward(x, False, None)
        
        with tf.name_scope("accuracy"):
            correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar("test_accuracy", accuracy)
        
        ema = tf.train.ExponentialMovingAverage(lenet_5_mnist_backward.MOVEING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        merged = tf.summary.merge_all()
        
        writer = tf.summary.FileWriter("./log/test/")
        
        counter = 0
        while True:
            with tf.Session() as sess:
                
                ckpt = tf.train.get_checkpoint_state(lenet_5_mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    
                    reshape_x = np.reshape(mnist.test.images, [mnist.test.num_examples, 
                                        lenet_5_mnist_forward.IMAGE_SIZE, 
                                        lenet_5_mnist_forward.IMAGE_SIZE, 
                                        lenet_5_mnist_forward.NUM_CHANNELS])
                    summary, accuracy_value = sess.run([merged, accuracy], feed_dict={x:reshape_x, y:mnist.test.labels})
                    writer.add_summary(summary, counter)
                    counter += 1
                    
                    print("After training %s steps, the accuracy in test set is %f" % (global_step, accuracy_value))
                else:
                    print("model checkpoint path is not found")
                    return
            time.sleep(TEST_INTERVEL_SEC)
            
def main():
    mnist = input_data.read_data_sets("../data/", one_hot=True)
    test(mnist)
    
if __name__ == "__main__":
    main()