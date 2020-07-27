
# coding: utf-8
import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("F:\work\FPGA_acc_cnn\mnist_data", one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder('float', shape=[None, 784])
    y_ = tf.placeholder('float', shape=[None, 10])
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# CNN1
with tf.name_scope("CNN_1"):
    W_conv1 = weight_variable([3,3,1,16])
    b_conv1 = bias_variable([16])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
# CNN2
with tf.name_scope("CNN_2"):
    W_conv2 = weight_variable([3,3,16,32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
# FCN1
with tf.name_scope("FCN_1"):
    W_fc1 = weight_variable([7*7*32, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
# drop_out
with tf.name_scope("Dropout"):
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
    
# FCN2
with tf.name_scope("FCN_2"):
    W_fc2 = weight_variable([128,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope("Loss"):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    
with tf.name_scope("Train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.global_variables_initializer().run()

time0 = time.time()
for i in range(10000):
    batch = mnist_data.train.next_batch(50)
    if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("Step %d, training accuracy %g" %(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

time1 = time.time()
time_cost0 = time1-time0
print("The training cost time is %f s"%(time_cost0))

print("test accuracy %g" %accuracy.eval(feed_dict={x:mnist_data.test.images, y_:mnist_data.test.labels, keep_prob:1.0}))

time2 = time.time()
time_cost1 = time2-time1
print("The test cost time is %f s"%(time_cost1))

def Record_Tensor(tensor, name):
    print("Recording tensor " + name + "...")
    f = open('F:\work\FPGA_acc_cnn\CNN_software\data/' + name + '.dat', 'w')
    array = tensor.eval()
    print ("The range: ["+str(np.min(array))+":"+str(np.max(array))+"]")
    if(np.size(np.shape(array)) == 1):
        Record_Array1D(array, name, f)
    elif(np.size(np.shape(array)) == 2):
        Record_Array2D(array, name, f)
    elif(np.size(np.shape(array)) == 3):
        Record_Array3D(array, name, f)
    elif(np.size(np.shape(array)) == 4):
        Record_Array4D(array, name, f)
    f.close()
    
def Record_Array1D(array, name, f):
    for i in range(np.shape(array)[0]):
        f.write(str(array[i]) + '\n')
        
def Record_Array2D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            f.write(str(array[i][j]) + '\n')
            
def Record_Array3D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                f.write(str(array[i][j][k]) + '\n')
                
def Record_Array4D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                for l in range(np.shape(array)[3]):
                    f.write(str(array[i][j][k][l]) + '\n')

Record_Tensor(W_conv1,"W_conv1")
Record_Tensor(b_conv1,"b_conv1")
Record_Tensor(W_conv2,"W_conv2")
Record_Tensor(b_conv2,"b_conv2")
Record_Tensor(W_fc1,"W_fc1")
Record_Tensor(b_fc1,"b_fc1")
Record_Tensor(W_fc2,"W_fc2")
Record_Tensor(b_fc2,"b_fc2")
sess.close()

