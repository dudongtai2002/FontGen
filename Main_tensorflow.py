# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx

Main function
"""

#%%

from Font import *
from utility import *
import tensorflow as tf

basis_size = 50
training_size = 1000
testing_size = 50
generateLetterSets(training_size, testing_size)

inputFont = Font(basis_size,'simsun.ttf') 

trainInput, testInput = inputFont.getLetterSets()

outputFont = Font(basis_size,'msyhbd.ttf') 

trainOutput, testOutput = outputFont.getLetterSets()


trainInput = np.reshape(trainInput, (basis_size**2 , training_size))
testInput = np.reshape(testInput, (basis_size**2, testing_size))
trainOutput = np.reshape(trainOutput, (basis_size**2, training_size))
testOutput = np.reshape(testOutput, (basis_size**2, testing_size))

trainInput = np.transpose(trainInput)
testInput = np.transpose(testInput)
trainOutput = np.transpose(trainOutput)
testOutput = np.transpose(testOutput)


#%% deep neural network
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, basis_size**2]) # input layer
y_ = tf.placeholder(tf.float32, shape=[None, basis_size**2]) # output layer
#w weight initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
# convolution  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# max pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# first convolutional layer, 5X5 patch size, 32 feature number
layer1_feature = 32

W_conv1 = weight_variable([5, 5, 1, layer1_feature])
b_conv1 = bias_variable([layer1_feature])

x_image = tf.reshape(x, [-1,basis_size,basis_size,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution layer
layer2_feature = 64
W_conv2 = weight_variable([5, 5, layer1_feature, layer2_feature ])
b_conv2 = bias_variable([layer2_feature])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
layer3_feature = 1024
W_fc1 = weight_variable([625 * 625 * layer2_feature, layer3_feature])
b_fc1 = bias_variable([layer3_feature])

h_pool2_flat = tf.reshape(h_pool2, [-1, 625*625*layer2_feature])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([layer3_feature, basis_size**2])
b_fc2 = bias_variable([basis_size**2])

y_conv=tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  #batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:trainInput[0:10,:], y_: trainOutput[0:10,:], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: trainInput[0:10,:], y_: trainInput[0:10,:], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testInput[0:50,:], y_: testOutput[0:50,:], keep_prob: 1.0}))
