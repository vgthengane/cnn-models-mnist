# Created on Fri Nov 23 20:10:25 2018
# @author: Thengane Vishal G

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.Session()
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('./temp', one_hot=True)


train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels = mnist.test.labels


batch_size = 225
learning_rate = 0.005
epochs = 5


x_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_target = tf.placeholder(tf.int32, shape=[None, 10])

def weights(shape):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))
    return weight

def biases(shape):
    bias = tf.Variable(tf.zeros(shape, dtype=tf.float32))
    return bias


def convnet(input_):
    # input_size = [batch_size, 28, 28, 1]
    wc1, bc1 = weights([5, 5, 1, 32]), biases([32])
    conv1 = tf.nn.conv2d(input_, wc1, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bc1)) # size = [batch_size, 28, 28, 32]
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # size = [batch_size, 14, 14, 32]
    wc2, bc2 = weights([5, 5, 32, 64]), biases([64])
    conv2 = tf.nn.conv2d(pool1, wc2, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bc2)) # size = [batch_size, 14, 14, 64]
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # size = [batch_size, 7, 7, 64]

    
    dim = pool2.get_shape().as_list()
    flat_dim = dim[1] * dim[2] * dim[3] # size = 7 * 7 * 64
    flat = tf.reshape(pool2, [-1, flat_dim])


    wfc1 = weights([flat_dim, 128])
    bfc1 = tf.Variable(tf.truncated_normal([128], stddev=0.1, dtype=tf.float32))
    fc1 = tf.nn.relu(tf.add(tf.matmul(flat, wfc1), bfc1))
    wfc2 = weights([128, 10])
    bfc2 = tf.Variable(tf.truncated_normal([10], stddev=0.1, dtype=tf.float32))

    final_out = tf.add(tf.matmul(fc1, wfc2), bfc2)
    return final_out


def get_accuracy(logits, targets):
    num_correct = np.sum(np.equal(np.argmax(logits, axis=1), np.argmax(targets, axis=1)))
    return num_correct


out = convnet(x_input)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_target))
prediction = tf.nn.softmax(out)
opt = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
sess.run(tf.global_variables_initializer())


# training
for epoch in range(epochs):
    x = np.arange(0, len(train_xdata))
    np.random.shuffle(x)
    n_batches = int((len(train_xdata)/batch_size))
    for index in range(n_batches):
        rand_index = x[index * batch_size:(index+1)*batch_size]
        rand_x = train_xdata[rand_index]
        rand_x = np.expand_dims(rand_x, 3)
        rand_y = train_labels[rand_index]
        train_dict = {x_input: rand_x, y_target: rand_y}
    
        _, temp_train_loss, temp_train_preds = sess.run([opt, loss, prediction], feed_dict=train_dict)
        temp_train_acc = get_accuracy(temp_train_preds, rand_y)
        acc = temp_train_acc/batch_size
        print('[Epoch: {:2d}]  [{:3d}/{:3d}]    [loss: {:.4f}]    [acc: {:.4f}]'.format((epoch + 1), 
                (index + 1), n_batches, temp_train_loss, acc))
  
print('\nOptimization finished!')  

# testing
n_batches = int(len(test_xdata) / batch_size)
total_correct_preds = 0
for i in range(n_batches):
    eval_x = test_xdata[i * batch_size:(i + 1) * batch_size]
    eval_x = np.expand_dims(eval_x, 3)
    eval_y = test_labels[i * batch_size:(i + 1) * batch_size]
    _, batch_loss, preds = sess.run([opt, loss, prediction], feed_dict={x_input: eval_x, y_target: eval_y})
    n_correct = get_accuracy(preds, eval_y)
    total_correct_preds += n_correct

print('\nAccuracy: {0}'.format(total_correct_preds / len(test_xdata)))