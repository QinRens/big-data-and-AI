from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time

'''
please refer to the kenelNNA.py file
'''
def kernel(real_x, predict_x):
    # define the kernel for train and prediction
    '''
    when used as train kernel, set the real_x,predict_x both as the x_data
    when used as predict kernel, set the real_x as x_data, set the predict_x as predict_x_data
    '''
    gamma = tf.constant(-1.0)  # -1.0
    real_x_cross = tf.reshape(tf.reduce_sum(tf.square(real_x), 1), [-1, 1])
    predict_x_cross = tf.reshape(tf.reduce_sum(tf.square(predict_x), 1), [-1, 1])
    sq_dists = tf.add(tf.sub(real_x_cross, tf.mul(2., tf.matmul(real_x, tf.transpose(predict_x)))),
                      tf.transpose(predict_x_cross))
    my_kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))
    return my_kernel


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# http://wenku.baidu.com/link?url=wAYUzPORx9ubP8Z76sQCyH6BHtQivg5W3ichKuOJF2L9ur5EP7iXLjxTGcZACLxWkDobstJVL4KR1tSLZ0sZhoVe2an-inQPLW4i9kn07IK
'''
def adjustLearn(Enow,Eprior,rateprior,lamda=0.0005):
    if Eprior/Enow>1.005:#Enow<Eprior:
        ratenow=1.05*rateprior
    elif Eprior/Enow<=1.005 and Eprior/Enow>=0.995:
        ratenow=math.exp(-lamda)*rateprior
    return ratenow
'''


# http://qing0991.blog.51cto.com/1640542/1825540
def adjustLearn(Enow, Eprior, rateprior):
    if Enow < Eprior:  # Enow<Eprior:
        ratenow = 1.05 * rateprior
    elif Enow > 1.04 * Eprior:
        ratenow = 0.7 * rateprior
    else:
        ratenow = rateprior
    return ratenow


# Make up some real data
# yipsilon=0.01
x_data = np.linspace(0.1, 30, 1000)[:, np.newaxis]
noise = np.random.normal(0, 0.005, x_data.shape)  # 0.00005
y_data = np.sin(x_data) / x_data + noise
# y_data_2 = y_data_1+2*yipsilon
# y_data=np.append(y_data_1,y_data_2)[:, np.newaxis]
# x_data=np.append(x_data,x_data)[:, np.newaxis]
# y_data = np.sin(x_data) - 0.5 + noise
length = len(x_data)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
kernel1 = kernel(xs, xs)
# l1 = add_layer(xs, 1, 7, activation_function=tf.nn.sigmoid)

# changing NN cell number
"""
TODO: Can we change the cell number of hidden layer?
"""

l1 = add_layer(kernel1, length, 7, activation_function=tf.nn.sigmoid)
# add output layer#7
prediction = add_layer(l1, 7, 1, activation_function=None)
# C=0.24874*1.5
C = 1
# the error between prediciton and real data
loss = C * tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

learning_rate = tf.placeholder(tf.float32, [])
# learning_rate=tf.Variable(0.0004)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
learnRate = 0.0004  # 0.0004
lossArr = []

start_time = time.clock()

for i in range(10000000):
    # training
    loss_temp = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
    lossArr.append(loss_temp)
    if i > 100:
        learnRate = adjustLearn(lossArr[i], lossArr[i - 1], learnRate)
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data, learning_rate: learnRate})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        plt.ylim(min(np.min(y_data), np.min(prediction_value)), max(np.max(y_data), np.max(prediction_value)), 500)
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        end_time = time.clock()
        plt.title('#time=' + str(end_time - start_time) + 's')
        plt.pause(0.1)
