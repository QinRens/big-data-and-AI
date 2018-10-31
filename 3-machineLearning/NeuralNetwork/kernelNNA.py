from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def kernel(real_x,predict_x):
    """
    The kernel can calculate the inner product of two vectors.
    when used as train kernel, set the real_x, predict_x both as the x_data
    when used as predict kernel, set the real_x as x_data, set the predict_x as predict_x_data
    :param real_x: real x vector
    :param predict_x: predict x vector
    :return: exp kernel
    """
    # define the kernel for train and prediction
    gamma = tf.constant(-1.0)  # -1.0 the constant can be changed
    # cross of real and predict vector
    real_x_cross = tf.reshape(tf.reduce_sum(tf.square(real_x), 1), [-1, 1])
    predict_x_cross = tf.reshape(tf.reduce_sum(tf.square(predict_x), 1), [-1, 1])
    # calculate (real_x - predict_x)^2
    sq_dists = tf.add(tf.sub(real_x_cross, tf.mul(2., tf.matmul(real_x, tf.transpose(predict_x)))),
                      tf.transpose(predict_x_cross))
    # exp kernel
    my_kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))
    return my_kernel

def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    You could add layer with prior layer's output as inputs,
    And the input size and output size must be given
    :param inputs: the prior layer's output as input
    :param in_size: input size
    :param out_size: output size
    :param activation_function: the activation function of output
    :return: layers outputs
    """
    # add one more layer and return the output of this layer
    # x*W+b-->y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # activate output
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# A method to adjust learning rate in the URL as follow.
# http://wenku.baidu.com/link?url=wAYUzPORx9ubP8Z76sQCyH6BHtQivg5W3ichKuOJF2L9ur5EP7iXLjxTGcZACLxWkDobstJVL4KR1tSLZ0sZhoVe2an-inQPLW4i9kn07IK
'''
def adjustLearn(Enow,Eprior,rateprior,lamda=0.0005):
    if Eprior/Enow>1.005:#Enow<Eprior:
        ratenow=1.05*rateprior
    elif Eprior/Enow<=1.005 and Eprior/Enow>=0.995:
        ratenow=math.exp(-lamda)*rateprior
    return ratenow
'''

def adjustLearn(Enow,Eprior,rateprior):
    """
    This function can adjust the learning rate while training
    http://qing0991.blog.51cto.com/1640542/1825540
    :param Enow: now loss or energy
    :param Eprior: prior loss or energy
    :param rateprior: prior learning rate
    :return: new learning rate for next
    """
    if Enow < Eprior:
        ratenow = 1.05*rateprior
    elif Enow > 1.04*Eprior:
        ratenow = 0.7*rateprior
    else:
        ratenow = rateprior
    return ratenow

'''Prepare the periodical data'''
# Make up some real data
x_data = np.linspace(0.1, 30, 1000)[:, np.newaxis]
# Add some noise for data
noise = np.random.normal(0, 0.005, x_data.shape)#0.00005
y_data = np.sin(x_data) / x_data + noise
# y_data_2 = y_data_1+2*yipsilon
# y_data=np.append(y_data_1,y_data_2)[:, np.newaxis]
# x_data=np.append(x_data,x_data)[:, np.newaxis]
# y_data = np.sin(x_data) - 0.5 + noise
# Get the length of data
length = len(x_data)

'''Establish the model'''
# define placeholder for inputs and outputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add the kernel as input layer of model
kernel1 = kernel(xs, xs)
# if you don't want to use kernel,
# l1 = add_layer(xs, 1, 7, activation_function=tf.nn.sigmoid)
# if you wanna use kernel as input layer of the model
# Output size is 7, means that you have 7 cells
l1 = add_layer(kernel1, length, 7, activation_function=tf.nn.sigmoid)
# add output layer
prediction = add_layer(l1, 7, 1, activation_function=None)
# C=0.24874*1.5
#
C = 1
# the error or loss between prediction data and real data
loss = C*tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# Declare a learning rate placeholder for changing it while training
learning_rate = tf.placeholder(tf.float32, [])
# if you wanna use a certain learning rate
# learning_rate=tf.Variable(0.0004)

# Declare a trainer as GradientDescentOptimizer to minimize loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# initialize all variables
init = tf.initialize_all_variables()
# declare a new session
sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
# learning rate initialization
learnRate = 0.0004  # 0.0004
# save the loss
lossArr = []
# tic the time for running
start_time = time.clock()
# train the model
for i in range(10000000):
    # training
    # run the loss for changing the learning rate
    loss_temp = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
    # save the loss
    lossArr.append(loss_temp)
    if i > 100:
        learnRate = adjustLearn(lossArr[i],lossArr[i-1],learnRate)
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data, learning_rate:learnRate})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        plt.ylim(min(np.min(y_data),np.min(prediction_value)),max(np.max(y_data),np.max(prediction_value)),500) 
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        end_time=time.clock()
        plt.title('#time='+str(end_time-start_time)+'s')
        plt.pause(0.1)



