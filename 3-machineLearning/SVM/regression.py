from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os



def adjustLearn(Enow, Eprior, rateprior):
    # type: (float, float, float) -> float
    if Enow < Eprior:  # Enow<Eprior:
        ratenow = 1.05 * rateprior
    elif Enow > 1.04 * Eprior:
        ratenow = 0.7 * rateprior
    else:
        ratenow = rateprior
    return ratenow

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

def mkdir(path):
    '''create the path'''
    path=path.strip()
    path=path.rstrip("/")
    isExists=os.path.exists(path)
    if not isExists:
        #not exist--> create
        print('#'+path+'# was created the dir successfully')
        os.makedirs(path)
        return True
    else:
        # exist --> not create
        print('#'+path+'# dir already exists')
        return False