"""
The SVM model can be established and trained with this module.
While training, the regression process can be seen dynamically.
As establishment finished, you could choose to save or restore model according to its performance
"""

from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import regression as re


class SVM(object):
    """
    In this class, you could establish the SVM model.
    """
    def __init__(self, _batch_size):
        """
        When a SVM object is declared, the __init__ function will be call automatically
        :param _batch_size: decide how much data you want to train each step.
        """
        # the graph for model
        # x data placeholder
        self.x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)  # use the double data for regression
        # y target placeholder
        self.y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # x prediction placeholder
        self.prediction_xgrid = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # the parameters of SVM. if you have any questions, please refer to SVM tutorials.
        # b dot multiply Kernel plus c
        b = tf.Variable(tf.random_normal(shape=[1, _batch_size]))
        c = tf.Variable(tf.random_normal(shape=[1, _batch_size]))
        # prepare the train kernel for train process and prediction process
        train_kernel = re.kernel(self.x_data, self.x_data)
        pred_kernel = re.kernel(self.x_data, self.prediction_xgrid)
        # declare the loss function components
        model_output = tf.add(tf.matmul(b, train_kernel), c)
        model_output = tf.reshape(model_output, [-1, 1])
        # declare loss, we use lsSVM(least squares Support Vector Machine)
        error = tf.sub(model_output, self.y_target)
        self._loss = tf.reduce_sum(tf.square(error))
        # prediction error
        self._predict_output = tf.add(tf.matmul(b, pred_kernel), c)
        prediction = tf.neg(tf.sub(self._predict_output, self.y_target))
        # calculate the brief accuracy for model
        self._error = tf.reduce_mean(tf.div(prediction, self.y_target))
        # declare a placeholder for learning rate
        self.learning_rate = tf.placeholder(tf.float32, [])
        # declare the optimizer
        my_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # target: minimize the loss function declared above
        self.train_step = my_opt.minimize(self._loss)
        # declare a saver
        self.saver = tf.train.Saver()


    @property
    def predict_output(self):
        # declare the loss function components
        return self._predict_output
    @property
    def loss(self):
        return self._loss

    @property
    def error(self):
        return self._error


def TrainSVM(data, target, askShow=True, trainStep=1000,
    dirName = "data/saver/", saveName = "model"):
    """
    A function for Train the SVM model we established.
    the SVM model will be initialized in the inner part of the function
    :param data: x data
    :param target: y target
    :param askShow: if True, you could see the training process dynamically
    :param trainStep: trainstep, 1000 default
    :param dirName: model dir
    :param saveName: file Name
    :return: NULL
    """

    if len(target) == len(data):
        # create the dir
        mkpath = dirName#the model save path
        re.mkdir(mkpath)
        # saver path
        path = mkpath+saveName+".ckpt"
        # choose data whole size as batch size
        _batch_size = len(data)
        xx = data
        yy = target
        # establish the model
        SVMtest = SVM(_batch_size=_batch_size)
        # important step
        # init = tf.initialize_all_variables()
        sess = tf.Session()
        # sess.run(init)
        learnRate = 0.0004  # 0.0004
        lossArr = []
        # ask whether to restore the model
        ask_restore = raw_input('Restore the model?[Y/N]')
        if ask_restore is not 'Y':
            # important step
            init = tf.initialize_all_variables()
            sess.run(init)

        if ask_restore is 'Y':
            SVMtest.saver.restore(sess, path)

        if askShow is True:
            # plot the real data
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(xx, yy)
            plt.ion()
            plt.show()
            start_time = time.clock()
        '''train'''
        for i in range(trainStep):
            # training
            loss_temp = sess.run(SVMtest._loss, feed_dict={SVMtest.x_data: xx, SVMtest.y_target: yy})
            lossArr.append(loss_temp)
            learnRate = re.adjustLearn(lossArr[i], lossArr[i - 1], learnRate)
            sess.run(SVMtest.train_step,
                     feed_dict={SVMtest.x_data: xx, SVMtest.y_target: yy, SVMtest.learning_rate: learnRate})
            # show the process dynamically per 50 step
            if i % 50 == 0:
                # to visualize the result and improvement
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(SVMtest._predict_output,
                                            feed_dict={SVMtest.x_data: xx, SVMtest.prediction_xgrid: xx})
                accuracy = sess.run(SVMtest._error,
                                    feed_dict={SVMtest.x_data: xx, SVMtest.prediction_xgrid: xx, SVMtest.y_target:yy})
                # c_value=sess.run(c,feed_dict={x_data:xx,y_target:yy})
                # plot the prediction
                plt.ylim(min(np.min(yy), np.min(prediction_value)), max(np.max(yy), np.max(prediction_value)), 500)
                # lines = ax.plot(xx, np.transpose(prediction_value), 'r-', lw=5)
                lines = ax.plot(xx, np.transpose(prediction_value), 'r-', lw=5)
                end_time = time.clock()
                #plt.title('#time=' + str(end_time - start_time) + 's')
                #ax.set_xlabel('#accuracy=' + str(accuracy*100) + '%' + ' #learnRate=' + str(learnRate))
                plt.pause(0.5)
                ask_save = raw_input('Save the model?[Y/N]')
                if ask_save is 'Y':
                    saver_path = SVMtest.saver.save(sess, path)

    else:
        print("The length of data and target is not the same")

# test this file
if __name__ == '__main__':
    # create the data
    x_data = np.linspace(0.1, 30, 1000)[:, np.newaxis]
    # x_data2=np.linspace(15,29,600)[:,np.newaxis]
    noise = np.random.normal(0, 0.005, x_data.shape)  # 0.00005
    y_data = np.sin(x_data) / x_data + noise
    # y_data_2 = y_data_1+2*yipsilon
    # y_data=np.append(y_data_1,y_data_2)[:, np.newaxis]
    # x_data=np.append(x_data,x_data)[:, np.newaxis]
    # y_data = np.sin(x_data) - 0.5 + noise
    TrainSVM(x_data, y_data)
