"""
Logistic Regression for binary time series prediction

Author: Yadong Zhang

Date: 2-28-2018

"""

import tensorflow as tf
import numpy as np
from Initializer import Initializer
import pandas
import csv
# model
#

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

look_back = 8  # input size --20
hidden_cell_num = 9
output_size = 1
total_series_length = 200
fileDir = 'baiduDirection.csv'
batch_size = 10
# placeholder
xs = tf.placeholder(tf.float32, [None, look_back])
ys = tf.placeholder(tf.float32, [None, output_size])

# layers
hidden_layer = add_layer(xs, look_back, hidden_cell_num,
                         activation_function=None)
prediction = add_layer(hidden_layer, hidden_cell_num, output_size,
                       activation_function=None)

# actual binary prediction
prediction_binary = tf.round(tf.sigmoid(prediction))

# loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=prediction))

# optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# accuracy
predictions_correct = tf.cast(tf.equal(prediction_binary, ys), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)



# prepare data
dataframe = pandas.read_csv(fileDir, usecols=[0], engine='python', skipfooter=0)

dataset = dataframe.values
dataset = dataset.astype('float32')

if (len(dataset) > total_series_length):
    y = dataset[0:total_series_length]

dataX, dataY = [], []
for i in range(len(y) - look_back - 1):
    a = y[i:(i + look_back), 0]
    dataX.append(a)
    dataY.append(y[(i + look_back), 0])

x_data = (np.array(dataX))
y_data = (np.array(dataY))

print y_data[0]
train_indices = np.random.choice(len(x_data), int(len(x_data)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_data))) - set(train_indices)))
x_vals_train = x_data[train_indices]
x_vals_test = x_data[test_indices]
y_vals_train = y_data[train_indices]
y_vals_test = y_data[test_indices]

# return (np.array(dataX)).T, (np.array(dataYint))

init = tf.initialize_all_variables()
# train
with tf.Session() as sess:

    sess.run(init)
    loss_vec = []
    train_acc = []
    test_acc = []
    for i in range(32000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={xs: rand_x, ys: rand_y})

        temp_loss = sess.run(loss, feed_dict={xs: rand_x, ys: rand_y})
        loss_vec.append(temp_loss)
        temp_acc_train = sess.run(accuracy, feed_dict={xs: x_vals_train, ys: np.transpose([y_vals_train])})
        train_acc.append(temp_acc_train)
        temp_acc_test = sess.run(accuracy, feed_dict={xs: x_vals_test, ys: np.transpose([y_vals_test])})
        test_acc.append(temp_acc_test)
        if (i+1)%300==0:
            print('Loss = ' + str(temp_loss))

def saveData2File(data, fileName = "loss"):
    csvfile = file(fileName+'.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow([fileName])
    # data = [np.reshape(predictAll, (-1, 1)).T, np.reshape(xAll, (-1, 1)).T]
    data = [np.reshape(data, (-1, 1))]  # , np.reshape(RMSE_list, (-1, 1))]
    # data = np.reshape(data, (1, -1)).T
    data = np.reshape(data, (1, -1)).T
    writer.writerows(data)
    # close file
    csvfile.close()

saveData2File(train_acc, 'train_acc')
saveData2File(test_acc, 'test_acc')
saveData2File(loss_vec, 'loss_vec')
