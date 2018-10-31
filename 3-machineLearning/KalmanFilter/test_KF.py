import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# prediction step
def kf_predict(X, P, A, Q, B, U):
    """
    Predict X and P
    :param X: state space vector
    :param P: possibility
    :param A: state space matrix
    :param Q: process noise
    :param B: input matrix
    :param U: input vector
    :return: X and P
    """
    X = tf.matmul(A, X) + tf.matmul(B, U)
    P = tf.matmul(A, tf.matmul(P, tf.transpose(A))) + Q
    return X, P

# Update step
# combine the update and prediction
def kf_predict_update(X, P, A, Q, B, U, Y, H, R):
    """
    Kalman Estimation
    :param Y: output vector
    :param H: output matrix
    :param R: measurement noise
    :return: X, P
    """
    X = tf.matmul(A, X) + tf.matmul(B, U)
    P = tf.matmul(A, tf.matmul(P, tf.transpose(A))) + Q
    IM = tf.matmul(H, X)  # 4*1
    IS = R + tf.matmul(H, tf.matmul(P, tf.transpose(H)))
    K = tf.matmul(P, tf.matmul(tf.transpose(H), tf.matrix_inverse(IS)))
    X = X + tf.matmul(K, Y-IM)
    P = P - tf.matmul(K, tf.matmul(IS, tf.transpose(K)))
    # LH = gauss_pdf(Y, IM, IS)
    return X, P  # IM, IS, LH

def gauss_pdf(X, M, S):

    pass

'''initial the parameters of model input'''
dt = 0.1

# state matrix
X = np.array([[0.0], [0.0], [0.1], [0.1]])  # 4*1
A = np.array([[1., 0, dt, 0], [0, 1., 0, dt],
     [0, 0, 1., 0], [0, 0, 0, 1.]])  # 4*4
Q = np.eye(4)
B = np.eye(4)
P = np.diag([0.1, 0.1, 0.1, 0.1])
U = np.zeros((4, 1))

# measurement matrix
Y = np.array([[X[0, 0] + abs(np.random.randn(1)[0])],
              [X[1, 0] + abs(np.random.randn(1)[0])]])

H = np.array([[1., 0, 0, 0],
              [0, 1., 0, 0]])
R = np.eye(2)

N_iter = 50

'''establish the model session'''
# state
X_tensor = tf.Variable(X, dtype=tf.float32)
A_tensor = tf.constant(A, dtype=tf.float32)
B_tensor = tf.constant(B, dtype=tf.float32)
# input
U_tensor = tf.constant(U, dtype=tf.float32)
# noise
Q_tensor = tf.Variable(tf.eye(4, dtype=tf.float32))
# output
# placeholder
Y_tensor = tf.placeholder(shape=[2, 1], dtype=tf.float32)
H_tensor = tf.constant(H, dtype=tf.float32)
# noise
R_tensor = tf.eye(2, dtype=tf.float32)  # tf.eye(2, dtype=tf.float32)
P_tensor = tf.Variable(P, dtype=tf.float32)  # tf.diag([0.1, 0.1, 0.1, 0.1])

X_tensor_t, P_tensor_t = kf_predict_update(X_tensor, P_tensor, A_tensor, Q_tensor, B_tensor, U_tensor, Y_tensor, H_tensor, R_tensor)

# (X_tensor, P_tensor) = kf_predict(X_tensor, P_tensor, A_tensor, Q_tensor, B_tensor, U_tensor)
# assign step
step = tf.group(
    X_tensor.assign(X_tensor_t), P_tensor.assign(P_tensor_t)
)

'''run the model'''
N = 50

output = []
estimation = []
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(N):
        sess.run(step, feed_dict={Y_tensor: Y})
        X = sess.run(X_tensor, feed_dict={Y_tensor: Y})
        # estimation
        # X[0,0],X[1,0]
        estimation.append([X[0,0],X[1,0]])
        # measurement
        Y = np.array([[X[0, 0] + abs(np.random.randn(1)[0])],
                      [X[1, 0] + abs(np.random.randn(1)[0])]])
        # P = sess.run(P_tensor, feed_dict={Y_tensor: Y})
        # print P
        output.append(Y)

'''draw the result'''
# print output
y_1 = [y[0] for y in output]
y_2 = [y[1] for y in output]
esti_1 = [e[0] for e in estimation]
esti_2 = [e[1] for e in estimation]
plt.scatter(y_1, y_2, label='Measurement')
plt.plot(esti_1, esti_2, label='Estimation')
#            plt.plot(trainPredictPlot, label='TrainPredict')
#            plt.plot(testPredictPlot, label='TestPredict')
plt.legend(loc='upper left')
plt.title('Kalman Filter')
plt.xlabel('y_1')
plt.ylabel('y_2')
# plt.show()
plt.savefig('Kalman Filter.png')

