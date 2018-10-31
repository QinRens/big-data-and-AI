"""
This is a simple keras LSTM module for stock price time series prediction.

Author: Yadong Zhang
Date: 6-10-2018
E-mail: ydup@foxmail.com

"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import matplotlib.pyplot as plt
from pandas import Series

class LSTMmodule(object):
	"""
	This module helps you establish a basic LSTM neural network.
	And the train function can train the network with the prepared input data.
	The predict function can predict with the testX input data
	"""
	def __init__(self, look_back, look_forward, featureName, targetName):
		self.model = Sequential()
		# self.model = Sequential()
		self.model.add(LSTM(10, input_shape=(look_back, len(featureName)), return_sequences=False))  # , return_sequences=True))
		self.model.add(Dense(look_forward))
		self.model.add(Activation('relu'))  # Activation('relu'))
		if targetName is 'diffLogi':
		    loss = 'binary_crossentropy'  # ''mean_squared_error'
		else:
		    loss = 'mean_squared_error'
		optimizer='adam'
		self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	def train(self, trainX, trainY, nb_epoch=10, batch_size=300, verbose=2):
		self.model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)

	def predict(self, testX):
		predictY = self.model.predict(testX)
		return predictY
