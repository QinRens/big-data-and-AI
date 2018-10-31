"""
This is a keras machine learning module for stock price time series prediction.
Note: 
	You can design the structure when initialize it.
	Such as :
	----------------
	Layer 1: LSTM-5 hidden state size - relu
	Layer 2: LSTM-10 hidden state size - None
	Layer 3: Dense-1 state size - sigmoid
	----------------
	Then, you can write 'LSTM-5-relu...LSTM-10-None...Dense-1-sigmoid' as a input string

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
from keras.utils.vis_utils import plot_model

class sequenceModel(object):
	"""
	This module helps you establish a basic LSTM neural network.
	And the train function can train the network with the prepared input data.
	The predict function can predict with the testX input data
	"""
	def __init__(self, struc, input_shape, targetType):
		'''
		Example:
			struc = 'LSTM-5-relu...LSTM-1-None...Dense-1-relu'
			input_shape = (10, 2)
			targetType = boolean
			-->
			layerStructure:
				[{'activation': 'relu', 'hiddenState': 5, 'input_shape': (10, 2), 'layerName': 'LSTM', 'return_sequences': True},
 				 {'activation': None, 'hiddenState': 1, 'layerName': 'LSTM', 'return_sequences': False},
 				 {'activation': 'relu', 'hiddenState': 1, 'layerName': 'Dense'}]
 			loss = 'binary_crossentropy'
		'''
		layerStructure = structureDecode(struc, input_shape)
		# Establish the model
		self.model = Sequential()
		for index, layerItem in enumerate(layerStructure):
			modelLayer(self.model, layerItem)

		# Choose the loss function according to the target type
		if targetType is 'boolean':
		    loss = 'binary_crossentropy'  # ''mean_squared_error'
		else:
		    loss = 'mean_squared_error'
		optimizer='adam'
		self.model.compile(loss=loss, optimizer=optimizer)

	def train(self, trainX, trainY, validation_data, callbacksFun=None, nb_epoch=10, batch_size=300, verbose=2):
		# train the model with trainX and trainY.
		# valiadate the model with validation_data
		self.model.fit(trainX, trainY, validation_data=validation_data, epochs=nb_epoch, batch_size=batch_size, verbose=verbose, callbacks=callbacksFun)

	def predict(self, testX):
		# predict testX
		predictY = self.model.predict(testX)
		return predictY

	def plotModel(self, filename):
		# plot the structure of model
		plot_model(self.model, to_file=filename, show_shapes=True)

	def saveModel(self, filename):
		# save the model parameters in to filename
		self.model.save(filename)

	def loadModel(self, filename):
		# load the model
		self.model.loadModel(filename)

def modelLayer(model, args):
	'''
	Add layer to model, according to the args.
	Return: None
	'''
	if args['layerName'] == 'LSTM':
		if args.has_key('input_shape'):
			model.add(LSTM(args['hiddenState'], input_shape=args['input_shape'], 
				return_sequences=args['return_sequences'], activation=args['activation']))
		else:
			model.add(LSTM(args['hiddenState'],
				return_sequences=args['return_sequences'], activation=args['activation']))
	else:
		if args['layerName'] == 'Dense':
			if args.has_key('input_shape'):
				model.add(Dense(args['hiddenState'], input_dim=args['input_shape'], 
					activation=args['activation']))
			else:
				model.add(Dense(args['hiddenState'], activation=args['activation']))

def structureDecode(struc, input_shape):
	'''
	Example:
		struc = 'LSTM-5-relu...LSTM-1-None...Dense-1-relu'
		input_shape = (10, 2)
		-->
		layerStructure:
			[{'activation': 'relu', 'hiddenState': 5, 'input_shape': (10, 2), 'layerName': 'LSTM', 'return_sequences': True},
				 {'activation': None, 'hiddenState': 1, 'layerName': 'LSTM', 'return_sequences': False},
				 {'activation': 'relu', 'hiddenState': 1, 'layerName': 'Dense'}]
	'''
	layerStructure = []
	try:
		strucList = struc.split('...')
		for index, layerItem in enumerate(strucList):
			layerName, hiddenState, activateFun = layerItem.split('-')
			if activateFun == 'None':
				activateFun = None
			setupDict = {'layerName': layerName, 'hiddenState': int(hiddenState), 'activation':activateFun}
			if index is 0:
				setupDict.update({'input_shape': input_shape})
			try:
				nextlayerName, _, _ = strucList[index+1].split('-')
				if nextlayerName == 'LSTM':
					setupDict.update({'return_sequences': True})
				else:
					setupDict.update({'return_sequences': False})
			except:
				pass
			layerStructure.append(setupDict)
		return layerStructure
	except:
		raise('The format of parameters struc must be layerName-hiddenState-activateFun...')

