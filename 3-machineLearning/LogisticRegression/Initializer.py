"""
This module is used for data initialization

Author: yadong ZHANG

E-mail: ydup@foxmail.com

Date: 12-29-2017

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

class Initializer(object):

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # normalization scaler

    def generateData(self, total_series_length=4171, fileDir="../../../data/tmp/sp500.csv"):
        # type: (object) -> object

        # import data from the directory "../..data/tmp/"
        dataframe = pandas.read_csv(fileDir, usecols=[0], engine='python', skipfooter=0)

        dataset = dataframe.values
        dataset = dataset.astype('float32')
        # normalize the data with the scaler
        dataset = self.scaler.fit_transform(dataset)

        if (len(dataset) > total_series_length):
            y = dataset[0:total_series_length]
            # x = x.reshape((truncated_backprop_len, -1))  # The first index changing slowest, subseries as rows
            # y = y.reshape((truncated_backprop_len, -1))
            return y
        else:
            pass

    def create_dataset(self, dataset, input_size=1):
        """
        create time series data frame
        :param dataset: data input
        :param input_size: have been explained in the class
        :return: numpy.array(dataX), numpy.array(dataY)
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - input_size - 1):
            a = dataset[i:(i + input_size), 0]
            dataX.append(a)
            dataY.append(dataset[i + input_size, 0])
        self.x_data = (np.array(dataX)).T
        self.y_data = (np.array(dataY)).T

        return (np.array(dataX)).T, (np.array(dataY))

    def mkdir(self,path):
        '''create the path'''
        path = path.strip()
        path = path.rstrip("/")
        isExists = os.path.exists(path)
        if not isExists:
            # not exist--> create
            print('#' + path + '# was created the dir successfully')
            os.makedirs(path)
            return True
        else:
            # exist --> not create
            print('#' + path + '# dir already exists')
            return False

def readCommandArg(Config):

    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("--truncated", "-tbl", help="set truncated_backprop_length", type=int)

    parser.add_argument("--epoch", "-e", help="set epoch", type=int)

    parser.add_argument("--stateSize", "-s", help="set state size", type=int)

    parser.add_argument("--trainPercentage", "-tp", help="set trainPercentage", type=float)

    # read arguments from the command line
    args = parser.parse_args()

    # check for --width
    if args.truncated:
        Config.truncated_backprop_len = args.truncated
    if args.epoch:
        Config.epoch = args.epoch
    if args.stateSize:
        Config.state_size = args.stateSize
    if args.trainPercentage:
        Config.Train_Percentage = args.trainPercentage
    return Config
