import pandas as pd
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt

def scalerModelData(dataframe, tran=True, columns=['Adj Close']):
    """
    This function will make the transformation for the dataframe
    :param dataframe: 
    :param inv: inverse=True: transformation, if Flase: inverse_transformation
    :return: dataframe
    """
    newFrame = dataframe.copy()
    mapArr = [[min(newFrame[name].values), max(newFrame[name].values)] for name in columns]
    maper = pd.DataFrame(np.transpose(mapArr), columns=columns)
    if tran is True:
        for name in columns:
            newFrame.loc[:, name] = scaler(dataframe[name].values, maper[name][0],
                                                  maper[name][1])
    return newFrame

def scaler(data, mapmin, mapmax):
    """
    The scaler for the data.
    :param data: numpy or pandas data
    :param mapmin: min of the map
    :param mapmax: max of the map
    :return: data
    """
    return (data - mapmin)/(mapmax - mapmin)

def create_dataset(dataset, look_back=1, look_forward=1, columns=['diff', 'neu', 'pos', 'neg', 'count', 'compound'], outputName='diff'):
    """
    create time series data frame
    :param dataset: data input
    :param look_back: have been explained in the class
    :return: numpy.array(dataX), numpy.array(dataY)
    """
    dataX, dataY, dateY = [], [], []
    dateIndex = dataset.index
    inputData = dataset[columns]
    inputData = inputData.values
    outputData = dataset[outputName]
    outputData = outputData.values
    # print dataset
    for i in range(len(dataset) - look_back - look_forward):
        if len(columns)==1:
            input = inputData[i:(i + look_back)]
        else:
            input = inputData[i:(i + look_back), 0: len(columns)]
        dataX.append(np.reshape(input, (-1, len(columns))))
        dataY.append(outputData[i + look_back: i + look_back + look_forward])
        dateY.append(dateIndex[i + look_back: i + look_back + look_forward])
    return np.array(dataX), np.array(dataY), np.array(dateY)

def evaluate(ideal, predict, targetName, maxDiff=0, minDiff=0):
    """
    :param ideal:
    :param predict:
    :param targetName:
    :param maxDiff: maxDiff = max(dataframe['diff'].values)
    :param minDiff: minDiff = min(dataframe['diff'].values)
    :return: None
    """
    ideal = np.squeeze(ideal)
    predict = np.squeeze(predict)
    if targetName is 'diffLogi':
        predict = np.round(predict)
        trendAcc = sum(predict == ideal) / float(len(ideal)) * 100
        print 'Accuracy: ', float(trendAcc), '%'
    else:
        if targetName is 'diff':
            sumup(ideal, predict, maxDiff, minDiff, filename='sumUpTrain')
        else:
            return Accuracy(np.squeeze(predict), np.squeeze(ideal))

def sumup(ideal, predict, maxDiff, minDiff, filename='sumUp'):
    # if the target data is difference data, we sum up the diff data to get the real stock price
    # maxDiff = max(dataframe['diff'].values)
    # minDiff = min(dataframe['diff'].values)
    unScalerPredict = [d1 * (maxDiff - minDiff) + minDiff for d1 in np.squeeze(predict)]
    unScalerIdeal = [d2 * (maxDiff - minDiff) + minDiff for d2 in np.squeeze(ideal)]
    sumUpPredict = []
    sumUpIdeal = []
    for index, values in enumerate(np.squeeze(predict)):
        sumUpIdeal.append(np.sum(unScalerIdeal[0:index]))
        sumUpPredict.append(np.sum(unScalerPredict[0:index]))
    Accuracy(np.squeeze(sumUpPredict), np.squeeze(sumUpIdeal))
    fig = plt.figure()
    plt.plot(sumUpPredict)
    plt.plot(sumUpIdeal)
    plt.legend(['predict', 'ideal'])
    fig.savefig(filename)
    
def Accuracy(ideal, predict):
    """
    Calculate the trend accuracy, RMSE, and correlation R, and print the value
    :return: none
    """
    if len(ideal) > 2:
        diffIdeal = ideal[1:] - ideal[0:len(ideal)-1]
        diffPredict = predict[1:] - predict[0:len(ideal)-1]
        trendIdeal = diffIdeal > 0
        trendPredict = diffPredict > 0
        testYSeries = Series(np.squeeze(ideal))
        predictYSeries = Series(np.squeeze(predict))
        trendAcc = sum(trendIdeal == trendPredict) / float(len(trendIdeal)) * 100
        RMSE = np.sqrt(np.mean((np.squeeze(ideal)-np.squeeze(predict))**2))
        corr = testYSeries.corr(predictYSeries)
        print 'Trend accuracy = ', trendAcc, '%'
        print 'RMSE = ', RMSE
        print 'Correlation, R:', corr
        return trendAcc, RMSE, corr
    else:
        print 'The length of data must be more than 2'

def draw(ideal, predict, filename):
    # draw the curve of ideal data and prediction data
    fig = plt.figure(figsize = (14, 5))
    plt.scatter(range(len(ideal)), ideal, 0.3)
    plt.scatter(range(len(predict)), np.squeeze(predict), 0.3)
    plt.legend(['ideal', 'predict'])
    fig.savefig(filename, dpi=300)


def mkdir(path):
    # create the path
    import os
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)

    if not isExists:
        os.makedirs(path) 
        print path+' is created!'
        return True
    else:
        print path+' already exists!'
        return False
 



