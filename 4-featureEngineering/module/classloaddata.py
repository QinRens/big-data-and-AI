import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from scipy.stats import norm, skew
from matplotlib.lines import Line2D
from copy import deepcopy
import scipy.stats as st
from scipy.stats import norm,skew,johnsonsu,lognorm,kurtosis
sns.set_style(style='darkgrid')

class LoadData(object):
    #global targetdf_list
    #global quantitative, qualitative, df_train
    
    def __init__(self,dataset,labellist,IDlist=[]):
        self.dataset=dataset
        self.labellist=labellist
        self.IDlist=IDlist
    
    
    def __loaddata__(self,dataset,labellist,IDlist):
        global quantitative, qualitative, df_train
        df_train=pd.read_csv(dataset)
        likely_cat = {}
        for var in df_train.columns:
            likely_cat[var] = 1.*df_train[var].nunique()/df_train[var].count() < 0.05
        qualitative = [f for f in df_train.columns if likely_cat[f]==True or df_train.dtypes[f] == 'object']
        quantitative = [f for f in df_train.columns if likely_cat[f]==False and df_train.dtypes[f] != 'object']
        for i in range(len(labellist)):
            if labellist[i] in quantitative:
                quantitative.remove(labellist[i])
            if labellist[i] in qualitative:
                qualitative.remove(labellist[i])
        for i in range(len(IDlist)):
            if IDlist[i] in quantitative:
                quantitative.remove(IDlist[i])
            if IDlist[i] in qualitative:
                qualitative.remove(IDlist[i])
    
    def __gentarget_quantitative_target__(self,label):
        global targetdf_list
        targetdf_list=[]
        for i in range(len(quantitative)):
            targetdf_list.append(df_train[[quantitative[i],label]])
     
    def init_dataset(self):
        self.__loaddata__(dataset=self.dataset,labellist=self.labellist,IDlist=self.IDlist)
        self.__gentarget_quantitative_target__(self.labellist[0])