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

class SingleEDA(object):
    
    def __init__(self,dataset):
        self.dataset=dataset
        
    def distplot(self,feature,fitmethod='norm',show_QQplot=False,show_skew=False,show_kurt=False,color=None,kde=True,rug=False):
        '''
        for numerical features
        '''
        if self.dataset[feature].dtypes == 'object':
            raise ValueError('input feature should be numeirical')
            
        else:
            if locals()['fitmethod']=='norm':
                sns.distplot(self.dataset[feature] , fit=norm,color=color,kde=kde,rug=rug)
            
            elif locals()['fitmethod']=='johnsonsu':
                sns.distplot(self.dataset[feature] , fit=johnsonsu, color=color,kde=kde,rug=rug)
            
            elif locals()['fitmethod']=='lognorm':
                sns.distplot(self.dataset[feature] , fit=lognorm,color=color,kde=kde,rug=rug)
            
            elif locals()['fitmethod']==None:
                sns.distplot(self.dataset[feature] , fit=None,color=color,kde=kde,rug=rug)
            
            else:
                print "we don't have this method"
        
            #Now plot the distribution
            (mu, sigma) = norm.fit(self.dataset[feature])
            plt.legend(['Fit Line. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
            plt.ylabel('Frequency')
            plt.title('%s distribution'%(feature))
            plt.show()
        
            if show_skew:
                sns.distplot(self.dataset.skew(),axlabel ='Skewness',color=color,kde=kde,rug=rug)
                plt.show()
            
            if show_kurt:
                sns.distplot(df_train.kurt(),axlabel ='Kurtosis',color=color,kde=kde,rug=rug)
                plt.show()

            #Get also the QQ-plot
            if show_QQplot:
                fig = plt.figure()
                res = stats.probplot(self.dataset[feature], plot=plt)
                plt.show()
            
    def countplot(self,feature,show_plot=True):
        '''
        for categorial features
        '''
        if show_plot:
            plt.figure(figsize = (12, 6))
            sns.countplot(x = feature, data = self.dataset)
            xt = plt.xticks(rotation=45)
            plt.show()
        return self.dataset[feature].value_counts()
            
    def show_skewness(self,show_plot=True):
        '''
        for numerical features
        '''
        skewness=self.dataset.skew()
        skewness=skewness.sort_values(ascending=False)
        skewness.name='skewness'
        if show_plot:
            f, ax = plt.subplots(figsize=(12, 9))
            plt.xticks(rotation='90')
            sns.barplot(x=skewness.index, y=skewness.values)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Skewness', fontsize=15)
            plt.title('The skewness of features', fontsize=15)
            plt.show()
        return skewness
    
    def show_kurtosis(self,show_plot=True):
        '''
        for numerical features
        '''
        kurtosis=self.dataset.kurt()
        kurtosis=kurtosis.sort_values(ascending=False)
        kurtosis.name='kurtosis'
        if show_plot:
            f, ax = plt.subplots(figsize=(12, 9))
            plt.xticks(rotation='90')
            sns.barplot(x=kurtosis.index, y=kurtosis.values)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Kurtosis', fontsize=15)
            plt.title('The kurtosis of features', fontsize=15)
            plt.show()
        return kurtosis
    
    def show_describe(self,feature,show_plot=True):
        '''
        for numerical features
        '''
        if self.dataset[feature].dtypes == 'object':
            raise ValueError('input feature should be numerical')
            
        else:
            desc=self.dataset[feature].describe()
            desc.name='describe'
            if show_plot:
                f, ax = plt.subplots(figsize=(12, 9))
                plt.xticks(rotation='90')
                sns.barplot(x=desc.index, y=desc.values)
                plt.xlabel('Statistics', fontsize=15)
                plt.ylabel('Value', fontsize=15)
                plt.title('The Statistics of feature %s'%(feature), fontsize=15)
                plt.show()
            return desc