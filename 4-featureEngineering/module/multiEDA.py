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

class multiEDA(object):
    
    def __init__(self,dataset,xlabel,ylabel):
        self.dataset=dataset
        self.xlabel=xlabel
        self.ylabel=ylabel
        
    def scatterplot(self,reg=True,color=None,logx=False,robust=False):
        '''
        only support numerical value
        '''
        data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
        sns.regplot(x=self.xlabel, y=self.ylabel,data=data,logx=logx,robust=robust,fit_reg=reg,color=color)
        x=plt.xticks(rotation=45)
        plt.show()
               
    def jointplot(self,kind='scatter',color='m',add_scatter=False):
        '''
        for numerical features
        kind :{ scatter | reg | resid | kde | hex } optional
        '''
        if locals()['kind']=='kde':
            if locals()['add_scatter']==True:
                data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
                g=sns.jointplot(x=self.xlabel, y=self.ylabel,kind=kind,color=color,data=data)
                g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
                g.ax_joint.collections[0].set_alpha(0)
                x=plt.xticks(rotation=45)
                plt.show()
            else:
                data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
                sns.jointplot(x=self.xlabel, y=self.ylabel,kind=kind,color=color,data=data)
                x=plt.xticks(rotation=45)
                plt.show()
        else:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
            sns.jointplot(x=self.xlabel, y=self.ylabel,kind=kind,color=color,data=data)
            x=plt.xticks(rotation=45)
            plt.show()         
    
        
    def boxplot(self,hue=None):
        '''
        for categorial features
        3 features visualization
        '''
        if locals()['hue']==None:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
        else:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[hue]],axis=1)
        sns.boxplot(x=self.xlabel, y=self.ylabel,hue=hue,data=data)
        x=plt.xticks(rotation=45)
        plt.show()
        
    def violinplot(self,hue=None):
        '''
        for categorial features
        3 features visualization
        '''
        if locals()['hue']==None:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
        else:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[hue]],axis=1)
        sns.violinplot(x=self.xlabel, y=self.ylabel,hue=hue,data=data)
        x=plt.xticks(rotation=45)
        plt.show()
        
    def swarmplot(self,hue=None):
        '''
        for categorial features
        3 features visualization
        '''
        if locals()['hue']==None:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
        else:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[hue]],axis=1)
        sns.swarmplot(x=self.xlabel, y=self.ylabel,hue=hue,data=data);
        x=plt.xticks(rotation=45)
        plt.show()
        
    def pointplot(self,hue=None):
        '''
        for categorial features
        3 features visualization
        '''
        if locals()['hue']==None:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
        else:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[hue]],axis=1)
        sns.pointplot(x=self.xlabel, y=self.ylabel, hue=hue, data=data)
        x=plt.xticks(rotation=45)
        plt.show()
        
    def stripplot(self,hue=None):
        '''
        for categorial features
        3 features visualization
        '''
        if locals()['hue']==None:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
        else:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[hue]],axis=1)
        sns.stripplot(x=self.xlabel, y=self.ylabel, hue=hue, data=data)
        x=plt.xticks(rotation=45)
        plt.show()
        
    def barplot(self,hue=None):
        '''
        for categorial features
        3 features visualization
        '''
        if locals()['hue']==None:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
        else:
            data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[hue]],axis=1)
        sns.barplot(x=self.xlabel, y=self.ylabel, hue=hue, data=data)
        x=plt.xticks(rotation=45)
        plt.show()
           
        
    def factorplot(self,color=None,hue='Street',col='RoofStyle',kind='point'):
        '''
        for categorial features
        4 features visualization
        kind parameter, including box plots, violin plots, bar plots, or strip plots.
        '''
        if locals()['hue']==None:
            if locals()['col']==None:
                data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel]],axis=1)
            else:
                data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[col]],axis=1)
                
        else:
            if locals()['col']==None:
                data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[hue]],axis=1)
            else:
                data=pd.concat([self.dataset[self.xlabel],self.dataset[self.ylabel],self.dataset[col],self.dataset[hue]],axis=1)
                
        sns.factorplot(self.xlabel, self.ylabel, data = data, color = color,hue=hue,col=col,kind=kind,\
               estimator = np.median, col_wrap=4,size = 5,aspect=0.6)
        plt.show()
        
    def pairplot(self,columns=[],kind='reg',diag_kind='kde',size=2):
        '''
        for numerical features
        kind : {scatter, reg} optional
        diag_kind : {hist, kde} optional
        '''
        sns.set()
        sns.pairplot(self.dataset[columns],size = size,kind =kind,diag_kind=diag_kind)
        plt.show()
        
    def correlation(self,show_heatmap=False,zoom=1,show_cm=True):
        correlation=self.dataset.corr()
        cols = correlation.nlargest(int(zoom*len(correlation)),self.ylabel)[self.ylabel].index
        print(cols)
        if show_heatmap:
            if show_cm:
                #cols = correlation.nlargest(int(zoom*len(correlation)),self.ylabel)[self.ylabel].index
                cm = np.corrcoef(self.dataset[cols].values.T)
                f , ax = plt.subplots(figsize = (14,12))
                sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
                    linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
                plt.show()
                
            else:
                f , ax = plt.subplots(figsize = (14,12))
                plt.title('Correlation of Numeric Features with %s'%(self.ylabel),y=1,size=16)
                sns.heatmap(correlation,square = True,  vmax=0.8)
                plt.show()
            
        print 'The correlation between features and %s'%(self.ylabel)
        return correlation[self.ylabel].sort_values(ascending = False)
           