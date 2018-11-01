from DCW.module.singleEDA import SingleEDA
from DCW.module.classloaddata import *
import DCW.core.load as load
df_train=load.df_train


def Distplot(feature,fitmethod='norm',show_QQplot=True,show_skew=False,show_kurt=False,color='m',kde=True,rug=False):
    s=SingleEDA(df_train)
    return s.distplot(feature=feature,fitmethod=fitmethod,show_QQplot=show_QQplot,show_skew=show_skew,show_kurt=show_kurt,color=color,kde=kde,rug=rug) 
    
def Describe(feature,show_plot=True):
    s=SingleEDA(df_train)
    return s.show_describe(feature=feature,show_plot=show_plot)
    
def Kurtosis(show_plot=True):
    s=SingleEDA(df_train)
    return s.show_kurtosis(show_plot=show_plot)
    
def Skewness(show_plot=True):
    s=SingleEDA(df_train)
    return s.show_skewness(show_plot=show_plot)    
    
def Countplot(feature,show_plot=True):
    s=SingleEDA(df_train)
    return s.countplot(feature=feature,show_plot=show_plot)