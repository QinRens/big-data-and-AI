from DCW.module.multiEDA import *
from DCW.module.classloaddata import *
import DCW.core.load as load
df_train=load.df_train

def Scatterplot(xlabel,ylabel,reg=True,color=None,logx=False,robust=False):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.scatterplot(reg=reg,color=color,logx=logx,robust=robust)

def Jointplot(xlabel,ylabel,kind='reg',color='b',add_scatter=False):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.jointplot(kind=kind,color=color,add_scatter=add_scatter)

def Pairplot(xlabel,ylabel,columns,kind='scatter',diag_kind='kde',size=3):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.pairplot(columns=columns,kind=kind,diag_kind=diag_kind,size=size)

def Correlation(xlabel,ylabel,zoom=0.2,show_cm=True,show_heatmap=True):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.correlation(zoom=zoom,show_cm=show_cm,show_heatmap=show_heatmap)

def Barplot(xlabel,ylabel,hue):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.barplot(hue=hue)

def Boxplot(xlabel,ylabel,hue):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.boxplot(hue=hue)

def Violinplot(xlabel,ylabel,hue):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.violinplot(hue=hue)

def Pointplot(xlabel,ylabel,hue):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.pointplot(hue=hue)

def Stripplot(xlabel,ylabel,hue):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.stripplot(hue=hue)

def Swarmplot(xlabel,ylabel,hue):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.swarmplot(hue=hue)

def Factorplot(xlabel,ylabel,hue,col,color=None,kind='point'):
    m=multiEDA(df_train,xlabel=xlabel,ylabel=ylabel)
    return m.factorplot(color=color,hue=hue,col=col,kind=kind)