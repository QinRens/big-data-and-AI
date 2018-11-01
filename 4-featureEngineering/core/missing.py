from DCW.module.classmissing import *
from DCW.module.classloaddata import *
import DCW.core.load as load
df_train=load.df_train

def Missing_Table():
    mis=Missingvalue(dataset=df_train)
    mis.plot_missing(show_plot=False,show_table=True)

def Missing_Plot():
    mis=Missingvalue(dataset=df_train)
    mis.plot_missing(show_plot=True,show_table=False) 
    
def Missing_Process(drop_threshold=0.90,filllist=[],fillnum='mean',fillcat='Missing'):
    mis=Missingvalue(dataset=df_train)
    mis.missing_process(drop_threshold=drop_threshold,filllist=filllist,fillnum=fillnum,fillcat=fillcat)
    return "the features you dropped are ",mis.drop_list