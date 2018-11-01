from DCW.module.classsingle import *
from DCW.module.classloaddata import *
import DCW.core.load as load
df_train=load.df_train

def single_process(drop_threshold=0.99,single_ratio=0.95,collect=True,show_plot=False,show_table=False,drop=False):
    s=Single(dataset=df_train)
    if collect:
        print s.single_collector(single_ratio)
    
    if show_plot:
        s.single_plot(single_ratio=single_ratio,show_plot=True,show_table=False)
    
    if show_table:
        s.single_plot(single_ratio=single_ratio,show_plot=False,show_table=True)
    
    if drop:
        s.single_process(drop=drop,single_ratio=drop_threshold)
    
def duplicate_process():
    s=Single(dataset=df_train)
    s.duplicate_process()