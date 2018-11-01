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
class Single(object):
    
    def __init__(self,dataset):
        self.dataset=dataset
        
    def duplicate_process(self):
        if self.dataset.duplicated().any():
            self.dataset=self.dataset.drop_duplicates()
            print "drop duplicated finished"
        else:
            print "no duplicated value exist"
            
    def single_collector(self,single_ratio=0.95):
        
        collector={}
        for i in self.dataset.columns:
            max_value_count=self.dataset[i].value_counts().values[0]
            total_count=self.dataset[i].count()
            ratio=float(max_value_count)/float(total_count)
            if ratio >= single_ratio:
                collector[i]=ratio
                #print "feature %s has too many single values"%(i,)
        print "single value ratio count"
        return collector
                    
    def single_plot(self,show_plot=True,show_table=True,single_ratio=0.8):
        if show_plot:
            f, ax = plt.subplots(figsize=(12, 9))
            plt.xticks(rotation='90')
            sns.barplot(x=self.single_collector(single_ratio).keys(), y=self.single_collector(single_ratio).values())
            plt.xlabel('Features', fontsize=15) 
            plt.ylabel('Percent of single values', fontsize=15)
            plt.title('Percent single data by feature', fontsize=15)
            plt.show()
            
        else:
            pass
            
        if show_table:
            single_df=pd.Series(self.single_collector(single_ratio))
            print single_df
            
        else:
            pass
            
            
    def single_process(self,drop=True,single_ratio=0.95):
        
        if drop:
            for i in self.single_collector(single_ratio).keys():
                self.dataset.drop(i,axis=1,inplace=True)
            print "drop over"
        else:
            pass
           