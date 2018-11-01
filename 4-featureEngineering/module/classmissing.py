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

class Missingvalue(object):
    
    def __init__(self,dataset):
        self.dataset=dataset
        self.total = dataset.isnull().sum().sort_values(ascending=False)
        self.percent=(dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
        self.missing_data = pd.concat([self.total, self.percent], axis=1, keys=['Total', 'Percent'])
        self.missing_data=self.missing_data[self.missing_data['Percent']>0.00]
        self.drop_list=[]
        self.after_drop=list(self.missing_data.index)
        #self.filllist=list(dataset.columns)
        
    def plot_missing(self,show_plot=False,show_table=True):

        if show_table:
            print self.missing_data
        if len(self.missing_data)==0:
            show_plot==False
            print "there is no missing data now"
        elif len(self.missing_data)!=0:
            if show_plot:
                f, ax = plt.subplots(figsize=(12, 9))
                plt.xticks(rotation='90')
                sns.barplot(x=self.missing_data.index, y=self.missing_data['Percent'])
                plt.xlabel('Features', fontsize=15)
                plt.ylabel('Percent of missing values', fontsize=15)
                plt.title('Percent missing data by feature', fontsize=15)
                plt.show()
            else:
                pass
            
    def missing_process(self,filllist=[],fillnum='mean',fillcat='Missing',drop_threshold=0.90):
        
        if locals()['filllist']==[]:
            self.filllist=list(self.dataset.columns)
        elif locals()['filllist']!=[]:
            self.filllist=filllist
            
        def mode(array):
            Mode=[]
            SetList=set(array)
            count=1
            if len(array)==len(SetList):
                return []
            for i in SetList:
                if list(array).count(i)>count:
                    Mode=[]
                    Mode.append(i)
                    count=list(array).count(i)
                elif list(array).count(i)==count:
                    Mode.append(i)
            return Mode
        
        for i in self.missing_data.index:
            if self.missing_data.ix[i]['Percent'] >= drop_threshold:
                self.drop_list.append(i)
                self.drop_list=list(set(self.drop_list))
        for i in self.drop_list:
            if i in self.dataset.columns:
                self.dataset.drop(self.drop_list,axis=1,inplace=True)
            else:
                print 'missing data has been processed'
        for i in self.drop_list:
            if i in self.after_drop:
                self.after_drop.remove(i)
            else:
                print "remove finished"
            
        for i in [i for i in self.after_drop if i in self.filllist]:
            if self.dataset.dtypes[i]!='object':
                if locals()['fillnum']=='median':
                    self.dataset[i]=self.dataset[i].fillna(self.dataset[i].median())
                    print "fill none value of numerical data with median"
                elif locals()['fillnum']=='mean':
                    self.dataset[i]=self.dataset[i].fillna(self.dataset[i].mean())
                    print "fill none value of numerical data with mean"
                elif locals()['fillnum']=='mode':
                    self.dataset[i]=self.dataset[i].fillna(self.dataset[i].mode())
                    print "fill none value of numerical data with mode"
                elif locals()['fillnum']==0:
                    self.dataset[i]=self.dataset[i].fillna(0)
                    print "fill none value of numerical data with 0"
                elif locals()['fillnum']==-999:
                    self.dataset[i]=self.dataset[i].fillna(-999)
                    print "fill none value of numerical data with -999"
                else:
                    if type(locals()['fillnum'])=='int' or 'float':
                        self.dataset[i]=self.dataset[i].fillna(locals()['fillnum'])
                        print "fill none value of numerical data with user-defined value"
                    else:
                        raise TypeError('input must be a int/float value, or you can choose value from mean/median/mode/0/-999')
                    
            if self.dataset.dtypes[i]=='object':
                if locals()['fillcat']=='Missing':
                    self.dataset[i]=self.dataset[i].fillna('Missing')
                    print "fill none value of categorial data with Missing"
                elif locals()['fillcat']=='None':
                    self.dataset[i]=self.dataset[i].fillna('None')
                    print "fill none value of categorial data with None"
                else:
                    self.dataset[i]=self.dataset[i].fillna(locals()['fillcat'])
                    print "fill none value of categorial data with user-defined category"   
