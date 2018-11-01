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

class OutlierNum(object):
    
    def __init__(self,totdata,target,targetdf):
        self.totdata=totdata
        self.target=target
        self.targetdf=targetdf
        self.max_value_count=self.targetdf[self.targetdf.columns[0]].value_counts().values[0]
        self.max_value_index=self.targetdf[self.targetdf.columns[0]].value_counts().index[0]
        self.total_count=self.targetdf[self.targetdf.columns[0]].count()
        if float(self.max_value_count)/float(self.total_count) >= 0.4:
            self.targetdf.drop(self.targetdf[targetdf[self.targetdf.columns[0]]==self.max_value_index].index,inplace=True)
        else:
            pass
        self.outlier_tot=[]
        self.grid_df={}
        self.value_count={}
        self.X_axis = target[target.columns[0]]
        self.Y_axis = target[target.columns[1]]
        self.X_1=float(self.X_axis.min())
        self.X_5=float(self.X_axis.max())
        self.X_3=(self.X_1+self.X_5)/2.0
        self.X_2=(self.X_1+self.X_3)/2.0
        self.X_4=(self.X_3+self.X_5)/2.0
        self.Y_1=float(self.Y_axis.min())
        self.Y_5=float(self.Y_axis.max())
        self.Y_3=(self.Y_1+self.Y_5)/2.0
        self.Y_4=(self.Y_3+self.Y_5)/2.0
        self.Y_2=(self.Y_1+self.Y_3)/2.0
        
        self.value_count['s11']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].count().values[0]
        self.value_count['s12']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].count().values[0]
        self.value_count['s13']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].count().values[0]
        self.value_count['s14']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].count().values[0]
        self.value_count['s21']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].count().values[0]
        self.value_count['s22']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].count().values[0]
        self.value_count['s23']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].count().values[0]
        self.value_count['s24']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].count().values[0]
        self.value_count['s31']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].count().values[0]
        self.value_count['s32']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].count().values[0]
        self.value_count['s33']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].count().values[0]
        self.value_count['s34']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].count().values[0]
        self.value_count['s41']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].count().values[0]
        self.value_count['s42']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].count().values[0]
        self.value_count['s43']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].count().values[0]
        self.value_count['s44']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].count().values[0]
        self.value_count['s1j']=self.value_count['s11']+self.value_count['s12']+self.value_count['s13']+self.value_count['s14']
        self.value_count['s2j']=self.value_count['s21']+self.value_count['s22']+self.value_count['s23']+self.value_count['s24']
        self.value_count['s3j']=self.value_count['s31']+self.value_count['s32']+self.value_count['s33']+self.value_count['s34']
        self.value_count['s4j']=self.value_count['s41']+self.value_count['s42']+self.value_count['s43']+self.value_count['s44']
        self.value_count['si1']=self.value_count['s11']+self.value_count['s21']+self.value_count['s31']+self.value_count['s41']
        self.value_count['si2']=self.value_count['s12']+self.value_count['s22']+self.value_count['s32']+self.value_count['s42']
        self.value_count['si3']=self.value_count['s13']+self.value_count['s23']+self.value_count['s33']+self.value_count['s43']
        self.value_count['si4']=self.value_count['s14']+self.value_count['s24']+self.value_count['s34']+self.value_count['s44']

        self.grid_df['s11']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].values
        self.grid_df['s12']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].values
        self.grid_df['s13']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].values
        self.grid_df['s14']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_4)&(targetdf[targetdf.columns[1]]<=self.Y_5)].values
        self.grid_df['s21']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].values
        self.grid_df['s22']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].values
        self.grid_df['s23']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].values
        self.grid_df['s24']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_3)&(targetdf[targetdf.columns[1]]<=self.Y_4)].values
        self.grid_df['s31']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].values
        self.grid_df['s32']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].values
        self.grid_df['s33']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].values
        self.grid_df['s34']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_2)&(targetdf[targetdf.columns[1]]<=self.Y_3)].values
        self.grid_df['s41']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_1)&(targetdf[targetdf.columns[0]]<=self.X_2)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].values
        self.grid_df['s42']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_2)&(targetdf[targetdf.columns[0]]<=self.X_3)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].values
        self.grid_df['s43']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_3)&(targetdf[targetdf.columns[0]]<=self.X_4)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].values
        self.grid_df['s44']=targetdf[(targetdf[targetdf.columns[0]]>=self.X_4)&(targetdf[targetdf.columns[0]]<=self.X_5)&\
                                         (targetdf[targetdf.columns[1]]>=self.Y_1)&(targetdf[targetdf.columns[1]]<=self.Y_2)].values

        self.grid_df=pd.DataFrame([self.grid_df])
        self.s4j=np.concatenate((self.grid_df['s41'].values[0],self.grid_df['s42'].values[0],self.grid_df['s43'].values[0],self.grid_df['s44'].values[0]), axis=0)
        self.s3j=np.concatenate((self.grid_df['s31'].values[0],self.grid_df['s32'].values[0],self.grid_df['s33'].values[0],self.grid_df['s34'].values[0]), axis=0)
        self.s2j=np.concatenate((self.grid_df['s21'].values[0],self.grid_df['s22'].values[0],self.grid_df['s23'].values[0],self.grid_df['s24'].values[0]), axis=0)
        self.s1j=np.concatenate((self.grid_df['s11'].values[0],self.grid_df['s12'].values[0],self.grid_df['s13'].values[0],self.grid_df['s14'].values[0]), axis=0)
        self.si4=np.concatenate((self.grid_df['s14'].values[0],self.grid_df['s24'].values[0],self.grid_df['s34'].values[0],self.grid_df['s44'].values[0]), axis=0)
        self.si3=np.concatenate((self.grid_df['s13'].values[0],self.grid_df['s23'].values[0],self.grid_df['s33'].values[0],self.grid_df['s43'].values[0]), axis=0)
        self.si2=np.concatenate((self.grid_df['s12'].values[0],self.grid_df['s22'].values[0],self.grid_df['s32'].values[0],self.grid_df['s42'].values[0]), axis=0)
        self.si1=np.concatenate((self.grid_df['s11'].values[0],self.grid_df['s21'].values[0],self.grid_df['s31'].values[0],self.grid_df['s41'].values[0]), axis=0)
        self.scan_row_list=[self.s1j,self.s2j,self.s3j,self.s4j]
        self.scan_columns_list=[self.si1,self.si2,self.si3,self.si4]
        
    def plot_data(self,plot_outlier=True,show_grid=True):
        figure, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(left=self.X_1, right=self.X_5)
        ax.set_ylim(bottom=self.Y_1, top=self.Y_5)
        line1 = [(self.X_1, self.Y_1), (self.X_1, self.Y_5)] 
        line2 = [(self.X_2, self.Y_1), (self.X_2, self.Y_5)]
        line3 = [(self.X_3, self.Y_1), (self.X_3, self.Y_5)]
        line4 = [(self.X_4, self.Y_1), (self.X_4, self.Y_5)]
        line5 = [(self.X_5, self.Y_1), (self.X_5, self.Y_5)]
        line6 = [(self.X_1, self.Y_1), (self.X_5, self.Y_1)]
        line7 = [(self.X_1, self.Y_2), (self.X_5, self.Y_2)]
        line8 = [(self.X_1, self.Y_3), (self.X_5, self.Y_3)]
        line9 = [(self.X_1, self.Y_4), (self.X_5, self.Y_4)]
        line10 = [(self.X_1, self.Y_5), (self.X_5, self.Y_5)]

        (line1_xs, line1_ys) = zip(*line1)
        (line2_xs, line2_ys) = zip(*line2)
        (line3_xs, line3_ys) = zip(*line3)
        (line4_xs, line4_ys) = zip(*line4)
        (line5_xs, line5_ys) = zip(*line5)
        (line6_xs, line6_ys) = zip(*line6)
        (line7_xs, line7_ys) = zip(*line7)
        (line8_xs, line8_ys) = zip(*line8)
        (line9_xs, line9_ys) = zip(*line9)
        (line10_xs, line10_ys) = zip(*line10)
        # scatter plot
        ax.scatter(self.target[[0]],self.target[[1]])
        if locals()['show_grid']==True:
            ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='black'))
            ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='black'))
            ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='black'))
            ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='black'))
            ax.add_line(Line2D(line5_xs, line5_ys, linewidth=2, color='black'))
            ax.add_line(Line2D(line6_xs, line6_ys, linewidth=2, color='black'))
            ax.add_line(Line2D(line7_xs, line7_ys, linewidth=1, color='black'))
            ax.add_line(Line2D(line8_xs, line8_ys, linewidth=1, color='black'))
            ax.add_line(Line2D(line9_xs, line9_ys, linewidth=1, color='black'))
            ax.add_line(Line2D(line10_xs, line10_ys, linewidth=1, color='black'))
        elif locals()['show_grid']==False:
            pass
    
        if locals()['plot_outlier']==False:
            pass
        elif locals()['plot_outlier']==True:
            for i in self.outlier_tot:
                ax.scatter(i[0],i[1],color='red')
        plt.plot()
        plt.grid(False)
        plt.xlabel(self.targetdf.columns[0])
        plt.ylabel(self.targetdf.columns[1])
        plt.show()
        
    def outlier_detector(self,array,k=4,scan_columns=False):
        Q1=float(np.percentile(array[:,scan_columns],[25])[0])
        Q2=float(np.percentile(array[:,scan_columns],[50])[0])
        Q3=float(np.percentile(array[:,scan_columns],[75])[0])
        up_limit=Q3+k*(Q3-Q1)
        low_limit=Q1-k*(Q3-Q1)
        outlier=[]
        for i,v in enumerate(array):
            if v[scan_columns] > up_limit or v[scan_columns] < low_limit:
                outlier.append(v)
        return outlier

    def outlier_collector(self,k=4):
        
        def removearray(L,arr):
            ind = 0
            size = len(L)
            while ind != size and not np.array_equal(L[ind],arr):
                ind += 1
            if ind != size:
                L.pop(ind)
            else:
                raise ValueError('array not found in list.')
                
        if (np.array(self.s1j==self.grid_df['s11'][0]).all() or np.array(self.s1j==self.grid_df['s12'][0]).all() or \
             np.array(self.s1j==self.grid_df['s13'][0]).all() or np.array(self.s1j==self.grid_df['s14'][0]).all()) and self.value_count['s1j']<=10:
            removearray(self.scan_row_list,self.s1j)

        if (np.array(self.s2j==self.grid_df['s21'][0]).all() or np.array(self.s2j==self.grid_df['s22'][0]).all() or \
             np.array(self.s2j==self.grid_df['s23'][0]).all() or np.array(self.s2j==self.grid_df['s24'][0]).all()) and self.value_count['s2j']<=10:
            removearray(self.scan_row_list,self.s2j)

        if (np.array(self.s3j==self.grid_df['s31'][0]).all() or np.array(self.s3j==self.grid_df['s32'][0]).all() or \
             np.array(self.s3j==self.grid_df['s33'][0]).all() or np.array(self.s3j==self.grid_df['s34'][0]).all()) and self.value_count['s3j']<=10:
            removearray(self.scan_row_list,self.s3j)

        if (np.array(self.s4j==self.grid_df['s41'][0]).all() or np.array(self.s4j==self.grid_df['s42'][0]).all() or \
             np.array(self.s4j==self.grid_df['s43'][0]).all() or np.array(self.s4j==self.grid_df['s44'][0]).all()) and self.value_count['s4j']<=10:
            removearray(self.scan_row_list,self.s4j)  

        if (np.array(self.si1==self.grid_df['s11'][0]).all() or np.array(self.si1==self.grid_df['s21'][0]).all() or \
             np.array(self.si1==self.grid_df['s31'][0]).all() or np.array(self.si1==self.grid_df['s41'][0]).all()) and self.value_count['si1']<=10:
            removearray(self.scan_columns_list,self.si1)

        if (np.array(self.si2==self.grid_df['s12'][0]).all() or np.array(self.si2==self.grid_df['s22'][0]).all() or \
             np.array(self.si2==self.grid_df['s32'][0]).all() or np.array(self.si2==self.grid_df['s42'][0]).all()) and self.value_count['si2']<=10:
            removearray(self.scan_columns_list,self.si2)

        if (np.array(self.si3==self.grid_df['s13'][0]).all() or np.array(self.si3==self.grid_df['s23'][0]).all() or \
             np.array(self.si3==self.grid_df['s33'][0]).all() or np.array(self.si3==self.grid_df['s43'][0]).all()) and self.value_count['si3']<=10:
            removearray(self.scan_columns_list,self.si3)

        if (np.array(self.si4==self.grid_df['s14'][0]).all() or np.array(self.si4==self.grid_df['s24'][0]).all() or \
             np.array(self.si4==self.grid_df['s34'][0]).all() or np.array(self.si4==self.grid_df['s44'][0]).all()) and self.value_count['si4']<=10:
            removearray(self.scan_columns_list,self.si4)
                   
        for i in range(len(self.scan_columns_list)):
            if len(self.scan_columns_list[i])==0:
                pass
            else:
                self.outlier_tot+=self.outlier_detector(self.scan_columns_list[i],k,scan_columns=True)
        for i in range(len(self.scan_row_list)):
            if len(self.scan_row_list[i])==0:
                pass
            else:
                self.outlier_tot+=self.outlier_detector(self.scan_row_list[i],k,scan_columns=False)            
            
        return self.outlier_tot
    
    def outlier_process(self, method='',droplist=[]):
        if len(self.outlier_tot)==0:
            pass
        else:
            if locals()['method']=='':
                pass
            elif locals()['method']=='dropall':
                droplist=self.outlier_tot
                for i in range(len(droplist)):
                    self.totdata.drop(self.totdata[self.totdata[self.targetdf.columns[0]].isin([self.outlier_tot[i][0]])\
                                &self.totdata[self.targetdf.columns[1]].isin([self.outlier_tot[i][1]])].index,inplace=True)
            elif locals()['method']=='dropsome':
                for i in range(len(droplist)):
                    self.totdata.drop(self.totdata[self.totdata[self.targetdf.columns[0]].isin([droplist[i][0]])\
                                &self.totdata[self.targetdf.columns[1]].isin([droplist[i][1]])].index,inplace=True)