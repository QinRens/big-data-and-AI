from DCW.module.classoutlier import *
from DCW.module.classloaddata import *
import DCW.core.load as load
df_train=load.df_train
data=load.data
targetdf_list=load.targetdf_list

def Outlier_Drop(feature,label,k=4,method='',droplist=[]):
    if locals()['feature']=='all':
	    for i in range(len(quantitative)):
			out=OutlierNum(totdata=df_train,target=targetdf_list[i],targetdf=deepcopy(targetdf_list[i]))
			out.outlier_collector(k)
			out.outlier_process(method=method,droplist=droplist)
			print "drop outlier of %s finished"%(targetdf_list[i].columns[0])
    else:
		out=OutlierNum(totdata=df_train,target=targetdf_list[quantitative.index(feature)],targetdf=deepcopy(targetdf_list[quantitative.index(feature)]))
		out.outlier_collector(k)
		out.outlier_process(method=method,droplist=droplist)
		print "drop outlier of %s finished"%(feature)
		
		
def Outlier_Plot(feature,label,k=4,show_grid=True,plot_outlier=True):
    if locals()['feature']=='all':
        for i in range(len(quantitative)):
            out=OutlierNum(totdata=df_train,target=targetdf_list[i],\
                   targetdf=deepcopy(targetdf_list[i]))
        
            out.outlier_collector(k)
            out.plot_data(show_grid,plot_outlier)
    else:
        out=OutlierNum(totdata=df_train,target=targetdf_list[quantitative.index(feature)],\
                   targetdf=deepcopy(targetdf_list[quantitative.index(feature)]))
        out.outlier_collector(k)
        out.plot_data(show_grid,plot_outlier)

def Outlier_collect(feature,label,k=4):
    
    if locals()['feature']=='all':
        all_collector={}
        for i in range(len(quantitative)):
            out=OutlierNum(totdata=df_train,target=targetdf_list[i],\
                   targetdf=deepcopy(targetdf_list[i]))
        
            all_collector[quantitative[i]]=out.outlier_collector(k)
        return all_collector
    else:
        out=OutlierNum(totdata=df_train,target=targetdf_list[quantitative.index(feature)],\
                   targetdf=deepcopy(targetdf_list[quantitative.index(feature)])) 
        out.outlier_collector(k)
        return out.outlier_tot		