from DCW.module.classloaddata import LoadData
#import DCW.module.classloaddata as cld

def loaddata(dataset,labellist,IDlist): 
    global df_train,qualitative,quantitative,targetdf_list,data
    data=LoadData(dataset=dataset,labellist=labellist,IDlist=IDlist)
    data.init_dataset()
    from DCW.module.classloaddata import df_train,qualitative,quantitative,targetdf_list
	

#import DCW.core.missing as ms
#df_train=ms.df_train
