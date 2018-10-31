import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from scipy.stats import norm, skew
import graphviz
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBRegressor

df_train=pd.read_csv('../data/train.csv')
df_test=pd.read_csv('../data/test.csv')
df_all=pd.concat([df_train,df_test],ignore_index=True)
print 'The shape of df_train is:',df_train.shape
print 'The shape of df_test is:',df_test.shape
print 'The shape of df_all is:',df_all.shape

predict_df = df_test[['Id']]
df_all.drop('Id',axis=1,inplace=True)
df_all.drop('SalePrice',axis=1,inplace=True)

missing = (df_all.isnull().sum() / len(df_all)) * 100
missing = missing.drop(missing[missing == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :missing})
#missing_data.head(10)

for feature in df_all.columns:
    if df_all[feature].dtype=='object':
        df_all[feature]=df_all[feature].fillna('Missing')
    elif df_all[feature].dtype!='object':
        df_all[feature]=df_all[feature].fillna(np.mean(df_all[feature].values))
		
def categorial_feature(rate,dataset):
    global qualitative, quantitative
    likely_cat = {}
    for var in dataset.columns:
        likely_cat[var] = 1.*dataset[var].nunique()/dataset[var].count() < float(rate)
    qualitative = [f for f in dataset.columns if likely_cat[f]==True]
    quantitative = [f for f in dataset.columns if likely_cat[f]==False]
    print('There are %d quantitative features and %d qualitative features'%(len(quantitative),len(qualitative)))
    return 0
	
categorial_feature(0.05,df_all)

def feature_encoding(dataset):
    for f in qualitative:
        lbl = preprocessing.LabelEncoder()
        dataset[f]=lbl.fit_transform(list(dataset[f].values)) 
        #lbl.transform(list(dataset[f].values))
    print('encoding finished')
    return 0
	
feature_encoding(df_all)

def displot_feature(feature,dataset):
    sns.distplot(dataset[feature] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(dataset[feature])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
    plt.ylabel('Frequency')
    plt.title('%s distribution'%(feature))

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(dataset[feature], plot=plt)
    plt.show()
    return 0
def show_skewness(dataset):
    global skewness
    numeric_feats = dataset.columns

    # Check the skew of all numerical features
    skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    return 0
	
show_skewness(df_all)

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
#displot_feature('SalePrice',df_train)

def log_transformation(skew_value,dataset):
    global skewness
    skewness=skewness[abs(skewness)>0.75].dropna()
    print("There are {} skewed numerical features to log transform".format(skewness.shape[0]))
    skewed_features = skewness.index
    for feat in skewed_features:
        dataset[skewed_features] = np.log1p(dataset[skewed_features])
    print('......')
    print('transform finished')
    return 0
def boxcox_transformation(skew_value,lam,dataset):
    skewness=skewness[abs(skewness)>0.75].dropna()
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    from scipy.special import boxcox1p
    skewed_features = skewnessvalue.index
    for feat in skewed_features:
        dataset[feat] = boxcox1p(dataset[feat], lam)
    print('......')
    print('transform finished')
    return 0

log_transformation(0.75,df_all)

df_all=pd.get_dummies(df_all)

df_all.head(5)

ntrain=df_train.shape[0]
ntest=df_test.shape[0]

df_all.drop(['PoolQC','MiscFeature'],axis=1,inplace=True)

################################################################
#add new features
################################################################

df_all['TotalSF']=df_all['1stFlrSF']+df_all['2ndFlrSF']+df_all['TotalBsmtSF']
df_all['TotalBathrooms'] = df_all['BsmtFullBath'] + (df_all['BsmtHalfBath'] * 0.5) + df_all['FullBath'] + (df_all['HalfBath'] * 0.5)
df_all['TotalOveral']=df_all['OverallQual']+df_all['OverallCond']
df_all['TotalBsmt']=df_all['BsmtQual']+df_all['BsmtCond']
df_all['TotalExter']=df_all['ExterQual']+df_all['ExterCond']
df_all['TotalGarage']=df_all['GarageCond']+df_all['GarageQual']+df_all['GarageFinish']

added_feature=pd.concat([df_all['TotalSF'],df_all['TotalBathrooms'],df_all['TotalOveral'],df_all['TotalBsmt'],df_all['TotalExter'],df_all['TotalGarage']])

print df_all.shape

train_data = df_all[:ntrain]
test_data = df_all[ntrain:]

################################################################
#method 1
################################################################

train_x1=train_data
train_y1=df_train['SalePrice']
test_x1=test_data
print 'The shape of train_x is',train_x1.shape
print 'The shape of train_y is',train_y1.shape
print 'The shape of test_x is',test_x1.shape

##################################################################
#method 2 use selected feature
##################################################################
useful_feature=df_train.columns[[x-1 for x in [2,3,4,5,13,18,20,21,22,24,25,26,28,30,31,34,35,39,41,42,43,44,45,47,50,53,54,55,57,59,60,61,62,63,80]]]
useful_feature=list(useful_feature)
useful_feature.extend(['TotalSF','TotalBathrooms','TotalOveral','TotalBsmt','TotalExter','TotalGarage'])
print useful_feature
train_x2=train_data[useful_feature]
train_y2=train_y1
test_x2=test_data[useful_feature]
print 'The shape of train_x2 is',train_x2.shape
print 'The shape of train_y2 is',train_y2.shape
print 'The shape of test_x2 is',test_x2.shape


##################################################################
#method 3 by the order of importance
##################################################################
feature_importance=['LotArea','LotFrontage','GrLivArea','YearBuilt','1stFlrSF','BsmtFinSF1','YearRemodAdd','MSSubClass','TotalBsmtSF','GarageArea','OverallCond','BsmtUnfSF','Neighborhood','MoSold','OverallQual','MasVnrArea','WoodDeckSF','GarageYrBlt','OpenPorchSF','2ndFlrSF','MSZoning','Exterior1st','Condition1','BsmtFinType1','SaleCondition','BsmtExposure','LandContour','Exterior2nd','TotRmsAbvGrd','Functional','BedroomAbvGr','Fireplaces','YrSold','BsmtQual','MasVnrType','LotConfig','KitchenQual','LotShape','HouseStyle','ExterQual']
feature_importance.extend(['TotalSF','TotalBathrooms','TotalOveral','TotalBsmt','TotalExter','TotalGarage'])
train_x3=train_data[feature_importance]
train_y3=train_y1
test_x3=test_data[feature_importance]
print 'The shape of train_x3 is',train_x3.shape
print 'The shape of train_y3 is',train_y3.shape
print 'The shape of test_x3 is',test_x3.shape

