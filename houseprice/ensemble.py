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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from feature1 import train_x1,train_y1,test_x1,predict_df


num_rounds = 3000
params = {
    'eta': 0.01,
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'min_child_weight': 1.1,
    'max_depth': 4,
    'silent': 1,
}

dt = xgb.DMatrix(train_x1, label=train_y1)
cv = xgb.cv(params, dt, num_boost_round=num_rounds, nfold=5, early_stopping_rounds=30, metrics='rmse', callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(30)
        ])
		
num_rounds = cv.shape[0] - 1
print('Best rounds: ', num_rounds)

params = {
    'n_estimators': num_rounds,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'min_child_weight': 1.1,
    'max_depth': 4,
    'silent': 1,
}

model = XGBRegressor(**params)

print('Starting Cross Validation...')
score = cross_val_score(model, train_x1, train_y1, cv=5)
print('Score: ', score)
print('Mean CV scores: ', np.mean(score))

print('Training...')
model.fit(train_x1,train_y1)
print('Predicting...')

########################################################################
#lasso
#######################################################################
from sklearn.linear_model import Lasso,ElasticNet
best_alpha = 0.0015


#model_lasso = make_pipeline(RobustScaler(),Lasso(alpha=best_alpha, max_iter=50000))
model_lasso = Lasso(alpha=best_alpha, max_iter=50000)
model_lasso.fit(train_x1, train_y1)

print('Predicting...')

########################################################################
# enet
########################################################################

#model_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0007, l1_ratio=0.9))
model_ENet = ElasticNet(alpha=0.0007, l1_ratio=0.9)
model_ENet.fit(train_x1,train_y1)


########################################################################
#lgbm
########################################################################

import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=1900,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.5,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf =1.1)
print("\nFitting LightGBM model ...")
model_lgb.fit(train_x1,train_y1)
print('Predicting...')

##################################################################################
n_fold=5
pred_y_xgb = model.predict(test_x1)
pred_y_xgb=np.expm1(pred_y_xgb)
pred_y_lgb = model_lgb.predict(test_x1)
pred_y_lgb=np.expm1(pred_y_lgb)
pred_y_lasso = model_lasso.predict(test_x1)
pred_y_lasso=np.expm1(pred_y_lasso)
pred_y_ENet = model_ENet.predict(test_x1)
pred_y_ENet =  np.expm1(pred_y_ENet)
pred_y_1=(pred_y_xgb*0.25+pred_y_lasso*0.25+pred_y_lgb*0.25+pred_y_ENet*0.25)
pred_y_2=(pred_y_xgb*0.2+pred_y_lasso*0.3+pred_y_lgb*0.2+pred_y_ENet*0.3) 
pred_y_3=(pred_y_xgb*0.5+pred_y_lasso*0.3+pred_y_lgb*0.2)
pred_y=(pred_y_1+pred_y_2+pred_y_3)/3.0
###################################################################################

#pred_y=pred_y_xgb
#pred_y=(pred_y_xgb+pred_y_lgb)/2.0
predict_df['SalePrice'] = pred_y
print predict_df.shape
predict_df.to_csv('sub/subv27.csv', index=None)

