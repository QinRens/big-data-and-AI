# Parameters
FUDGE_FACTOR = 1.1200  # Multiply forecasts by this

XGB_WEIGHT = 0.6200
BASELINE_WEIGHT = 0.0100
OLS_WEIGHT = 0.0620
NN_WEIGHT = 0.0800

XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from feature1 import train_x1,train_y1,test_x1,predict_df


d_train = lgb.Dataset(train_x1, label=train_y1)
##### RUN LIGHTGBM

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                               learning_rate=0.01, n_estimators=1900,
                               max_bin = 55, bagging_fraction = 0.8,
                               bagging_freq = 5, feature_fraction = 0.5,
                               feature_fraction_seed=9, bagging_seed=9,
                               min_data_in_leaf =6, min_sum_hessian_in_leaf =1.1)


print("\nFitting LightGBM model ...")
model_lgb.fit(train_x1,train_y1)
print('Predicting...')
pred_y = model_lgb.predict(test_x1)
pred_y=np.expm1(pred_y)
predict_df['SalePrice'] = pred_y
print predict_df.shape
predict_df.to_csv('sub/subv15.csv', index=None)
