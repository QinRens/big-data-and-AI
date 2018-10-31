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
from feature1 import train_x1,train_y1,test_x1,predict_df


num_rounds = 3000
params = {
    'eta': 0.01,
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
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
score = cross_val_score(model, train_x1, train_y1, cv=10)
print('Score: ', score)
print('Mean CV scores: ', np.mean(score))

print('Training...')
model.fit(train_x1,train_y1)
print('Predicting...')
pred_y = model.predict(test_x1)
pred_y=np.expm1(pred_y)
predict_df['SalePrice'] = pred_y
print predict_df.shape
predict_df.to_csv('sub/subv10.csv', index=None)
