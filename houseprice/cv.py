from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from feature1 import train_x1, train_y1
import numpy as np
import pandas as pd

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x1.values)
    rmse= np.sqrt(-cross_val_score(model, train_x1.values, train_y1, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

for ap in [0.0015]:
# 0,0015 is the best alpha
    lasso = make_pipeline(RobustScaler(), Lasso(alpha =ap,max_iter=50000))
   # score = rmsle_cv(lasso)
   # print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#for ap in [0.9,0.95,0.96,0.97,0.98,1]:
 # 0,0007 is the best alpha
#    ENet =  ElasticNet(alpha =0.0006,l1_ratio=ap)
#    score = rmsle_cv(ENet)
#    print("\nENet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0007, l1_ratio=0.9))
KRR = KernelRidge(alpha=0.1, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01,
                                   max_depth=4, 
                                   min_samples_leaf=12, min_samples_split=10, 
                                  )
model_xgb = xgb.XGBRegressor(colsample_bytree=0.5, 
                             learning_rate=0.01, max_depth=4, 
                             min_child_weight=1.3, n_estimators=1900,
                             objective= 'reg:linear',
                             subsample=0.8, silent=1,
                             )

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=1900,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.5,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf =1.1)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
                        # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
     #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#score = rmsle_cv(KRR)
#print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#score = rmsle_cv(GBoost)
#print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#score = rmsle_cv(model_xgb)
#print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#score = rmsle_cv(model_lgb)
#print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

averaged_models = AveragingModels(models = (model_xgb,ENet))
score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
