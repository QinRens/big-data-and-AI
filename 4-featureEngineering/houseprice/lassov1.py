from feature1 import train_x1, train_y1,test_x1,predict_df
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
best_alpha = 0.00099

model = Lasso(alpha=best_alpha, max_iter=50000)
model.fit(train_x1, train_y1)

print('Predicting...')
pred_y = model.predict(test_x1)
pred_y=np.expm1(pred_y)
predict_df['SalePrice'] = pred_y
print predict_df.shape
predict_df.to_csv('sub/subv13.csv', index=None)

