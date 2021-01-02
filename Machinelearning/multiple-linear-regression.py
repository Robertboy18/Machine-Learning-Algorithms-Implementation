
"""
Multiple Linear Regression
Template - Kaggle Competitions/General Testdata/Open Datasets/Machine Learning Competitions
Author - Robert Joseph

Reference : Kaggle
"""


# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# read data and summarize 
df = pd.read_csv('/input',index = 'id')
df.rename('columns={"target": "y"}')
y = df['y'].values
X = table.drop(columns=['y'])
df.head()
df.describe()
sns.heatmap(df.isnull(), cbar=False)

# split/Train/validate
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.3,random_state = 0)

#imputation
train_X_plus = train_X.copy()
val_X_plus = val_X.copy()
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
for col in cols_with_missing:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    val_X_plus[col + '_was_missing'] = val_X_plus[col].isnull()
my_imputer = SimpleImputer()
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_val_X_plus = pd.DataFrame(my_imputer.transform(val_X_plus))
imputed_train_X_plus.columns = train_X_plus.columns
imputed_val_X_plus.columns = val_X_plus.columns
print("MAE - An Extension to Imputation:")
print(score_dataset(imputed_train_X_plus, imputed_val_X_plus, train_y, val_y))

# One hot encoding
s = (train_x.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))
OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index
num_train_x = train_X.drop(object_cols, axis=1)
num_X_valid = val_X.drop(object_cols, axis=1)
OH_train_x = pd.concat([num_train_x, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE- One-Hot Encoding:") 
print(score_dataset(OH_train_x, OH_X_valid, train_y, val_y))

model = LinearRegression().fit(train_X, train_y)
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
sq = model.score(val_X,val_y)
print('coefficient of determination:', sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

#Test/submit
sub = pd.read_csv('/input',index = 'id')
y_pred = model.predict(x)
sub.drop(df.ix[:, 'column1':'columnN'].columns, axis = 1,inplace = True)
sub['target'] = ypred
sub.to_csv('submission.csv')

#decision trees/ extra boosting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

