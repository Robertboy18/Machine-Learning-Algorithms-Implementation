
"""
Linear Regression
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


# read data and summarize 
df = pd.read_csv('/input',index = 'id')
df['x1'] = df['column']
df['y'] = df['target']
df.head()
df.describe()
sns.heatmap(df.isnull(), cbar=False)

# split/Train/validate
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.3,random_state = 0)
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
sub['target'] = ypred
sub.to_csv('submission.csv')

#plot
p = figure()
p.circle(df.x1, df.y, legend='observation')
p.line(df.x1, y_pred, legend='model', color='red')
p.show(p)

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
