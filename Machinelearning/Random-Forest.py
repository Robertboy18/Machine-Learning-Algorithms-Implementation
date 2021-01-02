"""
Random Forest
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

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

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Bundle preprocessing and modeling code in a pipeline
my_model = RandomForestRegressor(random_state=1)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', my_model)
                             ])
my_pipeline.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], 
             verbose=False)

preds = my_pipeline.predict(val_X)
score = mean_absolute_error(val_y, preds)
print('MAE:', score)

preds = my_pipeline.predict(val_X)
score = mean_absolute_error(val_y, preds)
print('MAE:', score)

#Test/submit
sub = pd.read_csv('/input',index = 'id')
y_pred = my_pipeline.predict(x)
sub.drop(df.ix[:, 'column1':'columnN'].columns, axis = 1,inplace = True)
sub['target'] = ypred
sub.to_csv('submission.csv')

#cross validation
scores = -1 * cross_val_score(my_pipeline, train_X, train_y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

