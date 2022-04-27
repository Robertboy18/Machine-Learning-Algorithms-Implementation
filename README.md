# Templates for Machine Learning Competitions/Machine Learning Algorithms

<hr>

## Competitive Programming templates.
> I mainly use C++ and Python usually for Competitive Programming. 

<img src = "https://miro.medium.com/max/1600/1*OhjNZ4SA_VwcLJ93vRm3BA.png">

>Feel free to modify it and use it.
<hr>

## Machine Learning Templates
> Websites such as Kaggle/DPhi/Codelabs and such have a lot of Machine Learning Competitions and I do know that each problem has its own modifications and such. The templates present in the folder are separated according to the various Machine Learning Algorithms. Each template for each Algorithm has a defined format in a way such that most Machine Learning Competitions on Kaggle/Dphi are organized.
Feel free to change it according to the problem and add more Data Analysis step or such to understand the data more. 

<img src = "https://miro.medium.com/max/2400/1*c_fiB-YgbnMl6nntYGBMHQ.jpeg">

<hr>

## Code Samples

```
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Bundle preprocessing and modeling code in a pipeline
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=0)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', my_model)
                             ])
my_pipeline.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], 
             verbose=False)
```
   
 
<hr>
Hope this helps everyone and feel free to open any issues and suggest more ideas on how to improve/new templates for various other algorithms.
<hr>

-Robert

