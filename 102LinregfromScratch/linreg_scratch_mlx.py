# %% load package
import numpy as np
import pandas as pd
import mlx.core as mx
import seaborn as sns

# %% data import 
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

# %% visualizse model
# wt => weight ; mpg => miles per gallon
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)
# %%  convert data to tensor

# %% training

# %% check results

# %% linear Regression

# %% create graph visualisation

