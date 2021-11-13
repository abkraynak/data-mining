# debug.py

import numpy as np
from sklearn import metrics

# Debug: check for missing values
def print_missing_value_counts(df):
    print('Number of missing values per column:')
    for i in df:
        print(i, ': ', end='')
        if df[i].count() < df.shape[0]:
            print(df.shape[0] - df[i].count(), end='')
        print()

# Debug: check number of records and attributes
def print_num_recs_attr(df):
    print('Number of records:', df.shape[0])
    print('Number of attributes:', df.shape[1])

# Debug: check missing values and record/attribute count
def debug_data_summary(df):
    print_num_recs_attr(df)
    print_missing_value_counts(df)

# Debug: print all the statistics for a model
def print_model_stats(r2: float, mae: float, mse: float, rmse: float) -> None:
    print('R-squared:', r2)
    print('Mean absolute error:', mae)
    print('Mean squared error', mse)
    print('Root mean squared error', rmse)
    print()

# Debug: calculate and print all the statistics for a model
def get_model_stats(data, pred, verbose = False) -> None:
    r2 = metrics.r2_score(data, pred)
    mae = metrics.mean_absolute_error(data, pred)
    mse = metrics.mean_squared_error(data, pred)
    rmse = np.sqrt(mse)
    if verbose:
        print_model_stats(r2, mae, mse, rmse)

