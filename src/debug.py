# debug.py

import numpy as np
from sklearn import metrics

def print_model_stats(r2: float, mae: float, mse: float, rmse: float) -> None:
    print('R-squared:', r2)
    print('Mean absolute error:', mae)
    print('Mean squared error', mse)
    print('Root mean squared error', rmse)
    print()

def get_model_stats(data, pred, verbose = False) -> None:
    r2 = metrics.r2_score(data, pred)
    mae = metrics.mean_absolute_error(data, pred)
    mse = metrics.mean_squared_error(data, pred)
    rmse = np.sqrt(mse)
    if verbose:
        print_model_stats(r2, mae, mse, rmse)

