# random_forest.py

import math as mth
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import metrics

def best_tree_number_depth(model, x_train, y_train, test_split: float, cross_val: int, verbose = False):
    depths = [5, 10, 15]
    num_trees = [50, 75, 100]
    score_list = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    score_names = ['MSE', 'MAE']
    min_mse = float('inf')

    for num_tree in num_trees:
        for depth in depths:
            print('Number of trees:', num_tree)
            print('Max depth:', depth)

            rfr = RandomForestRegressor(n_estimators=num_tree, criterion='squared_error', max_depth=depth, max_features='auto', 
                min_samples_split=2, n_jobs=1, random_state=12345)
            scores = cross_validate(rfr, x_train, y_train, scoring=score_list, return_train_score=False, cv=cross_val)

            print('{:.<25s}{:>20s}{:>20s}'.format('Metric', 'Mean', 'Std. Dev.'))
            i = 0
            for s in score_list:
                var = 'test_' + s
                mean = mth.fabs(scores[var].mean())
                std = scores[var].std()
                label = score_names[i]
                i += 1
                print('{:.<25s}{:>20.4f}{:>20.4f}'.format(label, mean, std))
                if label == 'MSE' and mean < min_mse:
                    min_mse = mean
                    best_depth = depth
                    best_num_trees = num_tree

    print('Best number of trees:', best_num_trees)
    print('Best depth:', best_depth)

def model_random_forest(model, x_train, y_train, test_split: float, cross_val: int, verbose = False):
    best_tree_number_depth(model, x_train, y_train, test_split, cross_val, verbose)

def random_forest(model, test_split: float, cross_val: int, verbose = False):
    # Set up training and validation sets as numpy arrays
    x = np.asarray(model.drop(['ConvertedCompYearly'], axis=1)) # All data except salaries
    y = np.asarray(model['ConvertedCompYearly']) # Salaries only
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=test_split, random_state=12345)

    # Generate random forest model
    rf = model_random_forest(model, x_train, y_train, test_split, cross_val, verbose)