# decision_tree.py 

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import *

def best_tree_depth(model, test_split: float, verbose = False):
    x = np.asarray(model.drop(['ConvertedCompYearly'], axis=1)) # All data except salaries
    y = np.asarray(model['ConvertedCompYearly']) # Salaries only

    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=test_split, random_state=12345)

    score_list = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    search_depths = [4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25]
    max_MSE = float('-inf')
    best_depth = 0

    # Use mean absolute error to identify the best tree depth
    for d in search_depths:
        dtr = DecisionTreeRegressor(max_depth=d, min_samples_split=5, min_samples_leaf=5)
        if verbose:
            print(dtr)
            print('max_depth =', d)
            print('{:.<25s}{:>20s}{:>20s}'.format('Metric', 'Mean', 'Std. Dev.'))
        mean_score = []
        std_score = []
        for s in score_list:
            dtr_10 = cross_val_score(dtr, x_train, y_train, scoring=s, cv=10, error_score='raise')
            mean = dtr_10.mean()
            std = dtr_10.std()
            mean_score.append(mean)
            std_score.append(std)
            if verbose:
                print('{:.<25s}{:>20.4f}{:>20.4f}'.format(s, mean, std))
            if s == 'neg_mean_absolute_error' and mean > max_MSE:
                max_MSE = mean
                best_depth = d

    if verbose:
        print('Best depth =', best_depth)

    return best_depth