# random_forest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

from debug import get_model_stats

def best_tree_number_depth(model, x_train, y_train, test_split: float, cross_val: int, verbose = False):
    # Adjust these to check different depths and tree numbers
    depths =  [8, 10, 12]
    num_trees = [125, 150]

    best_depth = 0
    best_num_trees = 0

    score_list = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    max_err = float('-inf')

    # Use MSE or MAE to select the lowest error
    selected_test = score_list[1]

    # Identify the tree depth and tree number combination with the lowest error
    for num_tree in num_trees:
        for depth in depths:
            rfr = RandomForestRegressor(n_estimators=num_tree, criterion='squared_error', max_depth=depth, 
                max_features='auto', min_samples_split=2, n_jobs=1, random_state=12345)
            if verbose:
                print('num_trees =', num_tree)
                print('max_depth =', depth)
                print('{:<25s}{:>20s}{:>20s}'.format('Metric', 'Mean', 'Std Dev'))
            for s in score_list:
                rfr_cvs = cross_val_score(rfr, x_train, y_train, scoring=s, cv=cross_val, error_score='raise')
                mean = rfr_cvs.mean()
                std = rfr_cvs.std()
                if verbose:
                    print('{:<25s}{:>20.4f}{:>20.4f}'.format(s, mean, std))
                if s == selected_test and mean > max_err:
                    max_err = mean
                    best_depth = depth
                    best_num_trees = num_tree
            if verbose:
                print()

    if verbose:
        print('best_num_trees =', best_num_trees)
        print('best_depth =', best_depth)
        print()

    return best_num_trees, best_depth

def model_random_forest(model, x_train, y_train, test_split: float, cross_val: int, verbose = False):
    num_trees, depth = best_tree_number_depth(model, x_train, y_train, test_split, cross_val, verbose)
    rf = RandomForestRegressor(n_estimators=num_trees, criterion='squared_error', max_depth=depth, min_samples_split=2, 
        min_samples_leaf=1, max_features='auto', n_jobs=1, bootstrap=True, random_state=12345)
    return rf.fit(x_train, y_train)

def random_forest(model, country: str, test_split: float, cross_val: int, verbose = False):
    # Set up training and validation sets as numpy arrays
    x = np.asarray(model.drop(['ConvertedCompYearly'], axis=1)) # All data except salaries
    y = np.asarray(model['ConvertedCompYearly']) # Salaries only
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=test_split, random_state=12345)

    # Generate random forest model
    rf = model_random_forest(model, x_train, y_train, test_split, cross_val, verbose)


    # Print accuracy statistics of training and validation sets
    tr_pred = rf.predict(x_train)
    va_pred = rf.predict(x_validate)
    get_model_stats(y_train, tr_pred, verbose) # Training results
    get_model_stats(y_validate, va_pred, verbose) # Validation results

    # Print most important attributes to the random forest
    ind_model = model.drop(['ConvertedCompYearly'], axis=1)
    lst = list(ind_model.columns)
    col_impt = pd.Series(rf.feature_importances_, index=lst)
    print(col_impt.nlargest(10).sort_values(ascending=False))

    # Scatterplot of actual vs predicted values of testing set
    fig, ax = plt.subplots()
    ax.scatter(y_validate, va_pred)
    ax.plot([y_validate.min(), y_validate.max()], [y_validate.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.suptitle(country + ' RF: Predicted vs Actual ConvertedCompYearly')
    plt.title('test_split=' + str(test_split) + ' cross_val=' + str(cross_val))
    plt.show()