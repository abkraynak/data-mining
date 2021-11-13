# decision_tree.py 

import numpy as np
import pandas as pd
import graphviz as gv
import matplotlib.pyplot as plt
from pydotplus.graphviz import *
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import *

def best_tree_depth(model, x_train, y_train, test_split: float, cross_val: int, verbose = False):
    # Adjust this to check different depths
    depths = [4, 5, 6, 7] 

    best_depth = 0

    score_list = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    max_err = float('-inf')

    # Use MSE or MAE to select the lowest error
    selected_test = score_list[1]

    # Identify the tree depth with the lowest error
    for d in depths:
        dtr = DecisionTreeRegressor(max_depth=d, min_samples_split=5, min_samples_leaf=5)
        if verbose:
            print(dtr)
            print('max_depth =', d)
            print('{:<25s}{:>20s}{:>20s}'.format('Metric', 'Mean', 'Std Dev'))
        for s in score_list:
            dtr_cvs = cross_val_score(dtr, x_train, y_train, scoring=s, cv=cross_val, error_score='raise')
            mean = dtr_cvs.mean()
            std = dtr_cvs.std()
            if verbose:
                print('{:<25s}{:>20.4f}{:>20.4f}'.format(s, mean, std))
            if s == selected_test and mean > max_err:
                max_err = mean
                best_depth = d
        if verbose:
            print()

    if verbose:
        print('best_depth =', best_depth)
        print()

    return best_depth

def model_decision_tree(model, x_train, y_train, test_split: float, cross_val: int, verbose = False):
    depth = best_tree_depth(model, x_train, y_train, test_split, cross_val, verbose)
    dt = DecisionTreeRegressor(max_depth=depth, min_samples_split=5, min_samples_leaf=5)
    return dt.fit(x_train, y_train)

def get_dt_stats(data, pred, verbose = False) -> None:
    r2 = metrics.r2_score(data, pred)
    mae = metrics.mean_absolute_error(data, pred)
    mse = metrics.mean_squared_error(data, pred)
    rmse = np.sqrt(mse)
    if verbose:
        print_dt_stats(r2, mae, mse, rmse)

def print_dt_stats(r2: float, mae: float, mse: float, rmse: float) -> None:
    print('R-squared:', r2)
    print('Mean absolute error:', mae)
    print('Mean squared error', mse)
    print('Root mean squared error', rmse)
    print()

def decision_tree(model, test_split: float, cross_val: int, verbose = False):
    # Set up training and validation sets as numpy arrays
    x = np.asarray(model.drop(['ConvertedCompYearly'], axis=1)) # All data except salaries
    y = np.asarray(model['ConvertedCompYearly']) # Salaries only
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=test_split, random_state=12345)

    # Generate decision tree model
    dt = model_decision_tree(model, x_train, y_train, test_split, cross_val, verbose)

    # Print accuracy statistics of training and validation sets
    tr_pred = dt.predict(x_train)
    va_pred = dt.predict(x_validate)
    get_dt_stats(y_train, tr_pred, verbose) # Training results
    get_dt_stats(y_validate, va_pred, verbose) # Validation results

    # Print most important attributes to the decision tree
    ind_model = model.drop(['ConvertedCompYearly'], axis=1)
    lst = list(ind_model.columns)
    col_impt = pd.Series(dt.feature_importances_, index=lst)
    print(col_impt.nlargest(10).sort_values(ascending=False))

    # Scatterplot of actual vs predicted values of testing set
    fig, ax = plt.subplots()
    ax.scatter(y_validate, va_pred)
    ax.plot([y_validate.min(), y_validate.max()], [y_validate.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.title('DT: Predicted vs Actual ConvertedCompYearly')
    plt.show()

    # Show decision tree image
    #dot_data = export_graphviz(dt, filled=True, rounded=True, feature_names=lst, out_file=None)
    #graph = graph_from_dot_data(dot_data)
    #graph = gv.Source(dot_data)
    #graph.render('tree', view=True)
    #graph_pdf.view('tree')
    #print(graph)