# decision_tree.py 

import numpy as np
import pandas as pd
import graphviz as gv
import matplotlib.pyplot as plt
from pydotplus.graphviz import *
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import *

def best_tree_depth(model, x_train, y_train, test_split: float, verbose = False):
    score_list = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    search_depths = [4, 5, 6, 7,]
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
            print()

    if verbose:
        print('Best depth =', best_depth)
        print()

    return best_depth

def model_decision_tree(model, x_train, y_train, test_split: float, verbose = False):
    depth = best_tree_depth(model, x_train, y_train, test_split, verbose)
    dt = DecisionTreeRegressor(max_depth=depth, min_samples_split=5, min_samples_leaf=5)
    return dt.fit(x_train, y_train)

def get_dt_stats(data, pred, verbose = False):
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

def decision_tree(model, test_split, verbose = False):
    x = np.asarray(model.drop(['ConvertedCompYearly'], axis=1)) # All data except salaries
    y = np.asarray(model['ConvertedCompYearly']) # Salaries only
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=test_split, random_state=12345)

    dt = model_decision_tree(model, x_train, y_train, test_split, verbose)
    tr_pred = dt.predict(x_train)
    va_pred = dt.predict(x_validate)
    #print(tr_pred)
    #print(len(tr_pred))
    #print(va_pred)
    #print(len(va_pred))

    get_dt_stats(y_train, tr_pred, verbose)
    get_dt_stats(y_validate, va_pred, verbose)

    ind_model = model.drop(['ConvertedCompYearly'], axis=1)
    lst = list(ind_model.columns)
    col_impt = pd.Series(dt.feature_importances_, index=lst)
    print(col_impt.nlargest(10).sort_values(ascending=False))

    fig, ax = plt.subplots()
    ax.scatter(y_validate, va_pred)
    ax.plot([y_validate.min(), y_validate.max()], [y_validate.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.title('Predicted vs Actual ConvertedCompYearly')
    plt.show()

    #dot_data = export_graphviz(dt, filled=True, rounded=True, feature_names=lst, out_file=None)
    #graph = graph_from_dot_data(dot_data)
    #graph = gv.Source(dot_data)
    #graph.render('tree', view=True)
    #graph_pdf.view('tree')
    #print(graph)