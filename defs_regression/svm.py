"function (and parameter space) definitions for hyperband"
"regression with gradient boosting"

from copy import copy

from sklearn.svm import SVR

from common_defs import *

iterations_per_n = 500

space = {
    'scaler': hp.choice('s', (None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler')),
    'kernel': hp.choice('k', ('linear', 'rbf', 'sigmoid')),
    'epsilon': hp.uniform('e', 0.01, 1),
    'C': hp.choice('c', (0.1, 10, 100, 1000)),
    'gamma': hp.uniform('g', 0.001, 0.1)
}


def get_params():
    params = sample(space)
    return handle_integers(params)


#

def try_params(n_iterations, params, data):
    n_estimators = int(round(n_iterations * iterations_per_n))
    print("n_estimators:", n_estimators)
    pprint(params)
    scaler = copy(params['scaler'])
    del params['scaler']
    clf = SVR(max_iter=n_estimators, verbose=0, **params)

    return train_and_eval_sklearn_regressor(clf, data, scaler=scaler)
