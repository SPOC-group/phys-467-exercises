from re import X
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

import main as main


def test_optimizer():
    # You first need to make the data import work
    X, y = main.load_and_normalize_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, X_validation, y_validation = main.data_split(
        X, y)

    n_ = 80
    lmbdas = np.arange(0.01, 100., 0.5) 
    gen_error, lmbda  = main.optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename=f'optimal_lambda_ridge_n{n_}')
    assert np.isclose(gen_error, 0.862233609955,atol=0.001)
    assert np.isclose(lmbda, 33.51,atol=0.6)
