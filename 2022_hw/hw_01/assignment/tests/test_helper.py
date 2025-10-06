from re import X
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

import main as main


def test_mse():
    y = np.array([1.0, 2.0, 2.0, 1.0])
    y_hat = np.array([1.0, 2.0, 2.0, 1.0])
    assert main.mse(y, y_hat) == 0.0
    y_hat = np.array([2.0, 2.0, 3.0, 1.0])
    assert main.mse(y, y_hat) == 0.5


@pytest.mark.parametrize("reg", [('ridge'), (None), ('lasso')])
def test_regression_predict_test(reg):

    # You first need to make the data import work
    X, y = main.load_and_normalize_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, X_validation, y_validation = main.data_split(
        X, y)

    given = main.fit_predict_test(
        X_train, y_train, X_test, y_test, lmbda=1.3, regularization=reg)
    expected = pickle.load(open(f'results/expected/{reg}.pkl', 'rb'))

    assert np.isclose(given['mse_train'], expected['mse_train'])
    assert np.isclose(given['mse_test'], expected['mse_test'])
    assert np.isclose(given['lmbda'], expected['lmbda'])
    assert np.isclose(given['w'], expected['w']).all()
    assert np.isclose(given['c'], expected['c'])


def test_regression_predict():
    lr = LinearRegression()
    X = np.array([[1.0, 2.0, 7.0], [2.0, 5.0, 4.0], [7.0, 8.0, 9.0]])
    y = np.array([1.0, 2.0, 4.0])
    lr.fit(X, y)
    y_hat = lr.predict(X)
    assert np.isclose(y_hat, main.predict(X, lr.coef_, lr.intercept_)).all()
