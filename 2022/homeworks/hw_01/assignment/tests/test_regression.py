from re import X
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

import main as main


@pytest.mark.parametrize("reg", [('ridge'), (None), ('lasso')])
def test_regression_fit(reg):

    X, y = main.load_and_normalize_data()

    given = main.fit_linear_regression(X,y, lmbda=0.5, regularization=reg)
    expected = pickle.load(open(f'results/expected/fit-{reg}.pkl', 'rb'))

    assert np.isclose(given[0], expected[0]).all()
    assert np.isclose(given[1], expected[1])


def test_regression_fit2():
    lr = LinearRegression(fit_intercept = False)
    X = np.array([[1.0, 2.0, 7.0], [2.0, 5.0, 4.0], [7.0, 8.0, 9.0]])
    y = np.array([1.0, 2.0, 4.0])
    lr.fit(X, y)
    w, c = main.fit_linear_regression(X, y)
    assert np.isclose(w, lr.coef_).all()
    assert np.isclose(c, lr.intercept_).all()