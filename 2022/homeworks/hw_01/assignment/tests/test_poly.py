import numpy as np

import main as main

def test_poly():
    X = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    assert main.add_poly_features(X).shape == (2,6)
    assert main.add_poly_features(X)[0,0] == 1.0
    assert main.add_poly_features(X)[0,4] == 4.0
    assert main.add_poly_features(X)[1,5] == 36.0

