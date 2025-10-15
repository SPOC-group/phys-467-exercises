import pytest

import main as main
import numpy as np

data_13 = np.array([ 0.987428, 0.277449, 0.761089, 0.44333,  0.645304, -0.042653, 0.222528, -0.271035, 0.60417, -1.346039, 0.726136, 0.998025, 0.741762, -0.104098,
 0.505229, -0.776579, 0.837102, -1.099215, 0.593731, 0.532657, 0.406728,
 -0.447084, 0.760166, -1.837265, 0.777222, 0.191438, 1.050959, 0.200049,
 1.83819, -0.941819, 0.609407, -0.886004, -0.126386, 0.742373, 0.62543,
 2.079539, -0.351407, 1.286001, -0.2543, -0.7009, -1.622954, 0.162671,
 -0.521306, 0.847517, -0.528401, -0.476602, 0.354655, 1.721216, -1.08865,
 1.017613, 0.874105, -0.220946, 0.675416, -0.284276, 2.371388, -0.112095,
 0.080414, -0.325792, -1.693454, -0.263153, 0.232037, 1.500923, 0.491957,
 -0.189382, 0.059943, 0.32734, -0.378966, 0.682237, 1.625088, -0.363149,
 1.20753,  0.876326, 0.537245, 0.111926, 0.337784, -0.217393, 1.836869,
 -0.718403, 1.134945, 1.549722, -0.323134, 1.165847, 0.143981, 0.06236,
 -0.074564, -1.613731, 0.547924, 1.443552, -0.70472,  0.847339, -0.085205,
 -1.092689, 0.99789,  1.686037, -0.794252, 0.215802, -1.439736, -0.042007,
 0.486066, 1.75291 ])

def test_data():

    X,y = main.load_and_normalize_data()

    assert type(X) == np.ndarray
    assert type(y) == np.ndarray

    assert X.shape == (1000, 100)
    assert y.shape == (1000,)

    assert np.isclose(X[13], data_13).all()

    
def test_normalization():

    X,y = main.load_and_normalize_data()

    assert y.std() == 1


def test_data_summary():

    X,y = main.load_and_normalize_data()

    summary = main.data_summary(X,y)

    assert type(summary) == dict
    assert len(summary) == 6
    assert np.isclose(summary['X_mean'],0.00124193778999)
    assert np.isclose(summary['X_std'],0.999322655)
    assert np.isclose(summary['X_min'],-4.406934)
    assert np.isclose(summary['X_max'],4.185027)
    assert np.isclose(summary['y_mean'],-0.42947724566)
    assert np.isclose(summary['y_std'],1.0)
