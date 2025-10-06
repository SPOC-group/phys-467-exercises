import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd



def load_and_normalize_data():
    # load the numpy arrays inputs and labels from the data folder
    X = np.loadtxt("./data/inputs.txt")
    y = np.loadtxt("./data/labels.txt")

    # normalize the target y
    y = y/y.std()

    return X, y


def data_summary(X, y):

    # return several statistics of the data
    X_mean = X.mean()
    X_std = X.std()
    y_mean = y.mean()
    y_std = y.std()
    X_min = X.min()
    X_max = X.max()

    return {'X_mean': X_mean,
            'X_std': X_std, 
            'X_min': X_min, 
            'X_max': X_max, 
            'y_mean': y_mean, 
            'y_std': y_std}


def data_split(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.25, random_state=4)

    return X_train, X_test, y_train, y_test, X_validation, y_validation


def fit_linear_regression(X, y, lmbda=0.0, regularization=None):
    """
    Fit a ridge regression model to the data, with regularization parameter alpha if alpha > 0.
    Otherwise, fit a linear regression model.

    !! Do not fit the intersept in all cases.

    y = wx+c

    X: 2D numpy array of shape (n_samples, n_features)
    y: 1D numpy array of shape (n_samples,)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None

    Returns: The coefficients and intercept of the fitted model.
    """

    if regularization == 'ridge':
        model = Ridge(alpha=lmbda,fit_intercept = False)
    elif regularization == 'lasso':
        model = Lasso(alpha=lmbda,fit_intercept = False)
    elif regularization is None:
        model = LinearRegression(fit_intercept = False)
    else:
        raise AttributeError("Invalid regularization method")

    model.fit(X, y)

    w = model.coef_
    c = model.intercept_

    return w, c


def predict(X, w, c):
    """
    Return a linear model prediction for the data X.

    X: 2D numpy array of shape (n_samples, n_features)
    w: 1D numpy array of shape (n_features,)
    c: float

    Returns: 1D numpy array of shape (n_samples,)
    """

    y_pred = X @ w + c
    return y_pred


def mse(y_pred, y):
    """
    Return the mean squared error between the predictions and the true labels.

    y_pred: 1D numpy array of shape (n_samples,)
    y: 1D numpy array of shape (n_samples,)

    Returns: float
    """

    MSE = mean_squared_error(y, y_pred)

    return MSE



def fit_predict_test(X_train, y_train, X_test, y_test, lmbda=0.0, regularization=None):
    """
    Fit a linear regression model, possibly with L2 regularization, to the training data.
    Record the training and testing MSEs.
    Use methods you wrote before

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbda: float, regularization parameter

    Returns: The coefficients and intercept of the fitted model, the training and testing MSEs.
    """

    w, c = fit_linear_regression(X_train, y_train, lmbda, regularization)

    results = {
        'mse_train': mse(predict(X_train, w, c), y_train),
        'mse_test': mse(predict(X_test, w, c), y_test),
        'lmbda': lmbda,
        'w': w,
        'c': c,
    }

    return results


def plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=0.0, regularization=None, filename=None):
    """
    Plot the training and testing MSEs against the regularization parameter alpha.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    alphas: list of values, the dataset percentage to be checked
    lmbda: float, regularization parameter
    """

    results = []
    for alpha in alphas:
        samples = int(alpha * X_train.shape[1])
        res = fit_predict_test(
            X_train[:samples], y_train[:samples], X_test, y_test,lmbda, regularization)
        res['alpha'] = alpha
        res['lmdba'] = lmbda
        results.append(res)

    results = pd.DataFrame(results)

    plt.scatter(results['alpha'], results['mse_train'],
                label='Train', marker='.')
    plt.scatter(results['alpha'], results['mse_test'],
                label='Test', marker='.')
    plt.xlabel('alpha=n/d')
    plt.ylabel('mse')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()



def plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs, regularization='ridge', filename=None):
    """
    Plot the coefficients of the fitted model against the regularization parameter alpha.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbdas: list of values, the regularization parameter
    """

    results = []
    for lmbda in lmbdas:
        res = fit_predict_test(X_train, y_train, X_test, y_test, lmbda, regularization)
        for i, v in enumerate(res['w']):
            res[f"W{i}"] = v
        results.append(res)

    results = pd.DataFrame(results)

    if plot_coefs is None:
        plot_coefs = range(X_train.shape[1])

    for i in plot_coefs:
        plt.scatter(results['lmbda'],
                    results[f"W{i}"], label=f"W{i}", marker='.')
    plt.xlabel('lmbda')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()

def add_poly_features(X):
    """
    Add squared features to the data X and return a new vector X_poly that contains
    X_poly[i,j]   = X[i,j]
    X_poly[i,2*j] = X[i,j]^2

    X: 2D numpy array of shape (n_samples, n_features)

    Returns: 2D numpy array of shape (n_samples, 2 * n_features ) with the normal and squared features
    """

    X_poly = np.hstack((X, X**2))

    return X_poly

def optimize_lambda(X_train,X_test,X_validation,y_train,y_test,y_validation, lmbdas,filename):
    """
    Optimize the regularization parameter lambda for the training data.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbdas: list of values, the regularization parameter

    Returns: The best regularization parameter
    """

    regularization='ridge'

    results = []
    for lmbda in lmbdas:
        res = fit_predict_test(X_train, y_train, X_validation, y_validation, lmbda, regularization)
        results.append(res)

    results = pd.DataFrame(results)
    best_model = results[results['mse_test'] == results['mse_test'].min()].iloc[0]

    

    test_mse = mse(predict(X_test, best_model['w'], best_model['c']),y_test)
    best_lmbda = best_model['lmbda']

    plt.plot(results['lmbda'], results['mse_test'], label='Validation Error', marker='.')
    plt.axvline(best_lmbda, color='red', label='Best Lambda')
    plt.xlabel('lambda')
    plt.ylabel('validation error')
    plt.savefig(f'results/{filename}.png')
    plt.clf()

    return test_mse, best_lmbda
  

if __name__ == "__main__":

    """ 
    !!!!! DO NOT CHANGE THE NAME OF THE PLOT FILES !!!!!
    They need to show in the Readme.md when you submit your code.

    It is executed when you run the script from the command line.
    'conda activate ml4phys-a1'
    'python main.py'

    This already includes the code for generating all the relevant plots.
    """

    ## Exercise 1.
    # Load the data
    X, y = load_and_normalize_data()
    print("Successfully loaded and normalized data.")
    print(data_summary(X, y))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, X_validation, y_validation = data_split(X, y)

    n, d = X_train.shape

    ## Exercise 4.
    # Plot the learning curves
    print('Plotting dataset size vs. mse curve ...')
    alphas = np.arange(2, int(n/d), 0.02)
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, filename='dataset_size_vs_mse')

    print('Plotting dataset size vs. mse curve for L2 ...')
    alphas = np.arange(0.1, 2.5, 0.02)
    lmbda = 0.001
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda, regularization='ridge', filename='dataset_size_vs_mse_l2=001')

    lmbda = 10.0
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda, regularization='ridge', filename='dataset_size_vs_mse_l2=10')

    ## Exercise 5.
    print('Plotting regularizer vs. coefficient curve...')
    lmbdas = np.arange(0.1, 400., 3) 
    plot_coeffs = [0,3,7,8]
    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,plot_coefs=plot_coeffs, regularization='ridge', filename='regularizer_vs_coefficients_Ridge')
    

    ## Exercise 6.
    print('Find the optimal parameters for the Ridge regression...')
    lmbdas = np.arange(0.01, 100., 0.5) 
    n_ = 80
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='optimal_lambda_ridge_n50')
    print(n_, lmbda, gen_error)
    n_ = 150
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='optimal_lambda_ridge_n150')
    print(n_, lmbda, gen_error)
    print()

    ## Exercise 7.
    X_train_poly = add_poly_features(X_train) 
    X_test_poly = add_poly_features(X_test) 

    lmbdas = np.arange(0.001, 1.2, 0.005)
    plot_coeffs = [0,1,2,6,7]
    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas, plot_coefs=plot_coeffs, regularization='lasso', filename='regularizer_vs_coefficients_LASSO')
    
    plot_coeffs = [0,1,2,6,7,129,154]
    plot_regularizer_vs_coefficients(X_train_poly, y_train, X_test_poly, y_test, lmbdas, plot_coefs=plot_coeffs, regularization='lasso', filename='regularizer_vs_coefficients_LASSO_polyfeat')

    print('Done. All results saved in the results folder.')

