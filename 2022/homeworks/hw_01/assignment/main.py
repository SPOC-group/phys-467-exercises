import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd



def load_and_normalize_data():
    # load the numpy arrays inputs and labels from the data folder
    """
    Hint for autograding:
    Instead of using 'C:User/freya/Docs/EPFL/ML1/.../files/file1.txt',
    use "files/file1.txt", when your main script is in the same folder 
    as the files folder, regardless of the operating system.

    !! note the forward slashes !!

        assignment-1
        |-- main.py
        |-- files
        |--file1.txt
        |--file2.txt
        
    """
    # TODO
    X, y = ...

    # normalize the target y
    # TODO

    return X, y


def data_summary(X, y):

    # return several statistics of the data
    # TODO
    X_mean = ...
    X_std = ...
    y_mean = ...
    y_std = ...
    X_min = ...
    X_max = ...

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
    Fit a ridge regression model to the data, with regularization parameter lmbda and a given
    regularization method.
    If the selected regularization method is None, fit a linear regression model without a regularizer.

    !! Do not fit the intersept in all cases.

    y = wx+c

    X: 2D numpy array of shape (n_samples, n_features)
    y: 1D numpy array of shape (n_samples,)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None

    Returns: The coefficients and intercept of the fitted model.
    """

    # TODO: use the sklearn linear_model module

    w = ... # coefficients
    c = ... # intercept

    return w, c


def predict(X, w, c):
    """
    Return a linear model prediction for the data X.

    X: 2D numpy array of shape (n_samples, n_features) data
    w: 1D numpy array of shape (n_features,) coefficients
    c: float intercept

    Returns: 1D numpy array of shape (n_samples,)
    """
    # TODO
    y_pred = ...
    return y_pred


def mse(y_pred, y):
    """
    Return the mean squared error between the predictions and the true labels.

    y_pred: 1D numpy array of shape (n_samples,)
    y: 1D numpy array of shape (n_samples,)

    Returns: float
    """

    # TODO
    MSE = ...

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

    Returns: The coefficients and intercept of the fitted model, the training and testing MSEs in a dictionary.
    """

    w, c = ... # TODO

    results = {
        'mse_train':...,
        'mse_test': ...,
        'lmbda': ...,
        'w': ...,
        'c': ...,
    }

    return results


def plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=0.0, regularization=None, filename=None):
    """
    Plot the training and testing MSEs against the regularization parameter alpha.
    Use the functions you just wrote.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    alphas: list of values, the dataset percentage to be checked (alpha=n/d)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """
    
    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here

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
    plot_coefs: list of integers, the coefficients of w to be plotted
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """

    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here

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

    # TODO
    X_poly = ...

    return X_poly

def optimize_lambda(X_train,X_test,X_validation,y_train,y_test,y_validation, lmbdas,filename):
    """
    Optimize the regularization parameter lambda for the training data for ridge and plot the validation error against the parameter lambda
    Show the best parameter lambda and the corresponding validation error.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    X_validation: 2D numpy array of shape (n_validation_samples, n_features)
    y_validation: 1D numpy array of shape (n_validation_samples,)
    lmbdas: list of values, the regularization parameter, the bounds of the optimization range

    Returns: The best regularization parameter and the corresponding validation error.
    """

    regularization='ridge'

    # TODO: Experiment code goes here


    best_test_mse = ...
    best_lmbda = ...

    # TODO: Plotting code goes here

    plt.axvline(best_lmbda, color='red', label='Best Lambda')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()

    return best_test_mse, best_lmbda
  

if __name__ == "__main__":

    """ 
    !!!!! DO NOT CHANGE THE NAME OF THE PLOT FILES !!!!!
    They need to show in the Readme.md when you submit your code.

    It is executed when you run the script from the command line.
    'conda activate ml4phys-a1'
    'python main.py'

    This already includes the code for generating all the relevant plots.
    You need to fill in the ...
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
    alphas = ... 
    plot_dataset_size_vs_mse(..., filename='dataset_size_vs_mse')

    print('Plotting dataset size vs. mse curve for L2 ...')
    alphas = ...
    lmbda = 0.001
    plot_dataset_size_vs_mse(..., filename='dataset_size_vs_mse_l2=001')

    lmbda = 10.0
    plot_dataset_size_vs_mse(..., filename='dataset_size_vs_mse_l2=10')

    ## Exercise 5.
    print('Plotting regularizer vs. coefficient curve...')
    lmbdas = ...
    plot_coeffs = ...
    plot_regularizer_vs_coefficients(..., filename='regularizer_vs_coefficients_Ridge')
    

    ## Exercise 6.
    print('Find the optimal parameters for the Ridge regression...')
    lmbdas = ...
    n_ = 80
    lmbda, gen_error = optimize_lambda(..., lmbdas,filename='optimal_lambda_ridge_n50')
    print(n_, lmbda, gen_error)
    n_ = 150
    lmbda, gen_error = optimize_lambda(..., lmbdas,filename='optimal_lambda_ridge_n150')
    print(n_, lmbda, gen_error)
    print()

    ## Exercise 7.
    X_train_poly = add_poly_features(X_train) 
    X_test_poly = add_poly_features(X_test) 

    lmbdas = ...
    plot_coeffs = ...
    plot_regularizer_vs_coefficients(..., filename='regularizer_vs_coefficients_LASSO')
    
    plot_coeffs = ...
    plot_regularizer_vs_coefficients(..., filename='regularizer_vs_coefficients_LASSO_polyfeat')

    print('Done. All results saved in the results folder.')

