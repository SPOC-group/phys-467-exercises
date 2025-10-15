# Assignment 1 - Feedback

While the autograde badge and the automatically generated plots will not show locally on your computer, you can look at it on the landing page of the github repository.


## Grade
- Autograded: [![Points badge](../../blob/badges/.github/badges/points-bar.svg)](../../actions) 
- Exercise 4: __/10 points (manual)
- Exercise 5: __/10 points (manual)
- Exercise 6: __/6 points (manual) + (4P auto)
- Exercise 7: __/3 points (manual) + (3P auto)

## Plots
Automatically generated from your code

### Dataset size vs. MSE

![expected](../../blob/badges/.github/results/dataset_size_vs_mse.png)

### Dataset size vs. MSE with ridge regularization

lambda=0.001 | lambda=10 
:-------------------------:|:-------------------------
![expected](../../blob/badges/.github/results/dataset_size_vs_mse_l2=001.png) |![expected](../../blob/badges/.github/results/dataset_size_vs_mse_l2=10.png)

### Regularizer parameter vs. coefficients with ridge regularization
![expected](../../blob/badges/.github/results/regularizer_vs_coefficients_Ridge.png)

### Dataset size vs. optimal regularizer for Ridge
n=50 | n=150 
:-------------------------:|:-------------------------
![expected](../../blob/badges/.github/results/optimal_lambda_ridge_n50.png) | ![expected](../../blob/badges/.github/results/optimal_lambda_ridge_n150.png)


### Regularizer parameter vs. coefficients with LASSO regularization

without squared features | with squared features
:-------------------------:|:-------------------------
![expected](../../blob/badges/.github/results/regularizer_vs_coefficients_LASSO.png) | ![expected](../../blob/badges/.github/results/regularizer_vs_coefficients_LASSO_polyfeat.png)


