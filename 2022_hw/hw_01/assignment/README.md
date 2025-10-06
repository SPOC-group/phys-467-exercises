# Assignment 1 - Linear Regression

At the bottom of this file you will find a reminder on how to run the python scripts.

## Feedback and Grade

See [FEEDBACK.md](FEEDBACK.md)

## Plots from the partical exercises, pushed by you to the repo 

### Dataset size vs. MSE


![expected](results/dataset_size_vs_mse.png)

### Dataset size vs. MSE with ridge regularization

lambda=0.001 | lambda=10 
:-------------------------:|:-------------------------
![expected](results/dataset_size_vs_mse_l2=001.png) |![expected](results/dataset_size_vs_mse_l2=10.png)

### Regularizer parameter vs. coefficients with ridge regularization
![expected](results/regularizer_vs_coefficients_Ridge.png)

### Dataset size vs. optimal regularizer for Ridge
n=50 | n=150 
:-------------------------:|:-------------------------
![expected](results/optimal_lambda_ridge_n50.png) | ![expected](results/optimal_lambda_ridge_n150.png)


### Regularizer parameter vs. coefficients with LASSO regularization

without squared features | with squared features
:-------------------------:|:-------------------------
![expected](results/regularizer_vs_coefficients_LASSO.png) | ![expected](results/regularizer_vs_coefficients_LASSO_polyfeat.png)




## Reminder: How to run the code 
You might also want to check the complete assignment0 if you encounter problems.

### Run python locally

In the root directory of the repository in the terminal, run `conda env create -n ml4phys-a1 --file environment.yml`, this might take a couple of minutes. It installs the correct python environment version and packages for your use.

To activate this interpreter you should call `conda activate ml4phys-a1` in the terminal at root directory of the repository.
Then, your terminal should look like the something like following, where at the start of line in brackets the active environment is indicated.

```bash
(base) ➜  ml4phys-assignment1: conda activate ml4phys-a1   
(ml4phys-a1) ➜  ml4phys-assignment1: 
```

This means that when you run `python` now, all the packages specified in `environment.yml` are available to you!

### Run python tests

In parts we use automated grading, and this means running software tests on your code. 
Tests are small functions living in the `tests\` folder that call other functions from your code and check whether they give the right output for a given input.

Recall that to run the tests you use
`conda activate ml4phys-a1`
`PYTHONPATH=. pytest` or `python -m pytest`
in the terminal in the root directory of the repository.

If you have problems, consult this [thread on moodle](https://moodle.epfl.ch/mod/forum/discuss.php?d=80819#p157821) first.


> :warning: You might notice that we do not hide the tests from you, you can look at them, and even commit modified versions.
You could feel smart and delete the tests. Great, now you get all the assignment points...
Or just copy the test synatx into your own code. Whohoo, yet another way to obtain all the points without much work!
Well, DON'T try either. Because we will check your code for this and if we find this, we consider it as cheating.





