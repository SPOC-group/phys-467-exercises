import numpy as np
import matplotlib.pyplot as plt

def generate_data(rho, n, p_e):
    """
    arguments : 
        - rho   : proba that the sign is flipped
        - n     : number of nodes
        - p_e   : proba of an edge 
    returns   : 
        - s    : vector of size n containing the labels 
        - y    : n x n symmetric matrix with 0 diagonal
        - edges: n x n matrix encoding the edges
    """
    pass

def log_ratio_posterior(s, i, y, rho, edges):
    """
    arguments : 
        - s     : vector containing the labels
        - i     : index such that s[i] is flipped
        - y     : n x n matrix
        - rho   : flip probability
        - edges : matrix encoding the edges in the graph
    returns   : 
        - log( P(s' | Y) / P(s | Y) ) where s' is the vector defined as s'[i] = -s[i] and s[j] = s[j] for j != i
    """
    pass 

def run_pca(y):
    """
    arguments : 
        - y : n x n matrix
    returns   : 
        - eigenvector corresponding to the highest eigenvalue of y
    """
    pass 

def run_mcmc(y, T, rho, edges):
    """
    arguments : 
        - y   : n x n matrix
        - T   : number of iterations for mcmc
        - rho : flip probability
        - edges : matrix encoding the edges in the graph
    returns   : 
        - list of length T containing the mcmc iterates
    """
    pass 

def question_5():
    """
    Here goes the code for question 5 : plot the mixing time and overlap of the MCMC
    """
    pass

def question_6():
    """
    Here goes the code for question 6 : compare the performance of PCA and BO
    """
    pass

# Un-comment the line depending on the function you want to run
# question_5()
# question_6()
