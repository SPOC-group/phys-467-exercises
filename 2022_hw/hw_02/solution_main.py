import numpy as np
import matplotlib.pyplot as plt

def generate_data(rho, n, p_e):
    s     = np.sign(np.random.normal(0, 1, n))
    y     = np.zeros((n, n))
    edges = np.zeros((n, n))

    for i in range(1, n):
        for j in range(i):
            edges[i, j] = edges[j, i] = np.random.binomial(1, p_e)
            y[i, j] = y[j, i] = ( s[i] * s[j] * [1.0, -1.0][np.random.binomial(1, rho)] ) * edges[i, j]
    return s, y, edges

def run_pca(y):
    eigvals, eigvecs = np.linalg.eigh(y)
    return eigvecs[:, np.argmax(eigvals)]

def log_ratio_posterior(s, i, y, rho, edges):
    n_plus  = np.count_nonzero(y[i] == s[i]   * s)
    n_minus = np.count_nonzero(y[i] == - s[i] * s)
    return (n_minus - n_plus) * np.log( (1 - rho) / rho )

def run_mcmc(y, n_iter, rho, edges, s = None):
    n = len(y)
    if s is None:
        s_list = [ np.sign(np.random.normal(0, 1, n)) ]
    else:
        s_list = [ s ]
    
    for t in range(n_iter - 1):
        s  = np.copy(s_list[-1])
        i = np.random.randint(low = 0, high = n)

        log_ratio = log_ratio_posterior(s, i, y, rho, edges)
        if np.log(np.random.uniform()) < min(0.0, log_ratio):
            s[i] = - s[i]
        s_list.append(s)
    return s_list

### functions for the plots 

def plot_mcmc_mixing_time(n, rho, p_e):
    n_iter   = 50000

    s, y, edges = generate_data(rho, n, p_e)

    # start mcmc from random vector
    s_list_mcmc = np.array(run_mcmc(y, n_iter, rho, edges))
    # run starting from teacher
    s_list_mcmc_2 = np.array(run_mcmc(y, n_iter, rho, edges, s))

    overlaps   = np.abs(s_list_mcmc   @ s / n)
    s_overlaps = np.abs(s_list_mcmc_2 @ s / n)

    plt.plot(overlaps)
    plt.plot(s_overlaps)

    plt.title(f'Q as a function of t for rho = {rho}')
    plt.show()

def plot_mcmc_performance(n, rho, p_e):
    n_iter   = 100000
    T_therm  = 30000

    s, y, edges = generate_data(rho, n, p_e)
    s_list_mcmc = np.array(run_mcmc(y, n_iter, rho, edges))

    s_bo = np.sign(np.mean(s_list_mcmc[T_therm:], axis=0))
    return np.abs(np.mean(s_bo * s))

def plot_mcmc_pca_performance(n, n_iter, iter_burn):
    n_seeds = 20
    n_p_e   = 10

    rho     = 0.25
    p_e_list= np.linspace(2.0 / n, 6.0 / n, n_p_e)

    q_pca_list = np.zeros((n_p_e, n_seeds))
    q_bo_list  = np.zeros((n_p_e, n_seeds))

    for i, p_e in enumerate(p_e_list):
        for seed in range(n_seeds):
            s, y, edges = generate_data(rho, n, p_e)
            
            s_pca       = np.sign(run_pca(y))
            s_list_mcmc = run_mcmc(y, n_iter, rho, edges, s)
            s_list_mcmc = np.array(s_list_mcmc)
            s_bo        = np.sign(np.mean(s_list_mcmc[iter_burn:], axis=0))

            q_pca = np.abs(np.mean(s_pca * s))
            q_bo  = np.abs(np.mean(s_bo  * s))

            q_pca_list[i, seed]  = q_pca
            q_bo_list [i, seed]  = q_bo

    plt.plot(p_e_list, np.mean(q_pca_list, axis = 1), marker='.', label='PCA')
    plt.plot(p_e_list, np.mean(q_bo_list , axis = 1),  marker='.', label='BO')

    plt.xlabel("$p_e$")
    plt.ylabel("Q")
    plt.legend()
    plt.show()

### 

def question5():
    n        = 500
    p_e      = 4.0 / n
    
    rho_list = np.linspace(0.1, 0.35, 10)
    n_seeds  = 10

    overlaps = np.zeros((len(rho_list), n_seeds))

    for a, rho in enumerate(rho_list):
        for i in range(n_seeds):
            overlaps[a, i] = plot_mcmc_performance(n, rho, p_e)

    plt.plot(rho_list, np.mean(overlaps, axis=1))
    plt.xlabel('$\\rho$')
    plt.ylabel('overlap')
    plt.show()

    ## 

    n        = 500
    p_e      = 4.0 / n
    rho_list = np.linspace(0.1, 0.35, 10)

    for a, rho in enumerate(rho_list):
        plot_mcmc_mixing_time(n, rho, p_e)

def question6():
    plot_mcmc_pca_performance(n=500, n_iter=100000, iter_burn=50000)

question5()
question6()