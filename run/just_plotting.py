# coding: utf8

import matplotlib.pyplot as plt
import numpy as np
import os


# Plotting n_levelset dependency

def plot_n_levelset_dependency(base_source):
    err = np.zeros(100)
    std = np.zeros(100)
    n_levelset = 1
    while os.path.isdir(base_source + str(n_levelset) + 'lsim'):
        data = np.loadtxt(base_source + str(n_levelset) + 'lsim/test_errors_summary.txt')
        err[n_levelset - 1] = data[0]
        std[n_levelset - 1] = data[1]
        n_levelset += 1
    err = err[:n_levelset-1]
    std = std[:n_levelset-1]
    plt.errorbar(np.arange(n_levelset-1) + 1, err, std, fmt = 'o', label = 'NSIM')
    # add knn baseline
    if os.path.isdir(base_source + 'kNN/'):
        data = np.loadtxt(base_source + 'kNN/test_errors_summary.txt')
        plt.plot(np.arange(n_levelset-1) + 1, data[0] * np.ones(n_levelset - 1), '-o', label = 'kNN (+/-{0:1.2e}) '.format(data[1]))
    # add linreg baseline
    if os.path.isdir(base_source + 'linreg/'):
        data = np.loadtxt(base_source + 'linreg/test_errors_summary.txt')
        plt.plot(np.arange(n_levelset-1) + 1, data[0] * np.ones(n_levelset - 1), '-o', label = 'LinReg (+/-{0:1.2e}) '.format(data[1]))
    plt.legend()
    ax = plt.gca()
    # ax.set_yscale('log')
    plt.title(base_source)
    plt.xlabel('# Level sets')
    plt.ylabel('RMSE')
    plt.show()

if __name__ == "__main__":
    base_source = '../results/skillcraft_'
    plot_n_levelset_dependency(base_source)
