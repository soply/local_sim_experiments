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
    # if os.path.isdir(base_source + 'kNN/'):
    #     data = np.loadtxt(base_source + 'kNN/test_errors_summary.txt')
    #     plt.plot(np.arange(n_levelset-1) + 1, data[0] * np.ones(n_levelset - 1), '-o', label = 'kNN (+/-{0:1.2e}) '.format(data[1]))
    # # add linreg baseline
    # if os.path.isdir(base_source + 'linreg/'):
    #     data = np.loadtxt(base_source + 'linreg/test_errors_summary.txt')
    #     plt.plot(np.arange(n_levelset-1) + 1, data[0] * np.ones(n_levelset - 1), '-o', label = 'LinReg (+/-{0:1.2e}) '.format(data[1]))
    # # add SIRKnn baseline
    # if os.path.isdir(base_source + 'SIRKnn/'):
    #     data = np.loadtxt(base_source + 'SIRKnn/test_errors_summary.txt')
    #     plt.plot(np.arange(n_levelset-1) + 1, data[0] * np.ones(n_levelset - 1), '-o', label = 'SIRKnn (+/-{0:1.2e}) '.format(data[1]))
    # # add SAVEKnn baseline
    # if os.path.isdir(base_source + 'SAVEKnn/'):
    #     data = np.loadtxt(base_source + 'SAVEKnn/test_errors_summary.txt')
    #     plt.plot(np.arange(n_levelset-1) + 1, data[0] * np.ones(n_levelset - 1), '-o', label = 'SAVEKnn (+/-{0:1.2e}) '.format(data[1]))
    plt.legend()
    ax = plt.gca()
    ax.set_yscale('log')
    plt.title(base_source)
    plt.xlabel('# Level sets')
    plt.ylabel('RMSE')
    plt.show()

def plot_n_levelset_dependency_several(base_sources, labels):
    for i, base_source in enumerate(base_sources):
        err = np.zeros(100)
        std = np.zeros(100)
        n_levelset = 1
        while os.path.isdir(base_source + str(n_levelset) + 'lsim'):
            data = np.loadtxt(base_source + str(n_levelset) + 'lsim/test_errors_summary.txt')
            err[n_levelset - 1] = data[0]
            # std[n_levelset - 1] = data[1]
            n_levelset += 1
        err = err[:n_levelset-1]
        if len(err) == 0:
            continue
        # std = std[:n_levelset-1]
        plt.plot(np.arange(n_levelset-1) + 1, err/err[0], '--', label = labels[i])
    plt.legend()
    ax = plt.gca()
    plt.xlabel(r'J')
    plt.ylabel(r'RMSE(J)/RMSE(1)')
    # plt.yscale('log')
    plt.ylim([0.5,2.0])
    plt.show()


def print_average_best_parameters(folder):
    """ Assumes there exists a "best_params.npy" file in folder."""
    A = np.load(folder + '/best_params.npy')
    results = A[()]
    n_runs = len(results.keys())
    n_params = len(results[0].keys())
    params = {}
    print "Processing {0}...".format(folder)
    for i, (key, val) in enumerate(results.iteritems()):
        for j, (key2, val2) in enumerate(val.iteritems()):
            if key2 in params:
                params[key2].append(val2)
            else:
                params[key2] = [val2]
    for key in params.keys():
        print "{0}  : {1}".format(key, np.mean(params[key]))


if __name__ == "__main__":
    # base_source = ['../results/auto_mpg','../results/conrete_','../results/new_ames_',
    #     '../results/airquality','../results/powerplant','../results/skillcraft_',
    #     '../results/boston_','../results/yacht_','../results/EUStockExchange_']
    # labels = ['Auto', 'Concrete', 'Ames Housing', 'Airquality', 'Powerplant', 'Skillcraft', 'Boston', 'Yacht','EUStock']
    # plot_n_levelset_dependency_several(base_source, labels)
    import os
    base_source = [x[0] for x in os.walk("../results/")][1:] # First one is just "../results"
    import pdb; pdb.set_trace()
    for folder in base_source:
        print_average_best_parameters(folder)
