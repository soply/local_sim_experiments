import os
import sys
import time

import numpy as np
import sklearn
import json
from utils import load_data_set, load_estimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import KFold, ShuffleSplit

# Path leading to source of data set and estimator definitions
path_to_source = '../..'


def run_experiment(arguments):
    # Load data set
    X, Y, log_tf = load_data_set(arguments['dataset'], path_to_source)
    estim = load_estimator(arguments['estimator_kwargs'], path_to_source)
    # Prepare for experiments
    n_test_sets = arguments['n_test_sets']
    test_size = arguments['test_size']
    param_grid = arguments['param_grid'] # Parameter grid for estimator to CV over
    cv_folds = arguments['cv_folds']
    n_jobs = arguments['n_jobs']
    kf = ShuffleSplit(n_splits = arguments['n_test_sets'],
                        test_size = arguments['test_size'])
    test_error = np.zeros(n_test_sets)
    best_parameters = {}
    test_iter = 0
    computational_time = np.zeros(n_test_sets)
    # Extra array to store dot products if estimator is nsim
    almost_linearity_param = np.zeros(n_test_sets)
    for idx_train, idx_test in kf.split(X):
        start = time.time()
        reg = GSCV(estimator = estim, param_grid = param_grid,
                   scoring = 'neg_mean_squared_error', iid = False,
                   cv = cv_folds, verbose = 0,
                   pre_dispatch = n_jobs,
                   error_score = np.nan,
                   refit = True) # If estimator fitting raises an exception
        X_train, Y_train = X[idx_train,:], Y[idx_train]
        X_test, Y_test = X[idx_test,:], Y[idx_test]
        reg = reg.fit(X_train, Y_train)
        Y_predict = reg.best_estimator_.predict(X_test)
        end = time.time()
        best_parameters[test_iter] = reg.best_params_
        if arguments['estimator_kwargs']['estimator'] in ['isotron', 'slisotron']:
            best_parameters[test_iter] = reg.best_estimator_.n_iter_cv()
        if log_tf:
            test_error[test_iter] = np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(Y_predict)))
        else:
            test_error[test_iter] = np.sqrt(mean_squared_error(Y_test, Y_predict))
        computational_time[test_iter] = end - start
        if arguments['estimator_kwargs']['estimator'] == 'nsim':
            almost_linearity_param[test_iter] = reg.best_estimator_.measure_almost_linearity()
        test_iter += 1
        print best_parameters
        print test_error
    # Save results
    mean_error = np.mean(test_error)
    std_error = np.std(test_error)
    mean_computational_time = np.mean(computational_time)
    mean_almost_linearity_param = np.mean(almost_linearity_param)
    filename = arguments['filename']
    filename_mod = filename
    save_itr = 0
    while os.path.exists('../results/' + filename_mod + '/'):
        save_itr += 1
        filename_mod = filename + '_' + str(save_itr)
    else:
        os.makedirs('../results/' + filename_mod + '/')
        np.save('../results/' + filename_mod + '/test_errors.npy', test_error)
        np.savetxt('../results/' + filename_mod + '/test_errors.txt', test_error)
        np.savetxt('../results/' + filename_mod + '/computational_time.txt', computational_time)
        np.savetxt('../results/' + filename_mod + '/computational_time_summary.txt', [mean_computational_time])
        np.savetxt('../results/' + filename_mod + '/test_errors_summary.txt', np.array([mean_error, std_error]))
        np.save('../results/' + filename_mod + '/best_params.npy', best_parameters)
        if arguments['estimator_kwargs']['estimator'] == 'nsim':
            np.savetxt('../results/' + filename_mod + '/almost_linearity_param.txt', almost_linearity_param)
            np.savetxt('../results/' + filename_mod + '/almost_linearity_summary.txt', [mean_almost_linearity_param])
        with open('../results/' + filename_mod + '/best_params_json.txt', 'w') as file:
            file.write(json.dumps(best_parameters, indent=4))
        with open('../results/' + filename_mod + '/log.txt', 'w') as file:
            file.write(json.dumps(arguments, indent=4))




if __name__ == '__main__':
    # Get number of jobs from sys.argv
    if len(sys.argv) > 1:
        n_jobs = int(sys.argv[1])
    else:
        n_jobs = 4 # Default 4 jobs
    print 'Using n_jobs = {0}'.format(n_jobs)
    # Data set
    arguments = {
        'filename' : 'test_elm', # Name to store results
        'dataset' : 'auto_mpg', # Data set identifier
        'n_jobs' : n_jobs, # number of jobs to run in cv mode
        'n_test_sets' : 30, # number of repititions of the experiment
        'test_size' : 0.15, # size of the test set
        'cv_folds' : 5, # CV is used for parameter choices
        'estimator_kwargs' : { # Estimator details
            'estimator' : 'elm',
            'alpha' : 0.0,
            'activation_func' : 'sigmoid',
            # 'qp_solver' : None
        },
        'param_grid' : {
            'n_hidden' : np.arange(2,50).tolist()
        }
    }
    run_experiment(arguments)
