# coding: utf8
import copy
import sys

import numpy as np


def load_data_set(dataset, path_to_source):
    sys.path.insert(0, path_to_source + '/DataSets/')
    log_tf = False
    if dataset == 'boston':
        from handler_UCI_Boston import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
        Y = np.log(Y) # Using Logarithmic Data
        log_tf = True
    elif dataset == 'auto_mpg':
        from handler_AutoMPG import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
    elif dataset == 'concrete':
        from handler_UCI_Concrete import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
    elif dataset == 'ames':
        from handler_AmesHousing import read_all
        data = read_all(scaling = 'MeanVar', feature_subset = 'intuitive')
        X, Y = data[:,:-1], data[:,-1]
        Y = np.log(Y) # Using Logarithmic Data
        log_tf = True
    elif dataset == 'istanbul':
        from handler_UCI_IstanbulStockExchange import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
    elif dataset == 'airquality':
        from handler_UCI_AirQuality import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
    elif dataset == 'powerplant':
        # Not used
        from handler_UCI_CombinedCyclePowerPlant import read_all
        data = read_all()
        X, Y = data[:,:-1], data[:,-1]
        Y = np.log(Y) # Using Logarithmic Data
        log_tf = True
    elif dataset == 'skillcraft':
        from handler_UCI_SkillCraft import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
        Y = np.log(Y) # Using Logarithmic Data
        log_tf = True
    elif dataset == 'airfoil':
        # Not used
        from handler_UCI_Airfoil import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
    elif dataset == 'yacht':
        from handler_UCI_Yacht import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
        Y = np.log(Y) # Using Logarithmic Data
        log_tf = True
    elif dataset == 'EUStockExchange':
        from handler_EUStockExchange import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
        Y = np.log(Y)
        log_tf = True
    else:
        raise NotImplementedError('Load data: Data set does not exist.')
    return X, Y, log_tf


def load_estimator(estimator_kwargs, path_to_source):
    sys.path.insert(0, path_to_source + '/simple_estimation/')
    sys.path.insert(0, path_to_source + '/nsim_algorithm/')
    sys.path.insert(0, path_to_source + '/Python-ELM/')
    name = estimator_kwargs['estimator']
    copy_dict = copy.deepcopy(estimator_kwargs)
    del copy_dict['estimator']
    if name == 'SIRKnn':
        from simple_estimation.estimators.SIRKnn import SIRKnn
        estim = SIRKnn(**copy_dict)
    elif name == 'SAVEKnn':
        from simple_estimation.estimators.SAVEKnn import SAVEKnn
        estim = SAVEKnn(**copy_dict)
    elif name == 'PHDKnn':
        from simple_estimation.estimators.PHDKnn import PHDKnn
        estim = PHDKnn(**copy_dict)
    elif name == 'nsim':
        from nsim_algorithm.estimator import NSIM_Estimator
        estim = NSIM_Estimator(**copy_dict)
    elif name == 'knn':
        from sklearn.neighnors import KNeighborsRegressor
        estim = KNeighborsRegressor()
    elif name == 'linreg':
        from sklearn.linear_model import LinearRegression
        estim = LinearRegression()
    elif name == 'isotron':
        from simple_estimation.estimators.Isotron import IsotronCV
        estim = IsotronCV(**copy_dict)
    elif name == 'slisotron':
        from simple_estimation.estimators.Slisotron import SlisotronCV
        estim = SlisotronCV(**copy_dict)
    elif name == 'elm':
        from elm import ELMRegressor
        estim = ELMRegressor(**copy_dict)
    else:
        raise NotImplementedError('Load estimator: Estimator does not exist.')
    return estim
