# coding: utf8
"""
Contains utilities such as data set loader and estimator loader.
"""
import copy
import sys

import numpy as np


def load_data_set(dataset, path_to_source):
    """
    Loads a specific data set given by the identifier 'dataset'. Path to source
    should contain the data set handlers given in repository "github.com/soply/db_hand".

    Parameters
    ------------
    dataset : string
        Identifier for the data set

    path_to_source : string
        Folder where data set git repository "github.com/soply/db_hand" is cloned to.
    """
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
    """
    Loads a specific estimator. Path to source should contain source code for
    the nsim algorithm ("github.com/soply/nsim_algorithm"), simple estimators ("github.com/soply/simple_estimation")
    and ELM "github.com/dclambert/Python-ELM". Other estimators
    can be added flexible.

    Parameters
    ------------
    estimator_kwargs : dict
        Contains estimator name, and additional arguments if necessary.

    path_to_source : string
        Folder where source code for other estimators is hosted.
    """
    name = estimator_kwargs['estimator']
    copy_dict = copy.deepcopy(estimator_kwargs)
    del copy_dict['estimator']
    if name == 'SIRKnn':
        sys.path.insert(0, path_to_source + '/simple_estimation/')
        from simple_estimation.estimators.SIRKnn import SIRKnn
        estim = SIRKnn(**copy_dict)
    elif name == 'SAVEKnn':
        sys.path.insert(0, path_to_source + '/simple_estimation/')
        from simple_estimation.estimators.SAVEKnn import SAVEKnn
        estim = SAVEKnn(**copy_dict)
    elif name == 'PHDKnn':
        sys.path.insert(0, path_to_source + '/simple_estimation/')
        from simple_estimation.estimators.PHDKnn import PHDKnn
        estim = PHDKnn(**copy_dict)
    elif name == 'nsim':
        sys.path.insert(0, path_to_source + '/nsim_algorithm/')
        from nsim_algorithm.estimator import NSIM_Estimator
        estim = NSIM_Estimator(**copy_dict)
    elif name == 'knn':
        from sklearn.neighnors import KNeighborsRegressor
        estim = KNeighborsRegressor()
    elif name == 'linreg':
        from sklearn.linear_model import LinearRegression
        estim = LinearRegression()
    elif name == 'isotron':
        sys.path.insert(0, path_to_source + '/simple_estimation/')
        from simple_estimation.estimators.Isotron import IsotronCV
        estim = IsotronCV(**copy_dict)
    elif name == 'slisotron':
        sys.path.insert(0, path_to_source + '/simple_estimation/')
        from simple_estimation.estimators.Slisotron import SlisotronCV
        estim = SlisotronCV(**copy_dict) # Super slow implementation
    elif name == 'elm':
        sys.path.insert(0, path_to_source + '/Python-ELM/')
        from elm import ELMRegressor
        estim = ELMRegressor(**copy_dict)
    elif name == 'ffnn':
        from simple_estimation.estimators.FeedForwardNetwork import FeedForwardNetwork
        estim = FeedForwardNetwork(**copy_dict)
    else:
        raise NotImplementedError('Load estimator: Estimator does not exist.')
    return estim
