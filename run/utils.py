# coding: utf8
import sys
import copy
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
        Y = np.log(Y) # Using Logarithmic Data
        log_tf = True
    elif dataset == 'concrete':
        from handler_UCI_Concrete import read_all
        data = read_all(scaling = 'MeanVar')
        X, Y = data[:,:-1], data[:,-1]
        Y = np.log(Y) # Using Logarithmic Data
        log_tf = True
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
    else:
        raise NotImplementedError('Load data: Data set does not exist.')
    return X, Y, log_tf



def load_estimator(estimator_kwargs, path_to_source):
    sys.path.insert(0, path_to_source + '/simple_estimation/')
    sys.path.insert(0, path_to_source + '/nsim_estimator/')
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
    elif name == 'lsim':
        from nsim_estimator.estimators.tanPerLVSet import NSIMEst_TPLVSet
        estim = NSIMEst_TPLVSet(**copy_dict)
    elif name == 'knn':
        from sklearn.neighnors import KNeighborsRegressor
        estim = KNeighborsRegressor()
    else:
        raise NotImplementedError('Load estimator: Estimator set does not exist.')
    return estim
