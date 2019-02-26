# Nonlinear single index model experiments
Experimental test suite for real data experiments conducted for paper https://arxiv.org/abs/1902.09024 

### Pointers to necessary code:

- NSIM Estimator: https://github.com/soply/nsim_algorithm
- Processed data sets: https://github.com/soply/db_hand
- Other estimators: 
   -https://github.com/soply/simple_estimation for (SIR, Isotron, and shallow nets) using code from 
      https://github.com/soply/sdr_toolbox for SIR and scikit-neuralnetwork for the shallow network.
   -https://github.com/soply/Python-ELM for the extreme learning machine algorithm 
   
   
### Structure:
../nsim_algorithm/
../local_sim_experiments/
../simple_estimation/
../db_hand/ (utils.py in this repo might adjusting the folder name for the data sets)
../Python-ELM/
../sdr_toolbox/

should all be in the same base folder.
