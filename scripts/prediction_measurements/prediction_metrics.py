#====================================#
# =*= prediction_metrics_togo.py =*= #
#====================================#


input_data_name = "replication_input.csv"

### Example parameters set below to loop over

B_params = [0.05, 0.25, 0.5, 1, 2]
epsilon_1 = 3
delta_1 = 0.00015
epsilon_2 = 0.9999
delta_2 = 0.00008
k = 10000
sim_size = 5

targeting_threshold = 29
num_splits = 5

#------------------------------#
# -*- Load Python Packages -*- #
#------------------------------#

print(">>> Loading python packages")


import os
import sys

from sklearn.linear_model import Ridge

import numpy as np
rseed = 0
np.random.seed(rseed)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy.stats import rankdata


#---------------------------#
# -*- Workspace Configs -*- #
#---------------------------#

print(">>> Setting folder configuration paths")

prediction_script_directory = os.getcwd()
utility_script_directory = '../utility_functions'
input_data_folder = '../../data'
metrics_results_folder = '../../outputs/prediction_metrics'

#------------------------------------------#
# -*- Loading Custom Utility Functions -*- #
#------------------------------------------#

print(">>> Loading Custom Utility Functions")


os.chdir(utility_script_directory)
print(os.getcwd())

exec(open("prediction_consolidation_utils.py").read())
exec(open("prediction_utils.py").read())
exec(open("private_projection_utils.py").read())

#-------------------#
# -*- Load Data -*- #
#-------------------#

print('>>> Loading working datasets')
	
os.chdir(prediction_script_directory)
os.chdir(input_data_folder)
df = pd.read_csv(input_data_name)

#-------------------------#
# -*- Data Processing -*- #
#-------------------------#

print(">>> Processing Data ")

## Extract targeting variable 

y = df['targeting_variable']

## Run L2 normalization on each row

features = df.copy()
features.drop(['targeting_variable'], axis = 1, inplace = True)
features = np.asarray(features)
features = 1.0 * features ## for type consistency with the L2_normalizer function()

features_l2 = L2_normalizer(features)


#-------------------------#
# -*- K-Fold CV Setup -*- #
#-------------------------#

print(">>> Setting up K-Fold CV Variables")

## Get train-test indices for the folds
kf = KFold(n_splits = num_splits , shuffle = True, random_state = rseed)

fold_index_repo = dict((i,{'train': [], 'test': []}) for i in range(1, num_splits + 1,1))

i = 1
for train_index, test_index in kf.split(features_l2):
	fold_index_repo[i]['train'] = train_index
	fold_index_repo[i]['test'] = test_index
	i += 1

# removing unneeded values from the loop
del i
del train_index
del test_index

## Helpful Arrays to Organize Results in Latter Part of Script
stat_cols = ['accuracy', 'tpr', 'fpr', 'tnr', 'fnr', 'precision']

loop_params = ['fold', 'B', 'epsilon_1', 'epsilon_2', 'delta_1', 'delta_2'] 
col_names = stat_cols + loop_params

agg_over_params = ['B', 'epsilon_1', 'epsilon_2',  'delta_1', 'delta_2'] 


#-----------------------------------------#
# -*- Retrieve Non-Privatized Results -*- #
#-----------------------------------------#

print("------------------------------------------------")
print(">>> Running Iteration for Non-privatized Basline")

nonprivate_result = process_nonprivate(features_l2, y)

print("\n")

#-------------------------------------#
# -*- Retrieve Privatized Results -*- #
#-------------------------------------#

private_catcher = []

for B in B_params:
	print("----------------------------------------------------------------------")
	print(">>> Running Simulation Loop for Private Projections with B = " + str(B))
	private_result_per_sim_per_B = process_private(features_l2, y, sim_size, epsilon_1, delta_1, epsilon_2, delta_2, k, B)
	private_catcher.append(private_result_per_sim_per_B)

private_result_per_sim = pd.concat(private_catcher)

print("\n")

#-----------------------#
# -*- Write Results -*- #
#-----------------------#

os.chdir(prediction_script_directory)
os.chdir(metrics_results_folder)

print(">>> Writing Results to " + os.getcwd())

nonprivate_result.to_csv('nonprivate_prediction_metrics.csv', index = False)

privacy_param_string = "ep1_" + str(epsilon_1) + "_del1_" + str(delta_1) + "_ep2_" + str(epsilon_2) + "_del2_" + str(delta_2) + "_k_" + str(k)
private_result_per_sim.to_csv('private_prediction_metrics' + privacy_param_string + '.csv', index = False)
