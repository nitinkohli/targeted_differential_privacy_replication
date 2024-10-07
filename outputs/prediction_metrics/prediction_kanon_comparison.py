#========================================#
# =*= prediction_kanon_comparison.py =*= #
#========================================#

input_data_name = "replication_input.csv"

targeting_threshold = 29


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

##IMPORT MONDRIAN

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
exec(open("kanon_utils.py").read())


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

loop_params = ['k', 'fold'] 
col_names = stat_cols + loop_params

agg_over_params =['k']


#--------------------------------------------#
# -*- k-Anonymity via Mondrian Algorithm -*- #
#--------------------------------------------#

print(">>> Gathering k-anonymity Prediction Performanc Metrics")

k_anon_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200]

k_anon_results = []

for k in k_anon_vals:
	print("----------------------------------------")
	print("Running k-anon process for k = " + str(k))
	k_anon_vals.append(process_private_kanon(features_l2, k))


k_anon_results_df = pd.DataFrame(k_anon_results)

#-----------------------#
# -*- Write Results -*- #
#-----------------------#

os.chdir(prediction_script_directory)
os.chdir(metrics_results_folder)

print(">>> Writing Results to " + os.getcwd())

k_anon_results_df.to_csv('kanon_prediction_metrics.csv', index = False)

