#========================================#
# =*= measure_attribute_inference.py =*= #
#========================================#


input_data_name = "replication_input.csv"

### Example parameters set below to loop over

B_params = [0.05, 0.25, 0.5, 1, 2]
epsilon_1 = 3
delta_1 = 0.00015
epsilon_2 = 0.9999
delta_2 = 0.00008
k = 10000
sim_size = 5


#------------------------------#
# -*- Load Python Packages -*- #
#------------------------------#

import os
import sys

import numpy as np
rseed = 0
np.random.seed(rseed)

from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from anonymeter.evaluators import InferenceEvaluator

#---------------------------#
# -*- Workspace Configs -*- #
#---------------------------#

print(">>> Setting folder configuration paths")

privacy_script_directory = os.getcwd()
utility_script_directory = '../utility_functions'
input_data_folder = '../../data'
metrics_results_folder = '../../outputs/privacy_metrics'

#------------------------------------------#
# -*- Loading Custom Utility Functions -*- #
#------------------------------------------#

os.chdir(utility_script_directory)

exec(open("private_projection_utils.py").read())
exec(open("prediction_utils.py").read())
exec(open("attribute_inference_utils.py").read())

#-------------------#
# -*- Load Data -*- #
#-------------------#

print('>>> Loading working datasets')
	
os.chdir(privacy_script_directory)
os.chdir(input_data_folder)
df = pd.read_csv(input_data_name)

#-------------------------#
# -*- Data Processing -*- #
#-------------------------#

print(">>> Processing Data ")

## Run L2 normalization on each row

features = df.copy()
features.drop(['targeting_variable'], axis = 1, inplace = True)
features = np.asarray(features)
features = 1.0 * features ## for type consistency with the L2_normalizer function()

features_l2 = L2_normalizer(features)

#-----------------------------------------------------------------#
# -*- Measure Attribute Inference Risk of Non-Privatized Data -*- #
#-----------------------------------------------------------------#

print(">>> Measuring Inference Risk of Non-Privatized Data")

nonprivate_repo = []

for attack_approach in ["but_one", "half", "one"]:
	print("----------------------------------------------")
	print("*-> Running Attack Appraoch " + attack_approach)
	nonpriv_res = nonpriv_simulation_consolidator(features_l2, sim_size, attack_approach)
	nonprivate_repo.append(nonpriv_res)

nonprivate_risks = pd.concat(nonprivate_repo)

#-------------------------------------------------------------#
# -*- Measure Attribute Inference Risk of Privatized Data -*- #
#-------------------------------------------------------------#

print(">>> Measuring Inference Risk of Privatized Data")

private_repo = []

for B in B_params:
	for attack_approach in ["all_but_one", "half", "one"]:
		print("----------------------------------------------")
		print("*-> Running Attack Appraoch " + attack_approach + " for B = " + str(B))
		priv_res = priv_simulation_consolidator(features_l2, sim_size, attack_approach, epsilon_1, delta_1, epsilon_2, delta_2, k, B)
		private_repo.append(priv_res)


private_risks = pd.concat(private_repo)


#-----------------------#
# -*- Write Results -*- #
#-----------------------#

os.chdir(privacy_script_directory)
os.chdir(metrics_results_folder)

print(">>> Writing Results to " + os.getcwd())

nonprivate_risks.to_csv('nonprivate_attribute_inference_metrics.csv', index = False)

privacy_param_string = "ep1_" + str(epsilon_1) + "_del1_" + str(delta_1) + "_ep2_" + str(epsilon_2) + "_del2_" + str(delta_2) + "_k_" + str(k)
private_risks.to_csv('private_attribute_inference_metrics_' + privacy_param_string + '.csv', index = False)

