#==========================================#
# =*= create_togo_replication_input.py =*= #
#==========================================#

#---------------------------------#
# -*- Load Necessary Packages -*- #
#---------------------------------#

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
rseed = 0
np.random.seed(rseed)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

#---------------------------#
# -*- Workspace Configs -*- #
#---------------------------#

utility_directory = os.getcwd()
input_data_folder = '../../data'

#---------------------------------------#
# -*- Create Replication Input Data -*- #
#---------------------------------------#

## For our example demo datasets, we mimic the shape of the Togo dataset from our paper, which has  10 features on 4,201 individuals
togo_data_size = (4201, 10)

## Create targeting variable of consumption from uniform [0,1)
targeting_variable = np.random.random(size = togo_data_size[0])

## Create features from uniform, tracking with the consumption variable, with 100 "distinct" and 5 "very distinct individuals" introduced
features = [pd.DataFrame(np.random.uniform(low = -0.5, high = 0.5, size = togo_data_size[1]) - targeting_variable[i]).T for i in range(len(targeting_variable) - 105)]

for i in range(100):
	point = targeting_variable[len(targeting_variable) - (i+1)]
	features.append(pd.DataFrame([point]*togo_data_size[1]).T)

features.append(pd.DataFrame([1.5]*togo_data_size[1]).T)
features.append(pd.DataFrame([-1.5]*togo_data_size[1]).T)
features.append(pd.DataFrame([0]*togo_data_size[1]).T)
features.append(pd.DataFrame([1.5]*int(togo_data_size[1]/2) + [-1.5]*int(togo_data_size[1]/2)).T) ## this only works when the number of features is even
features.append(pd.DataFrame([0.75]*int(togo_data_size[1]/2) + [-0.75]*int(togo_data_size[1]/2)).T) ## this only works when the number of features is even

features = pd.concat(features)



## Consolidate together to make data

togo_df = features.copy()
togo_df['targeting_variable'] = targeting_variable

#-----------------------#
# -*- Write to Repo -*- #
#-----------------------#

os.chdir(input_data_folder)

togo_df.to_csv("replication_input.csv", index = False)