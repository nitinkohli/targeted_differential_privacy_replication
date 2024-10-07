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

## Create features from uniform, tracking with the consumption variable
features = [pd.DataFrame(np.random.uniform(low = -1, high = 1, size = togo_data_size[1]) - targeting_variable[i]).T for i in range(len(targeting_variable))]

features = pd.concat(features)



## Consolidate together to make data

togo_df = features.copy()
togo_df['targeting_variable'] = targeting_variable

#-----------------------#
# -*- Write to Repo -*- #
#-----------------------#

os.chdir(input_data_folder)

togo_df.to_csv("replication_input.csv", index = False)