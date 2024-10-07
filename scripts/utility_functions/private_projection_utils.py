#==========================#
# =*= privacy_utils.py =*= #
#==========================#

#---------------------------------#
# -*- Load necessary packages -*- #
#---------------------------------#

import numpy as np
rseed = 0
np.random.seed(rseed)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

#-------------------#
# -*- Functions -*- #
#-------------------#

'''

Function: covariance_gauss_noise

Inputs: 
	
	- scale_val = stanadard deviation of the gaussian
	- covariance_matrix_size = tuple representing the size of covariance matrix to be privatized [note that the values in the tuple should be the same]

Purpose:

	- constructs a symmetric matrix whose elements are sampled from the gaussian with standard deviation equal to scale_val

Outputs:
	
	- symmetric matrix of size covariance_matrix_size 

'''

def covariance_gauss_noise(scale_val, covariance_matrix_size):
    noise_matrix = np.random.normal(0, scale_val, covariance_matrix_size)
    num_rows, num_cols = covariance_matrix_size
    for i in range(num_rows):
        for j in range(i, num_cols):
            noise_matrix[j][i] = noise_matrix[i][j]
    return noise_matrix


'''

Function: priv_projections

Inputs: 
	
	- nonprivate_df = dataframe to be privatized (n rows and d columns)
	- epsilon_1 =  epsilon parameter for the privatization of the random projection
	- delta_1 = delta parameter of the privatization of the random projection
	- epsilon_2 = epsilon parameter of the covraiance privatization
	- delta_2 = delta parameter of the covariance privatization
	- k = dimensionality of the randomized projection
	- B = diameter of indistinguishability (note: when rows are normalized in the L2-unit ball, setting B = 2 yields classic differential privacy)

Purpose:

	- Compute the privatized version of nonprivate_df with (B, epsilon_1 + epsilon_2, delta_1 + delta_2)-targeted differential privacy

Outputs:
	
	- privatized numpy array with dataframe to be privatized (n rows and d columns)

'''

def priv_projections(nonprivate_df, epsilon_1, delta_1, epsilon_2, delta_2, k, B):
    # Setup
    X = nonprivate_df.to_numpy()
    n, d = nonprivate_df.shape
    R = np.random.randint(low = -1, high = 2, size = (d, k))
    # Computing Random Projection
    P = (1/k) * np.dot(X,R)
    # Privatizing Random Projection
    sigma_1_t1 = B / np.sqrt(k)
    sigma_1_t2 = np.sqrt(d * np.log((2 / 3) * (np.exp(1)-1) + 1) - ((1 / k)*np.log(delta_1 / 2)))
    sigma_1_t3 = (1  / epsilon_1) * np.sqrt(2 * (np.log(1 / delta_1) + epsilon_1))
    sigma_1 = sigma_1_t1 * sigma_1_t2 * sigma_1_t3
    P_private = P + np.random.normal(0, sigma_1, (n, k))
    # Computing Covariance Matrix
    X_cov = np.cov(X.T)
    # Computing Private Version of Right Singular Vectors
    sigma_2 = 2 * B * np.sqrt(2 * np.log(1.25 / delta_2)) / epsilon_2
    V = np.linalg.svd(X_cov + covariance_gauss_noise(sigma_2, (d,d)))[2] 
    V_keeps = V[:,:d] 
    # Building Privatized X
    X_private = np.dot(np.dot(P_private, np.linalg.pinv(np.dot(V_keeps.T, R))), V_keeps.T)
    return X_private

