#======================#
# =*= pso_utils.py =*= #
#======================#

### Setting this as in the paper

jitter_factor_array = [0.1, 0.33, 0.5, 0.67, 1]

'''

Function: multi_sd

Inputs:

	- row_Syn_num = the row number of the synthetic data
	- X_Syn = synthetic data that the adversary can leverage to build their predicate
	- X_Train = the original dataset, which we only use to check if the predicate constructed singled-out an individual

Purpose: 
	
	- Given a row number from the synthetic dataset, determine if the net-attack with the jitter_factor_array singles out an individual

Outputs: 

	- An array of length len(jitter_factor_array) with +1 in position i if the jitter factor at i isolated an individual, and -1 otherwise


'''

def multi_sd(row_Syn_num, X_Syn, X_Train):
    X_Syn_cols = list(X_Syn.columns)
    row_Syn = np.asarray(X_Syn.iloc[row_Syn_num])
    net_repo = [[] for i in jitter_factor_array]
    for v in range(len(jitter_factor_array)):
        jitter_factor = jitter_factor_array[v]
        predicate_array = []
        for j in X_Syn_cols:
            predicate_val = row_Syn[j]
            predicate_val_jitter = X_Syn[j].std() * jitter_factor
            predicate_array.append((X_Train[j] <= predicate_val + predicate_val_jitter) & (X_Train[j] >= predicate_val - predicate_val_jitter))
        # construct predicate from predicate array to determine number of individuals identified
        predicate = predicate_array[0]
        for j in range(1, len(predicate_array)):
            predicate = predicate & predicate_array[j]
        # compute the number of individuals identified
        num_net = np.sum(predicate)
        if num_net == 1:
            row_so = np.where(predicate == 1)[0][0]
            net_repo[v].append(row_so)
            print("Private row " + str(row_Syn_num) + " was able to single out a non-private row " + str(row_so) + " using Net " + str(v))
        else:
            print("Private row " + str(row_Syn_num) + " was unable to single out a non-private row using Net " + str(v) )
            net_repo[v].append(-1)
    return(net_repo)

'''

Function: all_rows_multi_sd

Inputs: 
	- X_Syn = synthetic data that the adversary can leverage to build their predicate
	- X_Train = the original dataset, which we only use to check if the predicate constructed singled-out an individual

Purpose: 

	- Apply multi_sd(row_Syn_num, X_Syn, X_Train) to all rows of the synthetic data X_Syn

Outputs: 

	- An array of length len(jitter_factor_array) with the proportion of individuals singled-out by the jitter factor in position i

'''

def all_rows_multi_sd(X_Syn, X_Train):
	all_repo = []
	for row_num in range(X_Syn.shape[0]):
		all_repo.append(multi_sd(row_num, X_Syn, X_Train))
	reshaped_result = [[all_repo[i][j][0] for i in range(len(all_repo))] for j in range(len(jitter_factor_array))]
	proportions_singled_out = [(1.0 * len(np.unique(list(filter(lambda x: x != -1, reshaped_result[i])))) / X_Train.shape[0]) for i in range(len(reshaped_result))]
	return(proportions_singled_out)




'''

Function: pso_nonpriv_run

Inputs:
	-df_input = nonprivatized dataset to test for singling-out risk

Purpose: 

	- Compute the PSO risk from the non-privatized data. Note that in this setting, setting X_Syn = X_Train in all_rows_multi_sd above does so.

Output: 

	- Dictionary with summary statistics about the singling-out risk

'''

### Note: The following code produces key names for the jitter_factor_array we hard_coded


def pso_nonpriv_run(df_input):
    jitter_outputs = all_rows_multi_sd(df_input, df_input)
    result_dict = {
    "sim_size" : ['Non-Private Baseline'],
    "B": ['Non-Private Baseline'],
    "epsilon_1": ['Non-Private Baseline'],
    "delta_1" : ['Non-Private Baseline'],
    "epsilon_2" : ['Non-Private Baseline'], 
    "delta_2" : ['Non-Private Baseline'],
    "k" : ['Non-Private Baseline'],
    "avg_priv_10_risk" : [jitter_outputs[0]],
    "sd_priv_10_risk" : [0],
    "avg_priv_33_risk" : [jitter_outputs[1]],
    "sd_priv_33_risk" : [0],
    "avg_priv_50_risk" : [jitter_outputs[2]],
    "sd_priv_50_risk" : [0],
    "avg_priv_67_risk" : [jitter_outputs[3]],
    "sd_priv_67_risk" : [0],
    "avg_priv_100_risk" : [jitter_outputs[4]],
    "sd_priv_100_risk" : [0]
    }
    return result_dict


'''

Function: sim_single_priv_run

Inputs:

	- df_input = nonprivate data that we wish to privatize, and then test for singling-out risk
	- epsilon_1, delta_1, epsilon_2, delta_2, k, B =  privacy parameters of priv_projections() function

Purpose: 
	
	- Determine the singling-out risk for the entire privatized data. Apply the multi_sd() function after we have privatized using priv_projections()

Output:

	- Dictionary with summary statistics about the singling-out risk

'''


def sim_single_priv_run(df_input, epsilon_1, delta_1, epsilon_2, delta_2, k, B):
    ## produce private data
    print("\n ==> Constructing Private Dataset using Private Projection Algorithm")
    X_priv = pd.DataFrame(priv_projections(df_input, epsilon_1, delta_1, epsilon_2, delta_2, k, B))
    ## run multisd attack with the jitter_factor_array 
    pso_multisd_results = all_rows_multi_sd(X_priv, df_input)
    return(pso_multisd_results)

'''

Function: sim_full_run

Inputs:
	- same as sim_single_priv_run, with one addition (below)
	- sim_size = number of times to regerenate a privatized dataset (Since the privatization approach uses randomness, 
				this can give a more accurate sense of the singling-out risk)

Purpose: 

	- Compute the PSO risk from the privatized data over multiple simulations. 

Output:

	- Dictionary with summary statistics about the singling-out risk

'''

### Note: The following code produces key names for the jitter_factor_array we hard_coded

def sim_full_run(df_input, sim_size, epsilon_1, delta_1, epsilon_2, delta_2, k, B):
    pso_full_catcher = [[] for i in range(len(jitter_factor_array))] 
    for sim in range(sim_size):
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("\n==> Running Simulation Number " + str(sim + 1) + " of " + str(sim_size))
        single_res = sim_single_priv_run(df_input, epsilon_1, delta_1, epsilon_2, delta_2, k, B)
        for v in range(len(single_res)):
            pso_full_catcher[v].append(single_res[v])
    result_dict = {
    "sim_size" : [sim_size],
    "B": [B],
    "epsilon_1": [epsilon_1],
    "delta_1" : [delta_1],
    "epsilon_2" : [epsilon_2], 
    "delta_2" : [delta_2],
    "k" : [k],
    "avg_priv_10_risk" : [np.mean(pso_full_catcher[0])],
    "sd_priv_10_risk" : [np.std(pso_full_catcher[0])],
    "avg_priv_33_risk" : [np.mean(pso_full_catcher[1])],
    "sd_priv_33_risk" : [np.std(pso_full_catcher[1])],
    "avg_priv_50_risk" : [np.mean(pso_full_catcher[2])],
    "sd_priv_50_risk" : [np.std(pso_full_catcher[2])],
    "avg_priv_67_risk" : [np.mean(pso_full_catcher[3])],
    "sd_priv_67_risk" : [np.std(pso_full_catcher[3])],
    "avg_priv_100_risk" : [np.mean(pso_full_catcher[4])],
    "sd_priv_100_risk" : [np.std(pso_full_catcher[4])]
    }
    return(pd.DataFrame(result_dict))
