#=============================#
# =*= prediction_utils.py =*= #
#=============================#

'''

Function: L2_normalizer

Inputs: 

    - X = numpy array
    
Purpose:

    - Rescale each row to have L2 norm at most 1

Outputs:
    
    -  A dataframe with each row rescaled to have L2 norm at most 1

'''

def L2_normalizer(X):
    max_norm_val = 0
    max_norm_index = 0
    for j in range(0, X.shape[0]):
        candidate_norm = np.linalg.norm(X[j], ord = 2)
        if candidate_norm > max_norm_val:
            max_norm_index = j
            max_norm_val = candidate_norm
    for j in range(0, X.shape[0]):
        X[j] = (1.0 * X[j]) / max_norm_val
    return(pd.DataFrame(X))


'''

Function: model_stats_generator

Inputs: 

    - y_actual = true targets
    - y_pred = predicted targets
    - targeting_threshold = threshold (between 0 and 1) used to do rank-based classication on the outputs (e.g., the botttom 100*(targeting_threshold) are classified as 1)
	
Purpose:

	- Compute the stats of the predictions

Outputs:
	
	- Dataframe of summary statistics 

'''


def model_stats_generator(y_actual, y_pred, targeting_threshold):
    repo = {
    'accuracy': [],
    'tpr': [], #P(C = 1 | Y = 1)
    'fpr': [], #P(C = 1 | Y = 0)
    'tnr': [],
    'fnr': [],
    'precision': []
    }
    # first, binarize for the actual vals
    threshold_val_actual = np.percentile(y_actual, targeting_threshold)
    actual_class = [y_actual[i] <= threshold_val_actual for i in range(len(y_actual))]
    # then, binarize for the predicted vals
    threshold_val_pred = np.percentile(y_pred, targeting_threshold)
    pred_class = [y_pred[i] <= threshold_val_pred for i in range(len(y_pred))]
    # next, get the stats
    actual = actual_class
    predicted = pred_class
    con_mat = confusion_matrix(actual_class, pred_class)
    # Write results to repo
    repo['accuracy'].append(1.0*(con_mat[0][0]+con_mat[1][1]) / np.sum(con_mat))
    repo['tpr'].append((1.0*con_mat[1][1]) / (con_mat[1][1] + con_mat[1][0])) #recall
    repo['fpr'].append((1.0*con_mat[0][1]) / (con_mat[0][1] + con_mat[0][0]))
    repo['tnr'].append((1.0*con_mat[0][0]) / (con_mat[0][1] + con_mat[0][0]))
    repo['fnr'].append((1.0*con_mat[1][0]) / (con_mat[1][1] + con_mat[1][0]) )
    repo['precision'].append((1.0*con_mat[1][1]) / (con_mat[1][1] + con_mat[0][1]))
    repo_df = pd.DataFrame(repo)
    return repo_df

'''

Function: process_nonprivate

Inputs: 

    - df_features = features used to build ML model (in K-Fold CV sense)
    - y = targets to build model (in K-Fold CV sense)
	
Purpose:

	- Compute the accuracy metrics of nonprivatized approach (as a baseline)

Outputs:
	
	- dataframe of summary statistics

'''


def process_nonprivate(df_features, y):
    sim_nonpriv_per_fold = []
    ## Run the experiment
    for fold in fold_index_repo:
        ## Now, pretend the researchers had the non-private top feature data; what would the model performance be?
        sys.stdout.write("\n***-> Running Non-Private Process on Fold %d" 
         % (fold))
        train_index = fold_index_repo[fold]['train']
        test_index = fold_index_repo[fold]['test']
        X_train, X_test = df_features.loc[train_index], df_features.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        ## build model
        reg_top = Ridge()
        reg_top.fit(X_train, y_train)
        y_pred = reg_top.predict(X_test)
        ## generate stats
        nonpriv_stats_per_fold = model_stats_generator(y_test.tolist(), y_pred, targeting_threshold)
        ## setup information for writing up results
        nonpriv_stats_per_fold['fold'] = fold
        nonpriv_stats_per_fold['B'] = 'Non-Private Baseline'
        nonpriv_stats_per_fold['epsilon_1'] = 'Non-Private Baseline'
        nonpriv_stats_per_fold['epsilon_2'] = 'Non-Private Baseline'
        nonpriv_stats_per_fold['delta_1'] = 'Non-Private Baseline'
        nonpriv_stats_per_fold['delta_2'] = 'Non-Private Baseline'
        sim_nonpriv_per_fold.append(nonpriv_stats_per_fold.values.flatten().tolist())
    sim_nonpriv_per_fold_df = pd.DataFrame(sim_nonpriv_per_fold, columns = col_names)
    sim_nonpriv = fold_performance_aggregation(sim_nonpriv_per_fold_df, agg_over_params, stat_cols)
    return sim_nonpriv




'''

Function: single_shot_process_private

Inputs: 

    - df_features = features used to build ML model (in K-Fold CV sense)
    - y = targets to build model (in K-Fold CV sense)
    - epsilon_1, delta_1, epsilon_2, delta_2, k, B = privacy parameters to priv_projections()
    
Purpose:

    - Compute the accuracy metrics of ML using privactized approach 

Outputs:
    
    - dataframe of summary statistics

'''



def single_shot_process_private(df_features, y, epsilon_1, delta_1, epsilon_2, delta_2, k, B):
    sim_priv = []
    ## Run the experiment
    ## Start by running the privatizer
    X_priv = pd.DataFrame(priv_projections(df_features, epsilon_1, delta_1, epsilon_2, delta_2, k, B))
    for fold in fold_index_repo:
        print("*-> Running Private Process on Fold %d" 
         % (fold) )
        train_index = fold_index_repo[fold]['train']
        test_index = fold_index_repo[fold]['test']
        X_train_priv, X_test_priv = X_priv.loc[train_index], X_priv.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        reg_top_priv = Ridge()
        reg_top_priv.fit(X_train_priv, y_train)
        y_priv = reg_top_priv.predict(X_test_priv)
        priv_stats = model_stats_generator(y_test.tolist(), y_priv, targeting_threshold)
        priv_stats['fold'] = fold
        priv_stats['B'] = B
        priv_stats['epsilon_1'] = epsilon_1
        priv_stats['epsilon_2'] = epsilon_2
        priv_stats['delta_1'] = delta_1
        priv_stats['delta_2'] = delta_2
        sim_priv.append(priv_stats.values.flatten().tolist())
    sim_priv_df = pd.DataFrame(sim_priv, columns = col_names)
    priv_df_single = fold_performance_aggregation(sim_priv_df, agg_over_params, stat_cols)
    return(priv_df_single)

'''

Function: process_private

Inputs: 

    - df_features = features used to build ML model (in K-Fold CV sense)
    - y = targets to build model (in K-Fold CV sense)
    - sim_size = number of times to replicate the privatization generation process
    - epsilon_1, delta_1, epsilon_2, delta_2, k, B = privacy parameters to priv_projections()
    
Purpose:

    - Compute the accuracy metrics of ML using privactized approach 

Outputs:
    
    - dataframe of summary statistics

'''

def process_private(df_features, y, sim_size, epsilon_1, delta_1, epsilon_2, delta_2, k, B):
    priv_df_repo = []
    for sim_num in range(sim_size):
        sys.stdout.write("\n***-> Running Simulation Number %d for %s" 
        % (sim_num + 1, sim_size))
        print("\n")
        priv_sim_res = single_shot_process_private(df_features, y, epsilon_1, delta_1, epsilon_2, delta_2, k, B)
        priv_sim_res['sim_num'] = sim_num + 1
        priv_df_repo.append(priv_sim_res)
    return(pd.concat(priv_df_repo))
