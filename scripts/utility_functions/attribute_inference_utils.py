#================================#
# =*= attribute_inference.py =*= #
#================================#

control_size = 500 ## 500 is the default n_attacks in anonymeter

'''

Function: risk_computer

Inputs:

    - df_Train =  nonprivate data used to generate df_Syn
    - df_Syn = privatized data
    - df_Control = held out nonprivate data that was not used in the creation of df_Syn

Purpose: 

    - Using the anonymeter package, compute the attribute inference risk when the adversary knows all but one column

Outputs:

    - Dictionary of risk statistics

'''

def risk_computer(df_Train, df_Syn, df_Control):
    n_attacks = np.min([df_Train.shape[0], df_Control.shape[0], control_size]) 
    ## measuring risk
    columns = [df_Train.columns[i] for i in range(df_Train.shape[1])]
    inf_results = []
    print("\n==> Running risk measures...")
    for secret in columns:
        #print(secret)
        aux_cols = [col for col in columns if col != secret]
        inf_evaluator = InferenceEvaluator(ori = df_Train, syn = df_Syn, control = df_Control, aux_cols = aux_cols, secret = secret, n_attacks = n_attacks)
        inf_evaluator.evaluate(n_jobs=-2)
        inf_results.append((secret, inf_evaluator.results()))
    inf_risks = [res[1].risk().value for res in inf_results]
    print("\n Risk measures computed")
    risk_vals = {'median': np.median(inf_risks), 'mean': np.mean(inf_risks), 'max': np.max(inf_risks)}
    return(risk_vals)

'''

Function: risk_computer_half

Inputs:

    - df_Train =  nonprivate data used to generate df_Syn
    - df_Syn = privatized data
    - df_Control = held out nonprivate data that was not used in the creation of df_Syn

Purpose: 

    - Using the anonymeter package, compute the attribute inference risk when the adversary knows half of the columns

Outputs:

    - Dictionary of risk statistics

'''

def risk_computer_half(df_Train, df_Syn, df_Control):
    n_attacks = np.min([df_Train.shape[0], df_Control.shape[0], control_size ])
    ## measuring risk
    columns = [df_Train.columns[i] for i in range(df_Train.shape[1])]
    inf_results = []
    print("\n==> Running risk measures...")
    for secret in columns:
        #print(secret)
        aux_cols = [col for col in columns if col != secret]
        half_or_just_more = int((1.0*len(aux_cols))/2.0 + 0.5*(len(aux_cols)%2 == 1))
        half_aux_cols = np.random.choice(aux_cols, size = half_or_just_more, replace = False)
        inf_evaluator = InferenceEvaluator(ori = df_Train, syn = df_Syn, control = df_Control, aux_cols = half_aux_cols, secret = secret, n_attacks = n_attacks)
        inf_evaluator.evaluate(n_jobs=-2)
        inf_results.append((secret, inf_evaluator.results()))
    inf_risks = [res[1].risk().value for res in inf_results]
    print("\n Risk measures computed")
    risk_vals = {'median': np.median(inf_risks), 'mean': np.mean(inf_risks), 'max': np.max(inf_risks)}
    return(risk_vals)

'''

Function: risk_computer_one

Inputs:

    - df_Train =  nonprivate data used to generate df_Syn
    - df_Syn = privatized data
    - df_Control = held out nonprivate data that was not used in the creation of df_Syn

Purpose: 

    - Using the anonymeter package, compute the attribute inference risk when the adversary knows only one of the columns

Outputs:

    - Dictionary of risk statistics

'''

def risk_computer_one(df_Train, df_Syn, df_Control):
    n_attacks = np.min([df_Train.shape[0], df_Control.shape[0], control_size ])
    ## measuring risk
    columns = [df_Train.columns[i] for i in range(df_Train.shape[1])]
    inf_results = []
    print("\n==> Running risk measures...")
    for secret in columns:
        #print(secret)
        aux_cols = [col for col in columns if col != secret]
        one_aux_cols = np.random.choice(aux_cols, size = 1, replace = False)
        inf_evaluator = InferenceEvaluator(ori = df_Train, syn = df_Syn, control = df_Control, aux_cols = one_aux_cols, secret = secret, n_attacks = n_attacks)
        inf_evaluator.evaluate(n_jobs=-2)
        inf_results.append((secret, inf_evaluator.results()))
    inf_risks = [res[1].risk().value for res in inf_results]
    print("\n Risk measures computed")
    risk_vals = {'median': np.median(inf_risks), 'mean': np.mean(inf_risks), 'max': np.max(inf_risks)}
    return(risk_vals)

'''

Function: nonpriv_simulation_consolidator

Inputs:

    - df_input = nonprivate dataset
    - sim_size = number of simulated attacks to run (note: when attack_style = "all_but_one", no randomness is involved, so set to 1)
    - attack_style = string in ["all_but_one", "half", "one"]

Purpose: 

    - Using the anonymeter package, compute the attribute inference risk based on the attack_style

Outputs:

    - Dataframe of risk statistics

'''

def nonpriv_simulation_consolidator(df_input, sim_size, attack_style):
    sim_repo = {'median': [], 'mean': [], 'max': []}
    ### begin simulation
    for sim_num in range(sim_size):
        sys.stdout.write("\nRunning Simulation Number %d of %d" % (sim_num + 1, sim_size))
        ## for nonprivate testing, X_Syn is X_Train
        X_Train, X_Control = train_test_split(df_input, test_size = control_size)
        X_Syn = X_Train.copy()
        if attack_style == "all_but_one":
            sim_result = risk_computer(X_Train, X_Syn, X_Control)
        elif attack_style == "half":
            sim_result = risk_computer_half(X_Train, X_Syn, X_Control)
        else:
            sim_result = risk_computer_one(X_Train, X_Syn, X_Control)
        sim_repo['median'].append(sim_result['median'])
        sim_repo['mean'].append(sim_result['mean'])
        sim_repo['max'].append(sim_result['max'])
    result_dict = {
    "sim_size" : [sim_size],
    "attack_style": [attack_style],
    "B": ["Non-Private Baseline"],
    "epsilon_1": ["Non-Private Baseline"],
    "delta_1" : ["Non-Private Baseline"],
    "epsilon_2" : ["Non-Private Baseline"], 
    "delta_2" : ["Non-Private Baseline"],
    "k" : ["Non-Private Baseline"],
    "avg_median_risk" : [np.mean(sim_repo['median'])],
    "sd_median_risk" : [np.std(sim_repo['median'])],
    "avg_mean_risk" : [np.mean(sim_repo['mean'])],
    "sd_mean_risk" : [np.std(sim_repo['mean'])],
    "avg_max_risk" : [np.mean(sim_repo['max'])],
    "sd_max_risk" : [np.std(sim_repo['max'])]
    }
    return(pd.DataFrame(result_dict))

'''

Function: npriv_simulation_consolidator

Inputs:

    - df_input = nonprivate dataset
    - sim_size = number of simulated attacks to run 
    - attack_style = string in ["all_but_one", "half", "one"]
    - epsilon_1, delta_1, epsilon_2, delta_2, k, B = privacy parameters of the priv_projections()

Purpose: 

    - Using the anonymeter package, compute the attribute inference risk based on the attack_style

Outputs:

    - Dataframe of risk statistics

'''

def priv_simulation_consolidator(df_input, sim_size, attack_style, epsilon_1, delta_1, epsilon_2, delta_2, k, B):
    sim_repo = {'median': [], 'mean': [], 'max': []}
    for sim_num in range(sim_size):
        sys.stdout.write("\nRunning Simulation Number %d of %d" % (sim_num + 1, sim_size))
        ## Compute X_Syn, which is the result of our algorithm
        X_Train, X_Control = train_test_split(df_input, test_size = control_size)
        print("\n==> Computing private dataset...")
        X_Syn = pd.DataFrame(priv_projections(X_Train, epsilon_1, delta_1, epsilon_2, delta_2, k, B))
        print("\nPrivate dataset computed")
        ##
        if attack_style == "but_one":
            sim_result = risk_computer(X_Train, X_Syn, X_Control)
        elif attack_style == "half":
            sim_result = risk_computer_half(X_Train, X_Syn, X_Control)
        else:
            sim_result = risk_computer_one(X_Train, X_Syn, X_Control)
        sim_repo['median'].append(sim_result['median'])
        sim_repo['mean'].append(sim_result['mean'])
        sim_repo['max'].append(sim_result['max'])
    result_dict = {
    "sim_size" : [sim_size],
    "attack_style": [attack_style],
    "B": [B],
    "epsilon_1": [epsilon_1],
    "delta_1" : [delta_1],
    "epsilon_2" : [epsilon_2], 
    "delta_2" : [delta_2],
    "k" : [k],
    "avg_median_risk" : [np.mean(sim_repo['median'])],
    "sd_median_risk" : [np.std(sim_repo['median'])],
    "avg_mean_risk" : [np.mean(sim_repo['mean'])],
    "sd_mean_risk" : [np.std(sim_repo['mean'])],
    "avg_max_risk" : [np.mean(sim_repo['max'])],
    "sd_max_risk" : [np.std(sim_repo['max'])]
    }
    return(pd.DataFrame(result_dict))
