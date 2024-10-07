#===========================================#
# =*= prediction_consolidation_utils.py =*= #
#===========================================#

'''

'''



'''

'''

def fold_performance_aggregation(sim_df, agg_params, stat_perf):
    stat_results = sim_df.groupby(agg_params).agg(['mean'])[stat_perf].reset_index()
    stat_results_new_cols = []
    for thing in stat_results.columns:
        new_thing = thing[0]
        stat_results_new_cols.append(new_thing)
    stat_results.columns = stat_results_new_cols
    return stat_results

'''

'''

def simulation_performance_aggregation(sim_df, agg_params, stat_perf):
    stat_results = sim_df.groupby(agg_params).agg(['mean',np.std])[stat_perf].reset_index()
    stat_results_new_cols = []
    for thing in stat_results.columns:
        if thing[1] != '':
            new_thing = thing[0] + '_' + thing[1]
        else:
            new_thing = thing[0]
        stat_results_new_cols.append(new_thing)
    stat_results.columns = stat_results_new_cols
    return stat_results