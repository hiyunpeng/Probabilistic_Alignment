in_dir=out_dorigo_tuned\tuning
algos=['GWO', 'WOA', 'MFO', 'FA', 'BA', 'ALO', 'PSO_GBEST', 'ES_1P1']
budgets=[300, 500, 800, 1000]
dim=10
instances_per_problem=20
dev_frac=0.4 (holdout=60)
R_eval=10
target_tols={'easy': 0.1, 'med': 0.01, 'hard': 0.001}
progress=on
