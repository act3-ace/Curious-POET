import os

# look through all the run folders and report number of not-yet-run coverage metrics remaining for each folder

Ns = [10000]#, 2000, 5000, 10000]
SCAN_ALL_RUNS = '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM'

if SCAN_ALL_RUNS:
    run_folders=[]
    for f in os.listdir(SCAN_ALL_RUNS):
        if os.path.isdir(os.path.join(SCAN_ALL_RUNS, f)) and '.vscode' not in f and 'data_collect' not in f:
            run_folders.append(os.path.join(SCAN_ALL_RUNS, f))
else:
    run_folders = [
        f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17nov",
        ]



def cm_results_present(folder, N):
    for f in os.listdir(folder):
        if f"coverage_metric_N{str(N)}" in f and f"_coverage" not in f: #folder name looks good
            #delete folder if empty
            if len(os.listdir(os.path.join(folder, f"coverage_metric_N{str(N)}"))) == 0:
                return False
            return True
    return False


for n_ind, N in enumerate(Ns):
    print(f"For N = {N}:")
    total = 0
    run_folders.sort()
    for run_folder in run_folders:
        checkpoints = []

        for fn in os.listdir(run_folder):
            if 'cp-' in fn and os.path.isdir(os.path.join(run_folder, fn)):
                # folder does not contain a coverage_metric_N... folder  (don't rerun cov metric)
                if not cm_results_present(os.path.join(run_folder, fn), N): 
                    checkpoints.append(fn)

        

        if len(checkpoints) > 0:
            print(f"\t{run_folder} has {len(checkpoints)} remaining CMs to run")
            
            total += len(checkpoints)
            # for checkpoint in checkpoints:
            #     print(f"\t{checkpoint}")
    print(f"\tTotal Remaining checkpoints to run CM: {total}")





