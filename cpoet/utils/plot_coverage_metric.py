import os, json, time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import shutil
# for each run folder of POET checkpoints, plot the (single) CM value vs POET iteration
# also save the scores and iterations in the run folder in 'cm_vs_iter.json'

Ns = [10000]#, 2000, 5000, 10000]
SCAN_ALL_RUNS = '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM'
# MODE = 'only_last' # 'normal', 'only_last', integer (every nth env)

modes = [{'index':20}, {'index':40}, {'index':60}, {'index':80}, {'index':100}]
for MODE in modes:

    if MODE == 'only_last':     CM_FOLDER_NAME='fixedset_coverage_metric_onlyLast_wRollouts'
    elif isinstance(MODE, int): CM_FOLDER_NAME=f"fixedset_coverage_metric_every{MODE}th_wRollouts"
    elif isinstance(MODE, dict): CM_FOLDER_NAME=f"fixedset_coverage_metric_only_{MODE['index']}th_wRollouts"
    
    else:                       CM_FOLDER_NAME='fixedset_coverage_metric_N10000_wRollouts'

    # 
    if SCAN_ALL_RUNS:
        run_folders=[]
        for f in os.listdir(SCAN_ALL_RUNS):
            if os.path.isdir(os.path.join(SCAN_ALL_RUNS, f)) and f[:9] == 'icm_gamma':
                run_folders.append(os.path.join(SCAN_ALL_RUNS, f))
    else:
        run_folders = [
            f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_7mar",
            f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_7marA",
            ]



    def cm_results_present(folder, N):
        for f in os.listdir(folder):
            if CM_FOLDER_NAME in f: #folder name looks good
                #delete folder if empty
                if len(os.listdir(os.path.join(folder, CM_FOLDER_NAME))) < 2:
                    print(f" folder: {os.path.join(folder, CM_FOLDER_NAME)} is empty.  Deleting.")
                    shutil.rmtree(os.path.join(folder, CM_FOLDER_NAME))
                    return False
                return True
        return False

    def get_iter(folder):  # 'folder' has form 'cp-1350'
        return int(folder[folder.rfind('-')+1:])

    for n_ind, N in enumerate(Ns):
        for run_folder in run_folders:
            print(f"current run folder: {run_folder}")
            output_filename = os.path.join(run_folder, f"cm_plot_N{str(N)}.png")
            checkpoints = []

            for fn in os.listdir(run_folder):
                # if 'cp-' in fn and os.path.isdir(os.path.join(run_folder, fn)):
                if 'cp-' in fn and fn[-3:] in ['250', '500', '750', '000'] and os.path.isdir(os.path.join(run_folder, fn)):
                    # folder does not contain a coverage_metric_N... folder  (don't rerun cov metric)
                    if cm_results_present(os.path.join(run_folder, fn), N): 
                        checkpoints.append(fn)
                        

            iterations = []
            scores = []


            for checkpoint in checkpoints:
                # open summary.json and pull out score
                fn = os.path.join(run_folder, checkpoint, CM_FOLDER_NAME, "summary.json")
                with open(fn) as f:
                    summary = json.load(f)
                scores.append(summary['K100T230_mean'])
                iterations.append(get_iter(checkpoint))

            if len(checkpoints) == 0:
                print(f"continuing on run folder {run_folder}")
                continue
            _iterations = []
            _scores = []
            for i in np.argsort(iterations): # sort ascending by iteration number
                _iterations.append(iterations[i])
                _scores.append(scores[i])
            fig, axs = plt.subplots(1,1)
            axs.plot(_iterations, _scores, label=f"N={str(N)}, Range={max(scores) - min([a for a in scores if a > 0]):0.2f}")
            
            if isinstance(MODE, dict):   ## forgot to div by 100 when running coverage metric.  this hack saves re-running them all.
                _scores = [n*100 for n in _scores]

            print(f"N={str(N)}, max_score: {max(scores)}, min: {min([a for a in scores if a > 0])}, range: {max(scores) - min([a for a in scores if a > 0]):0.2f}")
            cm_vs_iter_fn = os.path.join(run_folder, "cm_vs_iter.json")
            with open(cm_vs_iter_fn, 'r') as f:
                try:
                    temp = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    temp = {}
            temp[CM_FOLDER_NAME]={'scores': _scores, 'iterations':_iterations}
            with open(cm_vs_iter_fn, 'w') as f:
                json.dump(temp, f) 

            axs.grid()
            axs.set_xlabel('POET Iterations')
            axs.set_ylabel('Coverage Metric Score')
            # plt.title(f"Coverage Metric Score vs Training Iteration \nN={str(N)}, K=100, T=230")
            axs.set_title(f"Coverage Metric Score vs Training Iteration \nK=100, T=230\n{run_folder[run_folder.rfind('/')+1:]}")

            axs.legend()
            fig.savefig(output_filename)

            del fig, axs 


