import os, json, time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
# combine the cm scores across sequences of 'only_n_th' CMs.

CHECKPOINT=2000
PLOT_SHIFT=1.5
gammas = [0.0, 10.0]
colors = ['b', 'g', 'r', 'c', 'm']
light_colors=['dodgerblue', 'lightgreen', 'lightcoral', 'lightcyan', 'violet']
depths = [20, 40, 60, 80, 100]
import matplotlib 
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.family': 'serif'})
fig, axs = plt.subplots(1,1)

folder = f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM"
config = os.path.join(folder, "combine_cm_plots_config.json")

with open(config) as f:
    config=json.load(f)

def get_gamma(run_folder: str):
    if 'baseline' in run_folder or 'gamma0.0_' in run_folder:
        return 0.0
    else:
        start = run_folder.rfind('gamma')+5
        end = run_folder.find('_', start)
        return float(run_folder[start:end])

for gamma_index, GAMMA in enumerate(gammas):

    # Load score vs iteration data from each population
    data = OrderedDict()
    for run_folder in config['run_folders']:
        if run_folder[0] == '_':
            continue
        fn=os.path.join(folder, run_folder, 'cm_vs_iter.json')
        if os.path.exists(fn):
            with open(fn)as f:
                iterscores = json.load(f)
            gamma = get_gamma(run_folder) 
            if not gamma in data.keys():
                data[gamma] = []
            data[gamma].append(iterscores)
        else:
            print(f"Unsuccessful loading cm vs iter from: {run_folder}")
        
    # now aggregate and display

    scores = np.empty((len(depths), len(data[GAMMA]))) * np.nan #  depth, scores-at-depth
    for run_idx, run in enumerate(data[GAMMA]):
        for d_index, d in enumerate(depths):
            cm_name = f"fixedset_coverage_metric_only_{d}th_wRollouts"
            score_idx = data[GAMMA][run_idx][cm_name]['iterations'].index(CHECKPOINT)
            scores[d_index, run_idx] = data[GAMMA][run_idx][cm_name]['scores'][score_idx]


    # now plot mean CM vs POET iterations


    means = np.nanmean(scores, axis=1)
    stds = scores.std(axis=1)

    for d_index, depth in enumerate(depths):  # plot actual points in grey
        axs.plot([depth + PLOT_SHIFT*gamma_index]*scores[d_index].shape[0], scores[d_index,], 'o', color=light_colors[gamma_index])
    axs.fill_between(depths, means - stds, means + stds, alpha=0.2, color=colors[gamma_index])
    axs.plot(depths, means, 'o-', label=r"$\zeta$="+f"{GAMMA}", color=colors[gamma_index])        

# axs.set_title(f"Coverage Metric Scores vs Free Evolution Depth\nat POET iteration {CHECKPOINT}, (points shifted for clarity)")
axs.set_xlabel('Evolution Depth')
axs.set_ylabel('Coverage Metric Score\n(shaded area = 1 std)')
axs.set_ylim((0.3, 0.9))
axs.legend(loc='upper right')
# axs.set_xlim((0, LAST_CHECKPOINT))
axs.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(folder, f"cm_vs_cmEvolutionDepth_iter{CHECKPOINT}.png"))


print('done')




