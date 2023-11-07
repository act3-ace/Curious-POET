import os, json, time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})

# combine the coverage metric plots for initial data collect: 3 baseline and 3 each for multiple gamma values
LAST_CHECKPOINT=2000
NUM_CHECKPOINTS_PER_POP=int(LAST_CHECKPOINT/250)
MODE = 20 # 'normal', 'only_last', integer (every nth env)
EVERY_VS_ONLY = 'every'  # select either 'every' or 'only_'  # select proper entry in cm_vs_iter.json files

if MODE == 'only_last':     CM_FOLDER_NAME='fixedset_coverage_metric_onlyLast_wRollouts'
elif isinstance(MODE, int): CM_FOLDER_NAME=f"fixedset_coverage_metric_{EVERY_VS_ONLY}{MODE}th_wRollouts"
else:                       CM_FOLDER_NAME='fixedset_coverage_metric_N10000_wRollouts'


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
    
# Load score vs iteration data from each population
data = OrderedDict()
for run_folder in config['run_folders']:
    if run_folder[0] == '_':
        continue
    fn=os.path.join(folder, run_folder, 'cm_vs_iter.json')
    if os.path.exists(fn):
        with open(fn)as f:
            iterscores = json.load(f)[CM_FOLDER_NAME]
        gamma = get_gamma(run_folder) 
        if not gamma in data.keys():
            data[gamma] = []
        data[gamma].append(iterscores)
    else:
        print(f"Unsuccessful loading cm vs iter from: {run_folder}")
       
# now aggregate and display

num_gamma_values = len(list(data.keys()))
max_pops_per_gamma = 0
for k, v in data.items():
    if len(v) > max_pops_per_gamma: max_pops_per_gamma = len(v)

# convert to dictionary of numpy arrays.  One per gamma value.
gammas = list(data.keys())
scores = {} #np.zeros((num_gamma_values, max_pops_per_gamma, int(2000/250)))  # shape=(10 gamma_values, 3 entries for each gamma, 40 scores)
for g in gammas:
    scores[g]=[]
#copy entries to numpy array (for iters < 2000)
for i, gamma in enumerate(data):
    scores[gamma]=np.empty((len(data[gamma]), NUM_CHECKPOINTS_PER_POP))*np.nan
    for j, run in enumerate(data[gamma]):
        for (it, score) in zip(run['iterations'], run['scores']):
            if it < LAST_CHECKPOINT+1:
                scores[gamma][j, int(it/250)-1] = score

#interpolate missing entries, (when entries are present on either side of a zero)
for gamma_value, _ in scores.items():
    for j in range(scores[gamma_value].shape[0]):
        for index, score in enumerate(scores[gamma_value][j]): 
            if index > 0 and index<scores[gamma_value].shape[1]-1 and np.isnan(score) and scores[gamma_value][j,index-1]>0.0 and scores[gamma_value][j,index+1]>0.0:
                interpolated_value = (scores[gamma_value][j,index-1] + scores[gamma_value][j,index+1]) / 2.0
                scores[gamma_value][j,index] = interpolated_value
                print(f"replacing missing entry at {gamma_value,j,index} with interpolated value: {interpolated_value}")



# now plot mean CM vs POET iterations
fig, axs = plt.subplots(1,1)
iterations = list(range(250,LAST_CHECKPOINT+50,250))
for gamma_value, _ in scores.items():
    # yerr = (scores[i].mean(axis=0)-scores[i].min(axis=0), scores[i].max(axis=0)+scores[i].mean(axis=0))
    # axs.errorbar(iterations, scores[i].mean(axis=0), yerr=yerr, label=f"Gamma={gammas[i]}")
    means = np.nanmean(scores[gamma_value], axis=0)
    stds = scores[gamma_value].std(axis=0)
    axs.fill_between(iterations, means - stds, means + stds, alpha=0.2)
    axs.plot(iterations, means, 'o-', label=f"Gamma={gamma_value}")
    
    #yerr=[scores[i].min(axis=0), scores[i].max(axis=0) ]
# if MODE == 'only_last':     axs.set_title('(Last K) Mean Coverage Metric Scores vs iterations')#\n (cm score range is shaded)')
# elif isinstance(MODE, int): axs.set_title(f"(Every {MODE}th CM Env) Mean Coverage Metric Scores vs iterations")#\n (cm score range is shaded)")
# else:                       axs.set_title('(N=10,000) Mean Coverage Metric Scores vs iterations')#\n (cm score range is shaded)')
axs.set_xlabel('POET Iterations')
# axs.set_xticks(range(0,2250,250))
axs.set_ylabel('Coverage Metric Score\n' +r'(shaded area = 1 $\sigma$)')
axs.set_ylim((0.3, 0.5))
axs.legend(loc='upper left')
# axs.set_xlim((0, LAST_CHECKPOINT))
axs.grid(True)
fig.tight_layout()
if MODE == 'only_last':     fig.savefig(os.path.join(folder, 'mean_cm_vs_iterations_Lastk.png'))
elif isinstance(MODE, int): fig.savefig(os.path.join(folder, f"mean_cm_vs_iterations_Every_{MODE}th.png"))
else:                       fig.savefig(os.path.join(folder, 'mean_cm_vs_iterations_N10k.png'))

# now plot mean CM vs POET iterations

fig, axs = plt.subplots(1,1)
iterations = list(range(250,LAST_CHECKPOINT+50,250))
plot_iterations = [500, 1000, 1500, 2000]#, 2500, 3000]


for plot_iteration in plot_iterations:
    idx = iterations.index(plot_iteration)
    scores_vs_gamma_at_iter =  [scores[g][:,idx] for g in gammas]
    means = [np.nanmean(x) for x in scores_vs_gamma_at_iter]
    stds = [np.nanstd(x) for x in scores_vs_gamma_at_iter]
    label=f"POET iter={plot_iteration}"
    axs.fill_between(gammas, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.1)
    axs.plot(gammas, means, 'o-', label=label)
    
# if MODE == 'only_last':     axs.set_title('(Last K) Mean Coverage Metric Scores vs gamma')#\n (cm score range is shaded)')
# elif isinstance(MODE, int): axs.set_title(f'(every {MODE}th Env) Mean Coverage Metric Scores vs gamma')#\n (cm score range is shaded)')
# else:                       axs.set_title('(N=10,000) Mean Coverage Metric Scores vs gamma')#\n (cm score range is shaded)')

axs.set_xlabel(r'$\zeta$')
axs.set_ylabel('Coverage Metric Score')
axs.set_ylim((0.2, 0.7))
axs.legend(reverse=True) #loc='lower right'
axs.grid(True)
fig.tight_layout()
if MODE == 'only_last':     fig.savefig(os.path.join(folder, 'mean_cm_vs_gamma_lastK.png'))
elif isinstance(MODE, int): fig.savefig(os.path.join(folder, f'mean_cm_vs_gamma_Every_{MODE}th.png'))
else:                       fig.savefig(os.path.join(folder, 'mean_cm_vs_gamma_N10K.png'))
print('done')



