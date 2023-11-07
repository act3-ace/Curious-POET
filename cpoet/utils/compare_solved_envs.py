import numpy as np
import os, pickle, json
import argparse
import matplotlib.pyplot as plt
FONTSIZE = 16

def get_exact_checkpoint_name(folder, checkpoint):
    cp_string = f"cp-{str(checkpoint)}"

    for f in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, f)):
            if f[-len(cp_string):] == cp_string:  # ends in exact cp string
                return f
    return None

def remove_zeroes_case(cov_met_name, databrick):
    if 'only' in cov_met_name:
        return databrick[:100,:]
    else:    
        return databrick

def tabulate(baseline_solves, experiment_solves):
    baseline_pop_solves = np.any(baseline_solves, axis=1)
    baseline_rate = sum(baseline_pop_solves)/len(baseline_pop_solves)

    experiment_pop_solves = np.any(experiment_solves, axis=1)
    experiment_rate = sum(experiment_pop_solves)/len(experiment_pop_solves)

    stacked = np.stack((baseline_pop_solves, experiment_pop_solves))
    counts = np.sum(stacked, axis=0)

    both = counts==2
    neither = counts==0

    both_rate = sum(both)/len(both)
    neither_rate = sum(neither/len(neither))

    # what fraction of those envs solved by the baseline population were also solved by the experiment population?
    both_to_baseline_ratio = sum(both) / sum(baseline_pop_solves)
    return [baseline_rate, experiment_rate, both_rate]#, both_to_baseline_ratio]

def extract_label(s):
    if 'only in s':
        return f"{s[s.find('only_')+5:s.find('th')]} Steps"
    else:
        return s

def main(config):
    with open(config, 'r') as f:
        config = json.load(f)

    # remove commented out runs in json (# is comment character)
    for gamma in config['runs'].keys():
        temp_runs = []
        for run in config['runs'][gamma]:
            if run[0] != '#':
                temp_runs.append(run)
        config['runs'][gamma] = temp_runs

    output_folder = os.path.join(config['base_folder'], config['output_folder'])    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    gammas = list(config['runs'].keys())
    
    import matplotlib 
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams.update({'font.family': 'serif'})
    fig, axs = plt.subplots()
    for plot_count, cov_met_name in enumerate(config['cov_met_names']):
        label = extract_label(cov_met_name)
        # load databricks and write into simple 2d numpy arrays
        scores = {}
        solves = {}
        for gamma in gammas:
            scores[gamma] = []
            solves[gamma] = []
            for run_name in config['runs'][gamma]:
                cp_folder_name = get_exact_checkpoint_name(os.path.join(config['base_folder'], run_name), config['checkpoint'])
                databrick_fn = os.path.join(config['base_folder'], run_name, cp_folder_name, cov_met_name, 'kjnm_databrick.npy') 
                databrick = np.load(databrick_fn)
                assert databrick.shape[0:2] == (1,1)
                databrick = np.squeeze(databrick)
                databrick = remove_zeroes_case(cov_met_name, databrick)
                scores[gamma].append(databrick)
                solves[gamma].append(databrick > config['win_threshold'])


        # compute and plot bars and both/baseline ratios
        
        
        tabs = []
        for baseline in solves['0.0']:
            for experiment in solves['10.0']:
                tabs.append(tabulate(baseline, experiment))
        means = np.mean(np.array(tabs), axis=0)
        stds = np.std(np.array(tabs), axis=0)
        # axs.bar(x=[x+0.1*plot_count for x in list(range(means.shape[0]))], height=means, width=0.4, yerr=stds, label=label, capsize=15)
        axs.bar(x=[x+0.1*plot_count for x in list(range(means.shape[0]))], height=means, width=0.1, yerr=stds, label=label, capsize=3)

    # axs.set_title(f"CM Environment Solution Rate vs CM Free Evolution Depth\n (100 environments at indicated evolution depth)")
    # axs.set_title(f"CM Environment Solution Rate\n (10,000 environments w/ reset every 100)\n POET iteration {config['checkpoint']}")
    axs.set_xticks([x+0.0 for x in list(range(means.shape[0]))], ['Baseline POET\n'+r'$\zeta=0.0$'+'\nSolved', 'Curious POET\n'+r'$\zeta=10.0$'+'\nSolved', 'Both\nSolved'])
    axs.set_ylabel(f"Fraction of Coverage Metric \nEnvironments Solved")#.set_fontsize(FONTSIZE)
    # axs.set_ylim([0.0, 1.0])
    axs.grid(True)
    # axs.legend(loc='lower center')
    fig.tight_layout()
    if len(config['cov_met_names']) > 1:
        plot_fn = os.path.join(output_folder, f"env_solve_comparison_cp{config['checkpoint']}_multi.png")
    else:
        plot_fn = os.path.join(output_folder, f"env_solve_comparison_cp{config['checkpoint']}.png")
    fig.savefig(plot_fn)
    return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/compare_cm_config.json")
    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    args = parse_args()
    main(
        config=args.config,

    )