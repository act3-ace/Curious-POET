import os, json, time
import subprocess, shutil, random
# from poet_distributed.es import initialize_master_fiber
from poet_distributed.poet_algo import PopulationManager
from argparse import ArgumentParser
import yaml
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})
"""
1.) figure out run folder, checkpoint, and args.json as needed
2.) load two population managers with pop a and pop b (here are two populations of agent/env pairs)
3.) cross eval agents and optimizers
4.) record/plot results
"""

def eval_agentsA_against_envB(pop_a, pop_b, args):
    """
    Parameters
    ----------
    pop_a, pop_b: OrderedDict of ESOptimizer objects defined in es.py
    args: Arguments Namespace object
    """
    count = 0
    optimizer = PopulationManager(args=args)
    # we'll be using the Population Manager optimizer as a vehicle for running eval
    eval_tasks = []
    for agent_idx, agent in enumerate(pop_a.values()):  #for each agent in a
        # (envs are stored with their paired agents so load agents to get their envs)
        for env_index, env in enumerate(pop_b.values()):  #for each env in b 
            for seed_index in range(args.num_evals):
                ephemoral_optimizer = optimizer.create_optimizer(
                    env.get_env_params(), 
                    seed=seed_index, 
                    nolog=True,
                    optim_id=str(count),
                    )
                count += 1
                task = ephemoral_optimizer.start_theta_eval(agent.theta)
                eval_tasks.append((task, agent_idx, env_index, seed_index))
    #eval
    results = np.zeros((len(pop_a), len(pop_b), args.num_evals))
    for task, agent_idx, env_index, seed_index in eval_tasks:
        stats = ephemoral_optimizer.get_theta_eval(task)
        results[agent_idx, env_index, seed_index] = stats.eval_returns_mean
    return results

def cross_eval(args):
    args_copy = deepcopy(args)
    num_workers=args.num_workers


    # 2.) reload the two pop managers (these hold all the env/agent pairs)
    # initialize_master_fiber()
    # random.seed(0)


    args_a_path = os.path.join(args.output_dir, args.run_a, args.chk_a, 'args.json')
    with open(args_a_path, 'r') as f:
            print("Loading config from " + args_a_path)
            config = yaml.safe_load(f)
    for k,v in config.items():
        vars(args)[k] = v
    args.num_workers = num_workers
    opt_a = PopulationManager(args=args)
    opt_a.reload(os.path.join(args.output_dir, args.run_a, args.chk_a))

    pop_a = OrderedDict()
    pop_a.update(opt_a.optimizers)
    # pop_a.update(opt_a.archived_optimizers)
    del opt_a

    args_b_path = os.path.join(args.output_dir, args_copy.run_b, args_copy.chk_b, 'args.json')
    with open(args_b_path, 'r') as f:
            print("Loading config from " + args_b_path)
            config = yaml.safe_load(f)
    for k,v in config.items():
        vars(args_copy)[k] = v
    args_copy.num_workers = num_workers
    opt_b = PopulationManager(args=args_copy)
    opt_b.reload(os.path.join(args.output_dir, args_copy.run_b, args.chk_b))

    pop_b = OrderedDict()
    pop_b.update(opt_b.optimizers)
    # pop_b.update(opt_b.archived_optimizers)
    del opt_b

    print(f"Loaded populations.   Beginning evaluation.")
    print(f"The evaluation does not include archived agents/envs")
    # print(f"Note: for both populations, using both active and archived populations.  Is this right?")

    # 3.) cross evaluate the two populations
    print(f"Evaluating Agents from {args.run_a} against Environments from {args.run_b}")
    agents_a_vs_env_b = eval_agentsA_against_envB(pop_a, pop_b, args)

    # print(f"Evaluating Agents from {args.run_b} against Environments from {args.run_a}")
    # agents_b_vs_env_a = eval_agentsA_against_envB(pop_b, pop_a, args)
    # 4.) plot and report
    print('Evals complete!')
    print(f"\nAgents from {args.run_a} against Envs from {args.run_b}: \n{agents_a_vs_env_b}\nMean: {agents_a_vs_env_b.mean():2.0f} StDev:{agents_a_vs_env_b.std():2.1f}")
    # print(f"\nAgents from {args.run_b} against Envs from {args.run_a}: \n{agents_b_vs_env_a}\nMean: {agents_b_vs_env_a.mean():2.0f} StDev:{agents_b_vs_env_a.std():2.1f}")

    return agents_a_vs_env_b

# if __name__ == '__main__':
#     # 1.) specify the populations to be cross evaluated
#     parser = ArgumentParser()
#     parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
#     parser.add_argument('--num_workers', type=int, default=50)
#     parser.add_argument('--num_evals', type=int, default=5)
#     parser.add_argument('--run_a', type=str, default='icm_gamma10.0_wCM_8nov')
#     parser.add_argument('--chk_a', type=str, default='cp-300')
#     parser.add_argument('--run_b', type=str, default='baseline_wCM_14nov_seed24582923')
#     parser.add_argument('--chk_b', type=str, default='cp-300')
#     args = parser.parse_args()
#     a,b = cross_eval(args)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
    parser.add_argument('--num_workers', type=int, default=70)
    parser.add_argument('--num_evals', type=int, default=10)
    parser.add_argument("--gamma_labels", type=str, required=False, nargs='+', default=[
            0,0,0, 5,5,5, 7.5,7.5,7.5, 10,10,10, 15,15,15
            # 0,0,0,0, 10,10,10,10
            # '0 (ckp: 2000)', '0 (ckp: 4000)', '0 (ckp: 6000)', '0 (ckp: 7750)', 
            # '10 (ckp: 2000)', '10 (ckp: 4000)', '10 (ckp: 6000)', '10 (ckp: 7750)',
        ])
    parser.add_argument('--plot_mean_factor', type=int, default=1)    
    parser.add_argument("--run_names", type=str, required=False, nargs='+', default=[

        # ('baseline_wCM_17nov_seed24582924', '[cp-1500]->cp-2000'),
        # ('baseline_wCM_17nov_seed24582924', '[[cp-1500]->cp-2050]->cp-4000'),
        # ('baseline_wCM_17nov_seed24582924', '[[[cp-1500]->cp-2050]->cp-5150]->cp-6000'),
        # ('baseline_wCM_17nov_seed24582924', '[[[cp-1500]->cp-2050]->cp-5150]->cp-7750'),

        # ('icm_gamma10.0_wCM_30nov',  'cp-2000'),
        # ('icm_gamma10.0_wCM_30nov',  '[[cp-2050]->cp-2350]->cp-4000'),
        # ('icm_gamma10.0_wCM_30nov',  '[[cp-2050]->cp-2350]->cp-6000'),
        # ('icm_gamma10.0_wCM_30nov',  '[[cp-2050]->cp-2350]->cp-7750'),



        ('baseline_wCM_14nov_seed24582923', 'cp-2000'),
        ('baseline_wCM_17nov_seed24582924', '[cp-1500]->cp-2000'),
        ('icm_gamma0.0_wCM_17nov', 'cp-2000'),

        ('icm_gamma5.0_wCM_19nov', '[cp-1800]->cp-2000'),
        ('icm_gamma5.0_wCM_28nov', 'cp-2000'),
        ('icm_gamma5.0_wCM_29nov', 'cp-2000'),

        ('icm_gamma7.5_wCM_21nov', '[[cp-250]->cp-1300]->cp-2000'),
        ('icm_gamma7.5_wCM_2dec', 'cp-2000'),
        ('icm_gamma7.5_wCM_30nov_a', 'cp-2000'),

        ('icm_gamma10.0_wCM_8nov', '[[cp-1300]->cp-1550]->cp-2000'),
        ('icm_gamma10.0_wCM_28nov',  'cp-2000'),
        ('icm_gamma10.0_wCM_30nov',  'cp-2000'),

        ('icm_gamma15.0_wCM_28nov', 'cp-2000'),
        ('icm_gamma15.0_wCM_1dec', 'cp-2000'),
        ('icm_gamma15.0_wCM_2dec_a', 'cp-2000'),

        # 'icm_gamma20.0_wCM_8nov', ''),
        # 'icm_gamma20.0_start100_19nov', ''),
        # 'icm_gamma20.0_wCM_2dec', ''),
        
        # 'icm_gamma25.0_wCM_28nov', ''),
        # 'icm_gamma25.0_wCM_4dec', ''),
        # 'icm_gamma25.0_wCM_4dec_a', ''),

        # 'icm_gamma30.0_wCM_16nov', ''),
        # 'icm_gamma30.0_wCM_4dec', ''),
        # 'icm_gamma30.0_wCM_4dec_a', ''),

        ])
    parser.add_argument("--cross_eval_data_filename", type=str, required=False, default='cross_eval_data.npy')
    
    parser.add_argument("--output_figure_filename", type=str, required=False, default='cross_eval_confusion_15.png')
    args=parser.parse_args()
    return args

def nan_mean(matrix):
    nan_count = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i,j]):
                matrix[i,j] = 0.0
                nan_count += 1
            
    return matrix.sum() / (matrix.size-nan_count)


def plot_confusion_condensed(cm_means, cm_stds, args):
    size = args.plot_mean_factor
    fig, axs = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(10, 10)
    condensed_gamma_list = args.gamma_labels[0::args.plot_mean_factor]
    temp = np.zeros((int(cm_means.shape[0]/args.plot_mean_factor), int(cm_means.shape[1]/args.plot_mean_factor)))
    for i in range(cm_means.shape[0]):
        for j in range(cm_means.shape[1]):
            if i == j:  
                cm_means[i,j] = np.nan  # mark the diagonals with nans so they're excluded
    for i in range(int(cm_means.shape[0]/args.plot_mean_factor)):
        for j in range(int(cm_means.shape[1]/args.plot_mean_factor)):
            temp[i, j] = nan_mean(cm_means[i*size: (i*size)+size, j*size: (j*size)+size])
    cm_means = temp

    plt.imshow(cm_means, cmap='plasma')

    axs.set_xticks(np.arange(len(condensed_gamma_list)), labels=condensed_gamma_list)
    axs.set_yticks(np.arange(len(condensed_gamma_list)), labels=condensed_gamma_list)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    axs.set_ylabel(r'Agents $\zeta$')
    axs.set_xlabel(r'Environments $\zeta$')

    # Loop over data dimensions and create text annotations.
    for i in range(len(condensed_gamma_list)):
        for j in range(len(condensed_gamma_list)):
            text = axs.text(j, i, int(cm_means[i, j]),
                        ha="center", va="center", color="r")
    
    # axs.set_title("Aggregated Mean Env Scores confusion matrix (Agents vs Environments)")
    plt.colorbar(shrink=0.76)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'condensed_'+args.output_figure_filename))
    print('plot created')


def plot_confusion(cm_means, cm_stds, args):
    if args.plot_mean_factor != 1:
        plot_confusion_condensed(cm_means, cm_stds, args)
        return
    fig, axs = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(10,10)
        
    plt.imshow(cm_means, cmap='plasma')

    axs.set_xticks(np.arange(len(args.run_names)), labels=args.gamma_labels)
    axs.set_yticks(np.arange(len(args.run_names)), labels=args.gamma_labels)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    axs.set_ylabel(r'Agents $\zeta$')
    axs.set_xlabel(r'Environments $\zeta$')

    # Loop over data dimensions and create text annotations.
    for i in range(len(args.run_names)):
        for j in range(len(args.run_names)):
            text = axs.text(j, i, int(cm_means[i, j]),
                        ha="center", va="center", color="r")
    
    # axs.set_title("Mean Env Scores confusion matrix (Agents vs Environments)")
    plt.colorbar(shrink=0.755)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.output_figure_filename))
    print('plot created')

def main():
    """
    Given a list of run folders with annecsVsIterations.json files, plot them against one another for comparison. 

    """
    args = parse_args()


    eval_results = {}
    run_count=0

    if os.path.exists(os.path.join(args.output_dir, args.cross_eval_data_filename)):
        temp =  np.load(os.path.join(args.output_dir, args.cross_eval_data_filename)) 
        cm_means = temp[:temp.shape[1],:]
        cm_stds  = temp[temp.shape[1]:,:]

        plot_confusion(cm_means, cm_stds, args)
    else:
        cm_means = np.zeros((len(args.run_names), len(args.run_names)))
        cm_stds = np.zeros((len(args.run_names), len(args.run_names)))
        for agent_idx in range(len(args.run_names)):
            for env_idx in range(len(args.run_names)):
                run_count += 1
                print(f"cross evaluating agents: {args.run_names[agent_idx]} against environments: {args.run_names[env_idx]}")
                args.run_a = args.run_names[agent_idx][0]
                args.chk_a = args.run_names[agent_idx][1]
                args.run_b = args.run_names[env_idx][0]
                args.chk_b = args.run_names[env_idx][1]
                agents_a_vs_env_b = cross_eval(args)   #np.zeros(())#
                cm_means[agent_idx, env_idx] = agents_a_vs_env_b.mean()
                cm_stds[agent_idx, env_idx] = agents_a_vs_env_b.std()

                eval_results['agents:' + args.run_names[agent_idx][0] + args.run_names[agent_idx][1] + 'vsEnv:' + args.run_names[env_idx][0] + args.run_names[env_idx][1]] = \
                    agents_a_vs_env_b
                print(f"Saving results for eval {run_count}")
                with open(os.path.join(args.output_dir, args.cross_eval_data_filename), 'wb') as f:
                    np.save(f, np.concatenate((cm_means, cm_stds), axis=0))

                plot_confusion(cm_means, args)

    print('done')

        
if __name__ == "__main__":
    main()