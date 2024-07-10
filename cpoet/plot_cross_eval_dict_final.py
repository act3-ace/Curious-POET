import os, json, time, pickle
import subprocess, shutil, random
# from poet_distributed.es import initialize_master_fiber
from cpoet.poet_distributed.poet_algo import PopulationManager
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
from mpl_toolkits.axes_grid1 import make_axes_locatable



def get_run_names(args):
    run_names = []
    for r in args.run_names:
        run_names.append(r[0])
    return run_names





def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default=".")
    parser.add_argument("--run_names", type=str, required=False, nargs='+', default=[
        ('baseline_wCM_14nov_seed24582923', 'cp-2000'),
        ('baseline_wCM_17nov_seed24582924', '[cp-1500]->cp-2000'),
        ('icm_gamma0.0_wCM_17nov', 'cp-2000'),
        ('icm_gamma0.0_wCM_17feb', 'cp-2000'),
        ('icm_gamma0.0_wCM_21feb', 'cp-2000'),

        ('icm_gamma5.0_wCM_19nov', '[cp-1800]->cp-2000'),
        ('icm_gamma5.0_wCM_28nov', 'cp-2000'),
        ('icm_gamma5.0_wCM_29nov', 'cp-2000'),

        ('icm_gamma7.5_wCM_21nov', '[[cp-250]->cp-1300]->cp-2000'),
        ('icm_gamma7.5_wCM_2dec', 'cp-2000'),
        ('icm_gamma7.5_wCM_30nov_a', 'cp-2000'),
        ('icm_gamma7.5_wCM_21feb', 'cp-2000'),
        ('icm_gamma7.5_wCM_7mar', '[cp-1150]->cp-2000'),
        # ('icm_gamma7.5_wCM_7marA', '[cp-1200]->cp-2000'),  # weird ICM sampling histogram.  One rollout mid-run way oversampled.  also short stall from 1400-2100

        ('icm_gamma10.0_wCM_8nov', '[[cp-1300]->cp-1550]->cp-2000'),
        ('icm_gamma10.0_wCM_28nov',  'cp-2000'),
        ('icm_gamma10.0_wCM_30nov',  'cp-2000'),   
        # ('icm_gamma10.0_wCM_25jan',  'cp-2000'),   # this run diverged immediately and recovered at iteration 4k
        ('icm_gamma10.0_wCM_27jan',  'cp-2000'),
        ('icm_gamma10.0_wCM_8mar',  'cp-2000'),
        ('icm_gamma10.0_wCM_8marA',  'cp-2000'),

        ('icm_gamma15.0_wCM_28nov', 'cp-2000'),
        ('icm_gamma15.0_wCM_1dec', 'cp-2000'),
        ('icm_gamma15.0_wCM_2dec_a', 'cp-2000'),

        ('icm_gamma20.0_wCM_8nov', 'cp-2000'),
        ('icm_gamma20.0_start100_19nov', 'cp-2000'),
        ('icm_gamma20.0_wCM_2dec', 'cp-2000'),
        ])
    parser.add_argument("--cross_eval_data_filename", type=str, required=False, default='cross_eval_data_dict_extended.pkl')  #old is cross_eval_data.npy
    
    parser.add_argument("--output_figure_filename", type=str, required=False, default='cross_eval_confusion_dict_final.png')
    args=parser.parse_args()
    return args

def get_run_names(args):
    run_names = []
    for r in args.run_names:
        run_names.append(r[0])
    return run_names


def get_gamma_list(args):
    gamma_list = []
    for run, _ in args.run_names:
        if 'baseline' in run:
            gamma_list.append(str(0.0))
        else:
            s0 = run[run.find('gamma')+5: run.find('_wCM')]
            s1 = run[run.find('gamma')+5: run.find('_start')]
            if len(s0) < len(s1):
                gamma_list.append(s0)
            else:
                gamma_list.append(s1)
    return gamma_list

def get_agg_gammas(gammas):
    # return a list of unique gamma strings in ascending numerical order
    # given ["0.0", "0.0", "5.0", "7.5", "10.0", "10.0", "10.0", ]
    # return ["0.0", "5.0", "7.5", "10.0"]
    ret = []
    for g in gammas:
        if not g in ret:
            ret.append(g)
    return ret

def compute_cm_metric(eval_data:np.ndarray, agent_cm_threshold:float, env_cm_threshold:float, only_active=False, n_success=1) -> (float, float):
    # Given multi-seed evaluations of agent pop A on environment pop B and cm threshold, compute and return agent and env performance
    """
    For each population of agents, with $\zeta=A$, evaluated (10x) against the environments from another population with $\zeta=B$, 
    wE Can think about the eval data either from an agent perspective or an environment perspective.  
    * An agent winning, or solving an environment is dualed by an environment winning over an agent, or not-being-solved by an agent.
    * From an agent perspective: given a population of agents with $\zeta=A$, what fraction of environments with $\zeta=B$ 
        are bested *by any member* of the agent population?  
    * From the environment perspective: given a population of environments with $\zeta=A$, what fraction of agents with $\zeta=B$ 
        are bested *by any member* of the environment population?  
      
    * The output of the cross evaluation metric $C(popA, popB, T_{ce})$ is a scalar value in [0, 1] for both the agent and environment perspectives.
    
    eval_data: np.ndarray (agentpopsize, envpopsize, num_evals) typically (18,18,10)
    """
    BW_WIN_THRESHOLD = 230.0
    # ACTIVE_POP_SIZE2 = 5

    if only_active:
        eval_data = eval_data[:10, :10, :]

    agent_win_rates = np.sum(eval_data >= BW_WIN_THRESHOLD, axis=2) / eval_data.shape[2]
    agent_win_bool = agent_win_rates >= agent_cm_threshold
    agent_win_counts = np.sum(agent_win_bool, axis=0) # num agents that win each environment
    # agent_pop_performance = np.sum(agent_win_counts >= eval_data.shape[1]-ACTIVE_POP_SIZE2)/len(agent_win_counts)
    agent_pop_performance = np.sum(agent_win_counts >= n_success)/len(agent_win_counts)


    env_win_rates = np.sum(eval_data < BW_WIN_THRESHOLD, axis=2) / eval_data.shape[2]
    env_win_bool = env_win_rates >= env_cm_threshold
    env_win_counts = np.sum(env_win_bool, axis=1)
    # env_pop_performance = np.sum(env_win_counts >= eval_data.shape[0]-ACTIVE_POP_SIZE2)/len(env_win_counts)
    env_pop_performance = np.sum(env_win_counts >= n_success)/len(env_win_counts)

    return agent_pop_performance, env_pop_performance

def plot_confusion(ce_data, args):
    AGENT_CM_THRESHOLD = 0.9
    ENV_CM_THRESHOLD = 0.9
    fig, axs = plt.subplots(ncols=2, nrows=1)
    fig.set_size_inches(36, 18)
    
    size = len(args.run_names)
    agent_perf = np.zeros((size, size))
    env_perf = np.zeros((size, size))

    gammas = get_gamma_list(args)

    for agent_idx, agent in enumerate(get_run_names(args)):
        for env_idx, env in enumerate(get_run_names(args)):
            agent_perf[agent_idx, env_idx], env_perf[agent_idx, env_idx] = compute_cm_metric(ce_data[agent]['envs'][env], AGENT_CM_THRESHOLD, ENV_CM_THRESHOLD)

    im0=axs[0].imshow(agent_perf, cmap='plasma')
    im1=axs[1].imshow(env_perf, cmap='plasma')

    im0.set_clim([0,1])
    im1.set_clim([0,1])

    axs[0].set_xticks(np.arange(len(gammas)), labels=[str(int(float(g))) for g in gammas])
    axs[0].set_yticks(np.arange(len(gammas)), labels=[str(int(float(g))) for g in gammas])
    axs[1].set_xticks(np.arange(len(gammas)), labels=[str(int(float(g))) for g in gammas])
    axs[1].set_yticks(np.arange(len(gammas)), labels=[str(int(float(g))) for g in gammas])
    # axs[0].setp(axs[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    axs[0].set_ylabel(r'$\zeta_{agents}$')
    axs[0].set_xlabel(r'$\zeta_{environments}$')
    axs[1].set_ylabel(r'$\zeta_{agents}$')
    axs[1].set_xlabel(r'$\zeta_{environments}$')
    # Loop over data dimensions and create text annotations.
    for i in range(len(args.run_names)):
        for j in range(len(args.run_names)):
            text = axs[0].text(j, i, f"{agent_perf[i, j]:0.2f}\n", ha="center", va="center", color="k" if agent_perf[i, j]>0.3 else "y")
    
    axs[0].set_title(f"agent performance")
    axs[1].set_title(f"env performance")

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical', shrink=0.755)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical', shrink=0.755)  
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.output_figure_filename))
    print('CM plot created')

    # Now plot aggregated confusion where all evals for a given matrix location ($\zeta_{agent}$, $\zeta_{environment}$) are combined
    # and statistics are computed over them at once.
    fig, axs = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(8,6)
    gammas = get_gamma_list(args)
    agg_gammas = get_agg_gammas(gammas)

    size = len(agg_gammas)
    agg_means_agent = np.zeros((size, size))
    agg_stds_agent = np.zeros((size, size))


    for agg_agent_idx, agg_agent in enumerate(agg_gammas):  #raster over aggregated matrix
        for agg_env_idx, agg_env in enumerate(agg_gammas):
            accum_agent = []
            for agent_idx, agent in enumerate(get_run_names(args)):  # raster over whole matrix, accum_agentulating matching np arrays
                for env_idx, env in enumerate(get_run_names(args)):
                    if gammas[agent_idx] == agg_agent and gammas[env_idx] == agg_env:   #include these eval results
                        accum_agent.append(agent_perf[agent_idx, env_idx])
            # print('next')
            # for i in accum:
            #     print(i.mean(), i.std())
            agg_means_agent[agg_agent_idx, agg_env_idx] = np.mean(accum_agent)
            agg_stds_agent[agg_agent_idx, agg_env_idx] = np.std(accum_agent)

    im0 = axs.imshow(agg_means_agent, cmap='plasma')

    axs.set_xticks(np.arange(len(agg_gammas)), labels=agg_gammas)
    axs.set_yticks(np.arange(len(agg_gammas)), labels=agg_gammas)
    axs.set_ylabel(r'$\zeta_{agents}$')
    axs.set_xlabel(r'$\zeta_{environments}$')

    # Loop over data dimensions and create text annotations.
    for i in range(size):
        for j in range(size):
            axs.text(j, i, f"{agg_means_agent[i, j]:0.2f}\n"+ r"$\pm$" +f"{agg_stds_agent[i, j]:0.2f}",  ha="center", va="center",
                 color="y" if agg_means_agent[i, j] <0.80 else "k")

    # axs.set_title("Mean Env Scores confusion matrix (Agents vs Environments)")
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical', shrink=0.755)

    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.output_figure_filename[:-4]+f"_aggregated.png"))
    print('Aggregated CM plot created')


    # Now compute the agent means for table
    agent_means = []
    agent_stds = []
    for agg_agent_idx, agg_agent in enumerate(agg_gammas):  #raster over aggregated matrix
        accum_agent = []
        count = 0

        for agent_idx, agent in enumerate(get_run_names(args)):  # raster over whole matrix, accum_agentulating matching np arrays
            for env_idx, env in enumerate(get_run_names(args)):
                if gammas[agent_idx] == agg_agent:   #include these eval results
                    accum_agent.append(agent_perf[agent_idx, env_idx])
                    count += 1
            # print('next')
            # for i in accum:
            #     print(i.mean(), i.std())
        agent_means.append(np.mean(accum_agent))
        agent_stds.append(np.std(accum_agent))  
    print(f"means: {agent_means}")
    print(f"stds: {agent_stds}")  

    print(f"ratio of means:")
    for i in range(len(agent_means)):
        print(f"{agent_means[i]:0.2f} / {agent_means[0]:0.2f} = {agent_means[i]/agent_means[0]:0.2f} " + r"$\pm$" + f" {np.sqrt( agent_stds[0]**2 + agent_stds[i]**2 ):0.4f}")


    print('hi')

def main():
    """
    Given a list of run folders with annecsVsIterations.json files, plot them against one another for comparison. 

    """
    args = parse_args()

    if os.path.exists(os.path.join(args.output_dir, args.cross_eval_data_filename)):
        with open(os.path.join(args.output_dir, args.cross_eval_data_filename), 'rb') as f:
            ce_data = pickle.load(f)   

        plot_confusion(ce_data, args)


    else:
        print(f"no datafile found.")
    print('done')

        
if __name__ == "__main__":
    main()