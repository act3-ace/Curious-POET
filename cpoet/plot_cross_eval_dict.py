import os, json, time, pickle
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
from mpl_toolkits.axes_grid1 import make_axes_locatable



def get_run_names(args):
    run_names = []
    for r in args.run_names:
        run_names.append(r[0])
    return run_names





def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
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


        ])
    parser.add_argument("--cross_eval_data_filename", type=str, required=False, default='cross_eval_data_dict.pkl')  #old is cross_eval_data.npy
    
    parser.add_argument("--output_figure_filename", type=str, required=False, default='cross_eval_confusion_dict.png')
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
            gamma_list.append(run[run.find('gamma')+5: run.find('_wCM')])
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

def plot_confusion(ce_data, args):

    fig, axs = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(20,20)
    
    size = len(args.run_names)
    cm_means = np.zeros((size, size))
    cm_stds = np.zeros((size, size))

    gammas = get_gamma_list(args)

    for agent_idx, agent in enumerate(get_run_names(args)):
        for env_idx, env in enumerate(get_run_names(args)):
            cm_means[agent_idx, env_idx] = np.mean(ce_data[agent]['envs'][env])
            cm_stds[agent_idx, env_idx] = np.std(ce_data[agent]['envs'][env])

    plt.imshow(cm_means, cmap='plasma')

    axs.set_xticks(np.arange(len(gammas)), labels=gammas)
    axs.set_yticks(np.arange(len(gammas)), labels=gammas)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    axs.set_ylabel(r'$\zeta_{agents}$')
    axs.set_xlabel(r'$\zeta_{environments}$')

    # Loop over data dimensions and create text annotations.
    for i in range(len(args.run_names)):
        for j in range(len(args.run_names)):
            text = axs.text(j, i, f"{int(cm_means[i, j])}\n" + r"$\pm$" + f"{int(cm_stds[i, j])}",
                        ha="center", va="center", color="r")
    
    # axs.set_title("Mean Env Scores confusion matrix (Agents vs Environments)")
    plt.colorbar(shrink=0.755)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.output_figure_filename))
    print('CM plot created')

    # Now plot aggregated confusion where all evals for a given matrix location ($\zeta_{agent}$, $\zeta_{environment}$) are combined
    # and statistics are computed over them at once.
    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.set_size_inches(6,8)
    gammas = get_gamma_list(args)
    agg_gammas = get_agg_gammas(gammas)

    size = len(agg_gammas)
    agg_means = np.zeros((size, size))
    agg_stds = np.zeros((size, size))

    for agg_agent_idx, agg_agent in enumerate(agg_gammas):  #raster over aggregated matrix
        for agg_env_idx, agg_env in enumerate(agg_gammas):
            accum = []
            for agent_idx, agent in enumerate(get_run_names(args)):  # raster over whole matrix, accumulating matching np arrays
                for env_idx, env in enumerate(get_run_names(args)):
                    if gammas[agent_idx] == agg_agent and gammas[env_idx] == agg_env:   #include these eval results
                        if not type(ce_data[agent]['envs'][env]) == int:
                            accum.append(ce_data[agent]['envs'][env].flatten())
            # print('next')
            # for i in accum:
            #     print(i.mean(), i.std())
            agg_means[agg_agent_idx, agg_env_idx] = np.concatenate(accum).mean()
            agg_stds[agg_agent_idx, agg_env_idx] = np.concatenate(accum).std()

    im0 = axs[0].imshow(agg_means, cmap='plasma')
    im1 = axs[1].imshow(agg_stds, cmap='plasma')

    axs[0].set_xticks(np.arange(len(agg_gammas)), labels=agg_gammas)
    axs[1].set_xticks(np.arange(len(agg_gammas)), labels=agg_gammas)
    axs[0].set_yticks(np.arange(len(agg_gammas)), labels=agg_gammas)
    axs[1].set_yticks(np.arange(len(agg_gammas)), labels=agg_gammas)
    # plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    axs[0].set_ylabel(r'$\zeta_{agents}$')
    axs[1].set_ylabel(r'$\zeta_{agents}$')
    # axs[0].set_xlabel(r'$\zeta_{environments}$')
    axs[0].xaxis.set_tick_params(labelbottom=False); axs[0].set_xticks([])
    axs[1].set_xlabel(r'$\zeta_{environments}$')

    # Loop over data dimensions and create text annotations.
    for i in range(size):
        for j in range(size):
            axs[0].text(j, i, f"{int(agg_means[i, j])}",  ha="center", va="center",
                 color="y" if int(agg_means[i, j]) <200 else "k")
            axs[1].text(j, i, f"{int(agg_stds[i, j])}", ha="center", va="center", 
                color="y" if int(agg_stds[i, j]) <100 else "k")

    # axs.set_title("Mean Env Scores confusion matrix (Agents vs Environments)")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical', shrink=0.755)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical', shrink=0.755)    
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.output_figure_filename[:-4]+f"_aggregated.png"))
    print('Aggregated CM plot created')


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