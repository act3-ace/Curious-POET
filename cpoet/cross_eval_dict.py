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
    results = np.zeros((len(pop_a), len(pop_b), args.num_evals))
    if args.fake:
        return results
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
    
    for task, agent_idx, env_index, seed_index in eval_tasks:
        stats = ephemoral_optimizer.get_theta_eval(task)
        results[agent_idx, env_index, seed_index] = stats.eval_returns_mean
    return results

def cross_eval(args):
    niche_params = {
        "use_icm": True,
        "icm_stable_iteration": 200,
        "icm_training_gamma": 0.0
    }

    args_copy = deepcopy(args)
    num_workers=args.num_workers


    # 2.) reload the two pop managers (these hold all the env/agent pairs)
    # initialize_master_fiber()
    # random.seed(0)


    args_a_path = os.path.join(args.output_dir, args.run_a, args.chk_a, 'args.json')
    with open(args_a_path, 'r') as f:
            print("Loading config from " + args_a_path)
            config = yaml.safe_load(f)
    config['niche'] = 'ICM_ES_Bipedal'
    if not 'niche_params' in config:
        config['niche_params'] = niche_params

    for k,v in config.items():
        vars(args)[k] = v
    args.num_workers = num_workers
    opt_a = PopulationManager(args=args)
    opt_a.reload(os.path.join(args.output_dir, args.run_a, args.chk_a))

    pop_a = OrderedDict()
    pop_a.update(opt_a.optimizers)
    pop_a.update(opt_a.archived_optimizers)
    del opt_a

    args_b_path = os.path.join(args.output_dir, args_copy.run_b, args_copy.chk_b, 'args.json')
    with open(args_b_path, 'r') as f:
            print("Loading config from " + args_b_path)
            config = yaml.safe_load(f)
    config['niche'] = 'ICM_ES_Bipedal'
    if not 'niche_params' in config:
        config['niche_params'] = niche_params

    for k,v in config.items():
        vars(args_copy)[k] = v
    args_copy.num_workers = num_workers
    opt_b = PopulationManager(args=args_copy)
    opt_b.reload(os.path.join(args.output_dir, args_copy.run_b, args.chk_b))

    pop_b = OrderedDict()
    pop_b.update(opt_b.optimizers)
    pop_b.update(opt_b.archived_optimizers)
    del opt_b

    print(f"Loaded populations.   Beginning evaluation.")
    # print(f"The evaluation does not include archived agents/envs")
    print(f"Note: for both populations, using both active and archived populations.")

    # 3.) cross evaluate the two populations
    print(f"Evaluating Agents from {args.run_a} against Environments from {args.run_b}")
    agents_a_vs_env_b = eval_agentsA_against_envB(pop_a, pop_b, args)

    # print(f"Evaluating Agents from {args.run_b} against Environments from {args.run_a}")
    # agents_b_vs_env_a = eval_agentsA_against_envB(pop_b, pop_a, args)
    # 4.) plot and report
    print('Evals complete!')
    # print(f"\nAgents from {args.run_a} against Envs from {args.run_b}: \n{agents_a_vs_env_b}\nMean: {agents_a_vs_env_b.mean():2.0f} StDev:{agents_a_vs_env_b.std():2.1f}")
    # print(f"\nAgents from {args.run_b} against Envs from {args.run_a}: \n{agents_b_vs_env_a}\nMean: {agents_b_vs_env_a.mean():2.0f} StDev:{agents_b_vs_env_a.std():2.1f}")

    return agents_a_vs_env_b

def get_run_names(args):
    run_names = []
    for r in args.run_names:
        run_names.append(r[0])
    return run_names

def check_ce_data(ce_data, args):  # if a new run is added to args, then add it to the other ce_data object
    run_names = get_run_names(args)
    for r in run_names:
        if r not in ce_data:
            for q in list(ce_data.keys()):  # add new entrant to all the other existing entries
                ce_data[q]['envs'][r] = -1
            ce_data[r]={'envs':{}}          # create a new entry for the new entrant
            for q in list(ce_data.keys()): # populate new entry with all runs
                ce_data[r]['envs'][q] = -1
    return ce_data

def init_cd_dict(args):
    # run_names = get_run_names(args)
    ce_data = {}
    for run_name, cp in args.run_names:
        ce_data[run_name]={'envs':{}}
        # ce_data[run_name]['name']=run_name
        # ce_data[run_name]['cp']=cp
        for e, _ in args.run_names:
            ce_data[run_name]['envs'][e] = -1
    return ce_data

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
    parser.add_argument('--num_workers', type=int, default=85)
    parser.add_argument('--num_evals', type=int, default=10)
    parser.add_argument('--fake', type=int, default=0)
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
        ('icm_gamma7.5_wCM_7marA', '[cp-1200]->cp-2000'),

        ('icm_gamma10.0_wCM_8nov', '[[cp-1300]->cp-1550]->cp-2000'),
        ('icm_gamma10.0_wCM_28nov',  'cp-2000'),
        ('icm_gamma10.0_wCM_30nov',  'cp-2000'),   # diverged and recovered!!
        ('icm_gamma10.0_wCM_25jan',  'cp-2000'),   # first 4k diverged !
        ('icm_gamma10.0_wCM_27jan',  'cp-2000'),
        ('icm_gamma10.0_wCM_8mar',  'cp-2000'),
        ('icm_gamma10.0_wCM_8marA',  'cp-2000'),

        ('icm_gamma15.0_wCM_28nov', 'cp-2000'),
        ('icm_gamma15.0_wCM_1dec', 'cp-2000'),
        ('icm_gamma15.0_wCM_2dec_a', 'cp-2000'),

        ('icm_gamma20.0_wCM_8nov', 'cp-2000'),
        ('icm_gamma20.0_start100_19nov', 'cp-2000'),
        ('icm_gamma20.0_wCM_2dec', 'cp-2000'),
        
        # 'icm_gamma25.0_wCM_28nov', ''),
        # 'icm_gamma25.0_wCM_4dec', ''),
        # 'icm_gamma25.0_wCM_4dec_a', ''),

        # 'icm_gamma30.0_wCM_16nov', ''),
        # 'icm_gamma30.0_wCM_4dec', ''),
        # 'icm_gamma30.0_wCM_4dec_a', ''),

        ])
    parser.add_argument("--cross_eval_data_filename", type=str, required=False, default='cross_eval_data_dict_extended.pkl')  #old is cross_eval_data.npy
    
    args=parser.parse_args()
    return args



def main():
    """
    Given a list of run folders with annecsVsIterations.json files, plot them against one another for comparison. 

    """
    args = parse_args()
    run_count=0

    if os.path.exists(os.path.join(args.output_dir, args.cross_eval_data_filename)):
        with open(os.path.join(args.output_dir, args.cross_eval_data_filename), 'rb') as f:
            ce_data = pickle.load(f)   
    else:
        ce_data=init_cd_dict(args)
        with open(os.path.join(args.output_dir, args.cross_eval_data_filename), 'wb') as f:
            pickle.dump(ce_data, f)

    ce_data = check_ce_data(ce_data, args)

    for agents_run_name, agents_cp in args.run_names:
        for envs_run_name, envs_cp in args.run_names:
            if type(ce_data[agents_run_name]['envs'][envs_run_name]) is int:
                run_count += 1
                print(f"cross evaluating agents: {agents_run_name} against environments: {envs_run_name}")
                args.run_a = agents_run_name
                args.chk_a = agents_cp
                args.run_b = envs_run_name
                args.chk_b = envs_cp
                agents_a_vs_env_b = cross_eval(args)   
                ce_data[agents_run_name]['envs'][envs_run_name] = agents_a_vs_env_b
                print(f"Saving results for eval of {agents_run_name}, on {envs_run_name}")
                with open(os.path.join(args.output_dir, args.cross_eval_data_filename), 'wb') as f:
                    pickle.dump(ce_data, f)

            else:
                print(f"eval for {agents_run_name}, on {envs_run_name} has already been run")
    
    


    print('done')

        
if __name__ == "__main__":
    main()