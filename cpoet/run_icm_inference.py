import os, json, time
import subprocess, shutil, random
import pickle
from icm_inference import infer_ri 


MODE = 3 # 1= run ICM inference, 2=aggregate inference result jsons
mode_2_results_folder = '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/aggregated_ICM_inference_results'

SCAN_ALL_RUNS = None#'/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM'


# Note on use:  Just add a new run to run_folders and re-run this script in tmux.  It will only perform missing cross inference.
run_folders = [

    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17nov",
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17feb",
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_21feb",

    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_19nov",   
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_28nov", 
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_29nov", 

    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_2dec",   
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_21nov", 
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_30nov_a", 
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_21feb",

    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8nov",   
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_28nov", 
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_30nov",  
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_27jan",

    ]

except_folders = [
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/baseline_wCM_14nov_seed24582923",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/baseline_wCM_17nov_seed24582924", # not done yet.  Also a lot of empties
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_16nov",
]

cov_met_config={
    'num_cppns': 10000,  # overrides for looping experiment
    # t_list = [-100, -75, -50, 0, 50, 100, 150, 200, 230, 250, 270, 290] 
    # t_list = [-100, -75, -50, 0, 50,  230, 250, 270, 290] 
    't_list': [230], 
    'k_list': [100],
    # k_list = [20, 35, 50, 75, 100]
    'num_iters': 1,
}

def results_present(icm_folder, run_folder):
    #re-create fn to look for from run_folder
    run_name = run_folder[run_folder.rfind('/')+1:]

    # if ANY inference results are already in icm_folder/inference, return true
    inf_dir = os.path.realpath(os.path.join(icm_folder, '../inference'))

    if not os.path.exists(inf_dir):
        return False

    for fn in os.listdir(inf_dir):
        if run_name in fn:
            print('True')
            return True

    return False


def has_cm_folder(folder):
    for f in os.listdir(folder):
        if f"fixedset_coverage_metric_N{str(cov_met_config['num_cppns'])}_wRollouts" in f:
            if 'rollouts.pkl' in os.listdir(os.path.join(folder, f)):  
                return True
    return False

if SCAN_ALL_RUNS:
    run_folders=[]
    for f in os.listdir(SCAN_ALL_RUNS):
        if os.path.isdir(os.path.join(SCAN_ALL_RUNS, f)) and '.vscode' not in f and 'baseline' not in f:
            if os.path.join(SCAN_ALL_RUNS, f) not in except_folders:
                run_folders.append(os.path.join(SCAN_ALL_RUNS, f))
run_folders.sort()

print(f"Current Run Folder List:")
for run_folder in run_folders:
    print(run_folder)


    
for icm_folder in run_folders:
    icm_folder = os.path.join(icm_folder, 'icm', 'checkpoints')

    for run_folder in run_folders:
        checkpoints = []
        if MODE==1:
            for fn in os.listdir(run_folder):
                if 'cp-' in fn and '000' in fn and os.path.isdir(os.path.join(run_folder, fn)):
                    if has_cm_folder(os.path.join(run_folder, fn)): 
                        if not results_present(icm_folder, run_folder):
                            checkpoints.append(fn)
            checkpoints.sort()
            checkpoints.sort(key=len)
            print(checkpoints)
            # for each checkpoint: run analysis code


            for index, checkpoint in enumerate(checkpoints):

                print(f"*************** ICM loaded from {icm_folder} *********************")
                print(f"*************** Running ICM inference on Checkpoint {checkpoint} ({index} / {len(checkpoints)}) in run folder: {run_folder} ************************")

                infer_ri(
                    checkpoint_folder=icm_folder,
                    rollout_pickle_file=os.path.join(run_folder, checkpoint, 'fixedset_coverage_metric_N10000_wRollouts/rollouts.pkl')
                )
        if MODE==2:
            if 'inference' in os.listdir(os.path.join(run_folder, 'icm')):
                #copy contents to aggregated jsons folder
                for fn in os.listdir(os.path.join(run_folder, 'icm', 'inference')):
                    if '.json' in fn:
                        dest_fn = 'ICM_' + run_folder[run_folder.rfind('/')+1:] + '_inferring_on_AGENTS_' + fn
                        shutil.copy(os.path.join(run_folder, 'icm', 'inference', fn), os.path.join(mode_2_results_folder, dest_fn))


if MODE==3: # update json file with reward_var (moment from icm checkpoint).  (one time use probably)
    # for every icm folder, open the checkpoint, retrieve the reward_var (scalar)
    # aggregate into a dict, and finally store the dict in the mode2 output location
    import torch
    def get_latest_checkpoint_number(folder):  # borrowed from icm_inference.py and train_icm.py
        files = os.listdir(folder)
        checkpoint_numbers = []
        for fn in files:
            if '.pth' in fn:
                checkpoint_numbers.append(int(fn[fn.find('_')+1:fn.find('.pth')]))
        return max(checkpoint_numbers)

    reward_vars={}


    for icm_folder in run_folders:
        icm_folder = os.path.join(icm_folder, 'icm', 'checkpoints')
        checkpoint_number = get_latest_checkpoint_number(icm_folder)
        print(f"Loading checkpoint {checkpoint_number}")
        checkpoint = torch.load(os.path.join(icm_folder, f"checkpoint_{str(checkpoint_number)}.pth"))
        moments = checkpoint['moments']# moments

        remain = icm_folder[icm_folder.find('icm_gamma'):]
        icm_name = remain[:remain.find('/')]

        reward_vars[icm_name]=float(moments['reward_var'])
    
    reward_vars_fn = os.path.join(mode_2_results_folder, 'reward_vars.json')
    with open(reward_vars_fn, 'w') as f:
        json.dump(reward_vars, f)

print(f"done running in mode {MODE}")