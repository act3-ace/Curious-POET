import os, json, time
import subprocess, shutil, random
import pickle
from covmet_analysis import explore


MODE = 1 # 1= run coverage metric analysis, 2=aggregate gait char pickles
mode_2_output_folder= '/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/aggregated_gait_chars'
DEL_OUTPUT_FOLDER = False

SCAN_ALL_RUNS = None#'/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM'
OVERWRITE_ANALYSIS_FOLDERS = False
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
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_7mar",
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_7marA",

    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8nov",   
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_28nov", 
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_30nov",  
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_27jan", 
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8mar",   
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8marA",

    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma15.0_wCM_1dec",
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma15.0_wCM_2dec_a",
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma15.0_wCM_28nov",     
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



def already_has_cm_folder(folder):
    for f in os.listdir(folder):
        if f"fixedset_coverage_metric_N{str(cov_met_config['num_cppns'])}_wRollouts" in f:
            if 'kjnm_databrick.npy' in os.listdir(os.path.join(folder, f)):  # must be more than the analysis folder 
                if 'analysis' in os.listdir(os.path.join(folder, f)) and not OVERWRITE_ANALYSIS_FOLDERS:
                    return True
    return False


# state_action_list = []

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

if MODE == 2:
    if DEL_OUTPUT_FOLDER and os.path.exists(mode_2_output_folder):
        shutil.rmtree(mode_2_output_folder)
        os.makedirs(mode_2_output_folder)
    
for run_folder in run_folders:

    checkpoints = []
    for fn in os.listdir(run_folder):
        if 'cp-' in fn and '000' in fn and os.path.isdir(os.path.join(run_folder, fn)):
            # folder does not contain a valid coverage_metric_N... folder  (don't rerun cov metric)
            if not already_has_cm_folder(os.path.join(run_folder, fn)) or MODE == 2: 
                checkpoints.append(fn)
    checkpoints.sort()
    checkpoints.sort(key=len)
    print(checkpoints)
    # for each checkpoint: run analysis code


    for index, checkpoint in enumerate(checkpoints):
        if MODE == 1:
            print(f"*************** Running analysis on Checkpoint {checkpoint} ({index} / {len(checkpoints)}) in run folder: {run_folder} ************************")
            # try:
                # state_action_list.append(explore(os.path.join(run_folder, checkpoint, 'fixedset_coverage_metric_N10000_wRollouts')))
            explore(os.path.join(run_folder, checkpoint, 'fixedset_coverage_metric_N10000_wRollouts'))

            # except Exception as e:
            #     print(f"Encountered Exception: {e}")

            # with open(os.path.join(SCAN_ALL_RUNS, 'state_action_stats_data.pkl'), 'wb+') as f:
            #     pickle.dump(state_action_list, f)

        if MODE == 2:
            dst = os.path.join(mode_2_output_folder, f"{run_folder[run_folder.rfind('/')+1:]}_{checkpoint}_gait_chars.pkl")
            if not os.path.exists(dst):
                print(f"copying file: {dst}")
                shutil.copy(
                    src=os.path.join(run_folder, checkpoint, 'fixedset_coverage_metric_N10000_wRollouts','analysis','gait_chars.pkl'),
                    dst=dst,
                )

            dst = os.path.join(mode_2_output_folder, f"{run_folder[run_folder.rfind('/')+1:]}_{checkpoint}_stateactions.npy")
            if not os.path.exists(dst):  
                print(f"copying file: {dst}")          
                shutil.copy(
                    src=os.path.join(run_folder, checkpoint, 'fixedset_coverage_metric_N10000_wRollouts','analysis','stateactions.npy'),
                    dst=dst,
                )

print(f"done running in mode {MODE}")