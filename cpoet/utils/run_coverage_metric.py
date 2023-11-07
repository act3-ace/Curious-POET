import os, json, time
import subprocess, shutil, random

NUM_WORKERS = 4
# RENAME_DATABRICK = False
# REMOVE_IMAGES = True
RUN_FOREVER = False
SCAN_ALL_RUNS = None#'/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM'
TIMEOUT_SECONDS = 13000
CHECKPOINT_SUBSTRINGS = ['000']#['250', '500', '750', '000']
run_folders = [
    f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/baseline_wCM_17nov_seed24582924",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17nov",   ########### every 250
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17feb",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_21feb",

    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_19nov",   
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_28nov", 
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_29nov", 

    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_2dec",   
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_21nov", 
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_30nov_a", 
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_21feb",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_7mar",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma7.5_wCM_7marA",

    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8nov",  
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_28nov", 
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_30nov", ############# every 250
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_30nov_short", ############# every 250
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_27jan", 
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8mar",   
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_8marA",   

    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma15.0_wCM_1dec",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma15.0_wCM_2dec_a",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma15.0_wCM_28nov", 

    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma20.0_wCM_8nov",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma20.0_start100_19nov",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma20.0_wCM_2dec",

    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma25.0_wCM_28nov",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma25.0_wCM_4dec",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma25.0_wCM_4dec_a",

    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_16nov",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_4dec",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_4dec_a",

    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma35.0_wCM_21nov",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma35.0_wCM_6dec",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma35.0_wCM_6dec_a",

    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma40.0_wCM_8dec",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma40.0_wCM_8dec_a",
    # "/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma40.0_wCM_9dec"
    ]

except_folders = [
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma20.0_start100_19nov",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma15.0_wCM_1dec",
    # f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma30.0_wCM_16nov",
]

for n in [20]: # use this for mulitple integer n in mode below.  not used otherwise.
    print(f"\n\n\n\n\n n={n}--------------------------------------------")
    cov_met_config={
        'num_cppns': 10000,  # overrides for looping experiment
        # t_list = [-100, -75, -50, 0, 50, 100, 150, 200, 230, 250, 270, 290] 
        # t_list = [-100, -75, -50, 0, 50,  230, 250, 270, 290] 
        't_list': [230], 
        'k_list': [100],
        # k_list = [20, 35, 50, 75, 100]
        'num_iters': 1,
        'mode': {'index':100}, # 'normal', 'only_last', integer n = keep every nth, {'index':N}, where N is the one you want
    }
    if cov_met_config['mode']=='only_last':
        cov_met_template = f"fixedset_coverage_metric_onlyLast_wRollouts"
    if cov_met_config['mode'] == 'make_gifs':
        cov_met_template = f"fixedset_coverage_metric_make_gifs"
    elif isinstance(cov_met_config['mode'], int):
        cov_met_template = f"fixedset_coverage_metric_every{cov_met_config['mode']}th_wRollouts"
    elif isinstance(cov_met_config['mode'], dict):
        if 'index' in cov_met_config['mode']:
            cov_met_template = f"fixedset_coverage_metric_only_{cov_met_config['mode']['index']}th_wRollouts"
        elif 'specific_env' in cov_met_config['mode']:
            cov_met_template = f"fixedset_coverage_metric_specifically_env_{cov_met_config['mode']['specific_env']}"
    else:
        cov_met_template = f"fixedset_coverage_metric_N10000_wRollouts"

    def already_has_cm_folder(folder):
        for f in os.listdir(folder):
            if cov_met_template in f:
                if cov_met_template == f"fixedset_coverage_metric_make_gifs":
                    return True 
            # if f"fixedset_coverage_metric_N{str(cov_met_config['num_cppns'])}_wRollouts" in f:
                for g in os.listdir(os.path.join(folder, f)):
                    if 'summary.json' in g: # the cov metric must also have run to completion
                        return True
        return False

    count=0
    while RUN_FOREVER or count == 0:
        count += 1
        print(f"Run Forever Count = {count}")

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
        for run_folder in run_folders:
            
            checkpoints = []
            for fn in os.listdir(run_folder):
                if 'cp-' in fn and fn[-3:] in CHECKPOINT_SUBSTRINGS and os.path.isdir(os.path.join(run_folder, fn)):
                # if 'cp-' in fn and ('000' in fn or '250' in fn or '500' in fn or '750' in fn) and os.path.isdir(os.path.join(run_folder, fn)):
                    # folder does not contain a valid coverage_metric_N... folder  (don't rerun cov metric)
                    if not already_has_cm_folder(os.path.join(run_folder, fn)): 
                        checkpoints.append(fn)
            checkpoints.sort()
            checkpoints.sort(key=len)
            # for each checkpoint: 1.) update args, 2.)remove old files if necessary, 3.) run master in restart mode
                # update arguments in args.json
            cm_config_fn = os.path.join(run_folder, "cov_met_config.json")
            with open(cm_config_fn, 'w') as f:
                json.dump(cov_met_config, f)        

            for index, checkpoint in enumerate(checkpoints):
                # update arguments in args.json
                print(f"*************** Evaluating Checkpoint{checkpoint} ({index} / {len(checkpoints)}) in run folder: {run_folder} ************************")
                fn = os.path.join(run_folder, checkpoint, "args.json")
                with open(fn) as f:
                    args = json.load(f)
                    args['num_workers'] = NUM_WORKERS
                    args['run_coverage_metric'] = True
                    args['niche']="ICM_ES_Bipedal"
                    args['niche_params'] = {
                        "use_icm": True,
                        "icm_stable_iteration": 200,
                        "icm_training_gamma": 0.0                        
                    }
                    args["visualize_freq"] = 1 if (cov_met_config['mode'] == 'make_gifs' or (isinstance(cov_met_config['mode'], dict) and 'specific_env' in cov_met_config['mode'])) else 0
                with open(fn, 'w+') as f:
                    json.dump(args, f)

                # remove old coverage metric folder
                # if os.path.exists(os.path.join(run_folder, checkpoint, f"fixedset_coverage_metric_N{str(cov_met_config['num_cppns'])}")):
                #     shutil.rmtree(
                #         os.path.join(run_folder, checkpoint, f"fixedset_coverage_metric_N{str(cov_met_config['num_cppns'])}"),
                #         )
                
                try:
                    return_code=subprocess.run(
                        ['python', "master.py", "--log_file", run_folder, "--start_from", checkpoint], 
                        timeout=TIMEOUT_SECONDS,
                        ) 
                except subprocess.TimeoutExpired as e:
                    print(f"**************** subprocess Timeout after {TIMEOUT_SECONDS} seconds ***************************************")
                    return_code=1
                
                print(f"\n\nCoverage metric process returned with code: {return_code} for checkpoint {checkpoint}")
        time.sleep(10)