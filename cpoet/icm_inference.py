import os, time, h5py
from central_icm.model import ICMModel
from central_icm.rollout_dataset import MyIterableDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path
# import wandb
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from gym.spaces import Box
import torch
from multiprocessing import Pool
import socket, json, pickle
from pynng import Rep0
from central_icm.inference import Rollouts
import argparse
inf_stack_rollout = Rollouts.stack_rollout
import tqdm
from matplotlib import pyplot as plt

BATCH_SIZE = 100  
DISPLAY_INTERVAL = 1
CHECKPOINT_INTERVAL = 5000
INFERENCE_INTERVAL = 1  # check for and do inference every this many training cycles
NUM_FRAMES = 4
NUM_WORKERS = 0
CHECKPOINT = 'latest'  # the histogram mechanism always finds the newest histogram npy, indep of this setting.
HIST_ITER = 200000
PROFILE = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse = torch.nn.MSELoss()


def get_latest_checkpoint_number(folder):
    files = os.listdir(folder)
    checkpoint_numbers = []
    for fn in files:
        if '.pth' in fn:
            checkpoint_numbers.append(int(fn[fn.find('_')+1:fn.find('.pth')]))



    return max(checkpoint_numbers)

def infer_ri(checkpoint_folder, rollout_pickle_file):
    # rollouts = Rollouts(device=device)


    # load model and dataloader
    # dataset = MyIterableDataset(file_path=rollout_hdf5_folder, num_frames=NUM_FRAMES, hist_iter=HIST_ITER)
    
    dummy_action_space = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
    print('Creating ICM model...')
    icm_model = ICMModel(input_size=24*4, feature_size=20, hidden_size=512, output_size=4, action_space=dummy_action_space).to(device)
    opt = torch.optim.Adam(list(icm_model.parameters()), lr=0.0001)
    iteration=0
    try:
        if CHECKPOINT:## load specified checkpoint
            cp_folder = checkpoint_folder
            if type(CHECKPOINT) == str:
                checkpoint_number = get_latest_checkpoint_number(cp_folder)
            else:
                checkpoint_number = CHECKPOINT
            print(f"Loading checkpoint {checkpoint_number}")
            checkpoint = torch.load(os.path.join(cp_folder, f"checkpoint_{str(checkpoint_number)}.pth"))
            icm_model.load_state_dict(checkpoint['model']) # model
            moments = checkpoint['moments']# moments


    except Exception as e:
        print(f"Loading Checkpoint Encountered exception: {e} ")

        
    with open(rollout_pickle_file, 'rb') as f:
        rollouts = pickle.load(f)


    obs_mean = moments['obs_mean']
    obs_var  = moments['obs_var']
    reward_var = moments['reward_var']

    # ret_start = time.time()
    # inf_time=0
    print('Begin inference...') 
    ri=[]   
    for rollout in tqdm.tqdm(rollouts):

        state, next_state, actions = inf_stack_rollout(rollout[0])

        state = (state - obs_mean) / obs_var 
        next_state = (next_state - obs_mean) / obs_var

        state = torch.from_numpy(state.astype('float32')).to(device)
        next_state = torch.from_numpy(next_state.astype('float32')).to(device)
        actions = torch.from_numpy(actions.astype('float32')).to(device)

        real_next_state_feature, pred_next_state_feature, pred_action = icm_model.forward((state, next_state, actions))
        rollout_ri = torch.nn.MSELoss()(real_next_state_feature, pred_next_state_feature).cpu().detach().numpy()

        ri.append(rollout_ri / reward_var)

    stats = {
        'mean_ri' : float(np.mean(ri)),
        'max_ri'  : float(np.max(ri)),
        'min_ri'  : float(np.min(ri)),
        'std_ri'  : float(np.std(ri)),
    }

    inf_dir = os.path.realpath(os.path.join(checkpoint_folder, '../inference'))
    if not os.path.exists(inf_dir):
        os.makedirs(os.path.realpath(os.path.join(checkpoint_folder, '../inference')))

    a=rollout_pickle_file
    start = a.find('icm_gamma')
    end = a[start:].find('/') + start
    run_name = a[start:end]

    start = end+1
    end = a[start:].find('/') + start
    cp_name = a[start:end]

    out_fn = os.path.join(inf_dir, run_name+'_'+cp_name+'.json')
    with open(out_fn, 'w+') as f:
        json.dump(stats, f)
    
    c = checkpoint_folder
    start = c.find('icm_gamma')
    end = c[start:].find('/') + start
    cp_run_name = c[start:end]


    out_fig_fn = os.path.join(inf_dir, run_name+'_'+cp_name+'.png')
    fig, axs = plt.subplots()
    axs.plot(range(len(ri)), ri)
    axs.set_title(
        f"Intrinisc Reward from ICM trained in run: {cp_run_name}\n"+\
        f"Agents trained in run {run_name}\n (at checkpoint {cp_name})\n" +\
        f"Evaluated against 10k fixed Coverage Metric Environments" 
    )
    axs.grid(True)
    fig.tight_layout()
    fig.savefig(out_fig_fn)

    print(stats)


def main(args):
    infer_ri(
        checkpoint_folder=args.checkpoint_folder, 
        rollout_pickle_file=args.rollout_pickle_file
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_folder', type=str, 
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17feb/icm/checkpoints'
        )
    parser.add_argument('--rollout_pickle_file', type = str, 
        # default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma0.0_wCM_17feb/cp-2000/fixedset_coverage_metric_N10000_wRollouts/rollouts.pkl'
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma5.0_wCM_28nov/cp-2000/fixedset_coverage_metric_N10000_wRollouts/rollouts.pkl'

        )
    args = parser.parse_args()
    main(args)