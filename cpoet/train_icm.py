import os, time, h5py
from central_icm.model import ICMModel
from central_icm.rollout_dataset import MyIterableDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path
# import wandb
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from gymnasium.spaces import Box
import torch
from multiprocessing import Pool
import socket, json, pickle
from pynng import Rep0
from central_icm.inference import Rollouts

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

def my_collate_fn(data):
    state = []
    next_state = []
    action = []

    for rollout in data:
        state.append(rollout[0])
        next_state.append(rollout[1])
        action.append(rollout[2])
    
    concatenated_state = np.concatenate(state)
    concatenated_next_state = np.concatenate(next_state)
    concatenated_action = np.concatenate(action)
    # print(f"size: {concatenated_action.shape[0]}")
    return (concatenated_state, concatenated_next_state, concatenated_action)

def get_latest_checkpoint_number(folder):
    files = os.listdir(folder)
    checkpoint_numbers = []
    for fn in files:
        if '.pth' in fn:
            checkpoint_numbers.append(int(fn[fn.find('_')+1:fn.find('.pth')]))



    return max(checkpoint_numbers)

def main():
    rollouts = Rollouts(device=device)
    
    # os.environ["WANDB_SILENT"] = "true"
    # wandb.init(project=os.environ['RUN_NAME']+'_ICM')
    # parser = ArgumentParser()
    # parser.add_argument('log_file')
    # args = parser.parse_args()
    rollout_hdf5_folder = os.path.join(os.environ['OUTPUT_DIR'], os.environ['RUN_NAME'], 'icm/_rollouts')

    hostinfo = "tcp://" + socket.gethostbyname(socket.gethostname()) + ":13131"
    with open(os.path.join(os.environ['OUTPUT_DIR'], os.environ['RUN_NAME'], 'icm/hostinfo.json'), 'w') as f:
        f.write(json.dumps(hostinfo))


    print("Creating ICM training dataset & dataloader...")
    # load model and dataloader
    dataset = MyIterableDataset(file_path=rollout_hdf5_folder, num_frames=NUM_FRAMES, hist_iter=HIST_ITER)
    
    dummy_action_space = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
    print('Creating ICM model...')
    icm_model = ICMModel(input_size=24*4, feature_size=20, hidden_size=512, output_size=4, action_space=dummy_action_space).to(device)
    opt = torch.optim.Adam(list(icm_model.parameters()), lr=0.0001)
    iteration=0
    try:
        if CHECKPOINT:## load specified checkpoint
            cp_folder = os.path.join(os.environ['OUTPUT_DIR'], os.environ['RUN_NAME'], 'icm/checkpoints')
            if type(CHECKPOINT) == str:
                checkpoint_number = get_latest_checkpoint_number(cp_folder)
            else:
                checkpoint_number = CHECKPOINT
            print(f"Loading checkpoint {checkpoint_number}")
            checkpoint = torch.load(os.path.join(cp_folder, f"checkpoint_{str(checkpoint_number)}.pth"))
            icm_model.load_state_dict(checkpoint['model']) # model
            opt.load_state_dict(checkpoint['optimizer']) # optimizer
            iteration = checkpoint['iteration']
            moments = checkpoint['moments']# moments

            rollouts.obs_moments.mean = moments['obs_mean']
            rollouts.obs_moments.var = moments['obs_var']
            rollouts.obs_moments.count = moments['obs_count']
            rollouts.rewards_moments.mean = moments['reward_mean']
            rollouts.rewards_moments.var = moments['reward_var']
            rollouts.rewards_moments.count = moments['reward_count']
    except Exception as e:
        print(f"Loading Checkpoint Encountered exception: {e} ")

        
    # wandb.watch(icm_model, log_freq=100)
    # Train loop

    print(f"Checking for rollout files in {rollout_hdf5_folder}...")
    h5_files_present = False
    while not h5_files_present:
        p = Path(rollout_hdf5_folder)
        files = sorted(p.glob('*.h5'))  
        if len(files) >= 1:
            h5_files_present = True
        else:
            rollouts.infer_ri(icm_model, hostinfo)
    start=time.time()

    ret_start = time.time()
    inf_time=0
    print('Begin Training...')    
    for (state, next_state, actions) in DataLoader(
                                            dataset, 
                                            batch_size=BATCH_SIZE, 
                                            num_workers=NUM_WORKERS, 
                                            collate_fn=my_collate_fn,
                                            pin_memory=True
                                            ):
        iteration += 1
        # print(f"main: batch size: {actions.shape[0]}")
        loop_time = time.time()-start
        # print(f"loop time: {loop_time}")
                
        if PROFILE: start = time.time(); print(f"retreive time: {start - ret_start}")
        
        #normalize obs
        rollouts.obs_moments.update(state)#.detach().numpy())  # these lines currently are the long pole: ~100ms
        rollouts.obs_moments.update(next_state)#.detach().numpy())
        if PROFILE: update_time = time.time(); print(f"update time {update_time-start}")

        obs_mean = torch.from_numpy(rollouts.obs_moments.mean.astype(np.float32)).to(device)
        obs_var = torch.sqrt(torch.from_numpy(rollouts.obs_moments.var.astype(np.float32))).to(device)

        if PROFILE: norm_time = time.time();        print(f"retrieve norm time: {norm_time - update_time}")

        state = torch.from_numpy(state).to(device)
        next_state = torch.from_numpy(next_state).to(device)
        actions = torch.from_numpy(actions).to(device)
        if PROFILE: xfer_time = time.time();        print(f"xfer time: {xfer_time - norm_time}")

        state = (state - obs_mean) / obs_var 
        next_state = (next_state - obs_mean) / obs_var 
        if PROFILE: n_time = time.time();        print(f"norm obs on gpu time:{n_time-xfer_time}")

        opt.zero_grad()
        real_next_state_feature, pred_next_state_feature, pred_action = icm_model.forward((state, next_state, actions))
        if PROFILE: fwd_time = time.time();        print(f"infer time:{fwd_time - n_time}")

        forward_loss = mse(real_next_state_feature, pred_next_state_feature)
        inverse_loss = mse(actions, pred_action)
        loss = forward_loss + inverse_loss
        
        loss.backward()
        opt.step()
        if PROFILE: learn_time = time.time();        print(f"learn time {learn_time - fwd_time}")


        if iteration%DISPLAY_INTERVAL == 0:
            print(f"step: {iteration} forward: {forward_loss} inverse: {inverse_loss} loss: {loss}")
            # start_time = time.time()
        if iteration>0 and iteration%CHECKPOINT_INTERVAL == 0:

            torch.save(
                {
                    'model':icm_model.to('cpu').state_dict(),
                    'optimizer':opt.state_dict(),
                    'moments': {
                                'reward_mean':rollouts.rewards_moments.mean, 'reward_var':rollouts.rewards_moments.var, 
                                'reward_count':rollouts.rewards_moments.count,
                                'obs_mean':rollouts.obs_moments.mean, 'obs_var':rollouts.obs_moments.var, 'obs_count':rollouts.obs_moments.count,
                                },
                    'iteration': iteration,
                }, 
                os.path.join(os.environ['OUTPUT_DIR'], os.environ['RUN_NAME'], 'icm/checkpoints', f"checkpoint_{str(iteration)}.pth")
                )
            icm_model.to(device)


        # if wandb.run is not None:
        #     wandb.log({
        #         "icm_training_step": iteration,
        #         "forward_loss":forward_loss,
        #         'inverse_loss':inverse_loss,
        #         'loss':loss,
        #         'obs.mean':rollouts.obs_moments.mean.mean() if rollouts.obs_moments else 0,
        #         'obs.var':rollouts.obs_moments.var.mean() if rollouts.obs_moments else 1,
        #         'rewards.mean':rollouts.rewards_moments.mean.mean() if rollouts.rewards_moments else 0,
        #         'rewards.var':rollouts.rewards_moments.var.mean() if rollouts.rewards_moments else 1,
        #         'inference_time': inf_time,
        #         })
        tic = time.time()  
        if iteration%INFERENCE_INTERVAL == 0:       
            rollouts.infer_ri(icm_model, hostinfo)
        inf_time = time.time()-tic
        ret_start = time.time()
            


if __name__ == '__main__':

    main()