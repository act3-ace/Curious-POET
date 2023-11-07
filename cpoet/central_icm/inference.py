import os, time, h5py, pickle
import numpy as np
from pathlib import Path
from stable_baselines3.common.running_mean_std import RunningMeanStd
# import wandb
import torch
from pynng import Rep0
# from central_icm.rollout_dataset import MyIterableDataset

NUM_FRAMES = 4  #frame stacking length


class Rollouts():
    def __init__(self, device='cuda:0', max_file_size=1e9):
        self.rollout_hdf5_folder = os.path.join(os.environ['OUTPUT_DIR'], os.environ['RUN_NAME'], 'icm/rollouts')
        self.curiosity_rewards_folder = os.path.join(os.environ['OUTPUT_DIR'], os.environ['RUN_NAME'], 'icm/curiosity_rewards')
        self._rollout_hdf5_folder = os.path.join(os.environ['OUTPUT_DIR'], os.environ['RUN_NAME'], 'icm/_rollouts')
        self.current_h5_filename = None
        if not os.path.exists(self._rollout_hdf5_folder):
            os.makedirs(self._rollout_hdf5_folder, exist_ok=True)
        self.device = device
        self.rollout_files = []
        self.obs_moments = None
        self.rewards_moments = None
        self.obs_moments=RunningMeanStd(shape = 96)
        self.rewards_moments = RunningMeanStd(shape = (1))
        self.max_file_size = max_file_size
        self.rollout_file_index = self.get_rollout_index()

    def get_rollout_index(self):
        p = Path(self._rollout_hdf5_folder)  # update file list (this might stress the file system)
        assert(p.is_dir())
        files = sorted(p.glob('*.h5'))
        files.sort(key=os.path.getctime)
        if len(files) > 0:
            start_char = files[-1].stem.rfind('_')+1
            return int(files[-1].stem[start_char:])
        else:
            return 0


    @staticmethod
    def stack_rollout(rollout):

        r = rollout
        length = r.shape[0]
        s = r[:,:24]
        a = r[:,24:28]  #discard last bc of last next state
        mat = np.zeros((len(rollout)+(NUM_FRAMES-1), 24*NUM_FRAMES), dtype=np.float32)
        for i in range(NUM_FRAMES):                             # [0,1,2,3]
            mat[i:i+length, i*24:(i+1)*24] = s
            
        state       = mat[NUM_FRAMES-1:-(NUM_FRAMES),:]
        next_state  = mat[NUM_FRAMES:-(NUM_FRAMES-1),:]
        action =        a[NUM_FRAMES-1:-1,:]
        return state, next_state, action

    def infer_ri(self, model, hostinfo):
        start = time.time()
        rollout_message = None

        try:
            with Rep0(listen=hostinfo, block_on_dial=False, recv_timeout=500 ) as rep:
            
                rollout_message = pickle.loads(rep.recv())
                intrinsic_rewards = self._infer_ri(rollout_message, model)
                rollout_message['intrinsic_rewards'] = intrinsic_rewards
                rep.send(pickle.dumps(intrinsic_rewards))
                # rep.send(intrinsic_rewards)
                print(f"Sent an inference response!  Mean Ri = {intrinsic_rewards.mean():3.3f}, mean Re={rollout_message['extrinsic_rewards'].mean():3.3f} @ POET iter: {rollout_message['iteration']}")
        except Exception as e:
            pass
            # print(e)
        if rollout_message:
            self.store(rollout_message)
        return


    def store(self, rollout_message):
        if not self.current_h5_filename:
            self.current_h5_filename = os.path.join(self._rollout_hdf5_folder, f"rollout_buffer_{self.rollout_file_index}.h5")
        if os.path.exists(self.current_h5_filename) and os.path.getsize(self.current_h5_filename) > self.max_file_size:
            self.rollout_file_index += 1
            self.current_h5_filename = os.path.join(self._rollout_hdf5_folder, f"rollout_buffer_{self.rollout_file_index}.h5")

        try:
            with h5py.File(self.current_h5_filename, 'a') as f:
                group_name = f"optim_{rollout_message['optim_id']}_iter_{rollout_message['iteration']}"
                f.create_group(group_name)
                count = 0
                for i, mirrored_rollout in enumerate(rollout_message['rollouts']):
                    for j, rollout in enumerate(mirrored_rollout):
                        f.create_dataset(f"{group_name}/data_{str(count)}", data=rollout, dtype='float32')
                        # f.create_dataset(f"{group_name}/Ri_{str(count)}", data=rollout_message['intrinsic_rewards'][i,j], dtype='float32')
                        count += 1
           
        except Exception as e:
            print(f"Failed to write to rollout buffer file: {self.current_h5_filename} with exception {e}")

        return

    # def infer_ri(self, model, hostinfo):
    #     start = time.time()
    #     rollout_message = None

    #     with Rep0(listen=hostinfo, recv_timeout=500) as rep:
    #         try:
    #             rollout_message = rep.recv() #pickle.loads(rep.recv())
    #             # print(f"fd {rep.recv_fd}")
    #             intrinsic_rewards = self._infer_ri(rollout_message, model)
    #             # rep.send(pickle.dumps(intrinsic_rewards))
    #             rep.send(intrinsic_rewards)
    #             print(f"Sent an inference response!")
    #         except Exception as e:
    #             print(f"{e}")
    #     return

    def _infer_ri(self, rollout_message, model):

        # rollout_message = pickle.loads(rollout_message)

        start = time.time()
        rollouts = rollout_message['rollouts']
        rewards = []
        mirrored_rewards = []
        for rollout_pair in rollouts:
            for rollout in rollout_pair:
                
                state, next_state, action = self.stack_rollout(rollout)

                #normalize observations
                obs_mean = self.obs_moments.mean.astype(np.float32)
                obs_var = np.sqrt(self.obs_moments.var.astype(np.float32))
                state = (state - obs_mean) / obs_var 
                next_state = (next_state - obs_mean) / obs_var
                _norm_time = time.time()


                # convert to torch and move to device
                state = torch.from_numpy(state).detach().to(self.device)
                next_state = torch.from_numpy(next_state).detach().to(self.device)
                action = torch.from_numpy(action).detach().to(self.device)
                _conv_time = time.time(); convert_time = _conv_time-_norm_time#;  print(f"convert time: {convert_time}")

                # infer Ri
                real_next_state_feature, pred_next_state_feature, _ = model.forward((state, next_state, action), train=False)
                ri = torch.nn.MSELoss()(real_next_state_feature, pred_next_state_feature).cpu().detach().numpy()
                _infer_time = time.time(); infer_time = _infer_time - _conv_time#; print(f"infer time {infer_time}")

                mirrored_rewards.append(ri)
                if len(mirrored_rewards)>1:
                    rewards.append(mirrored_rewards)
                    mirrored_rewards = []
                loop_time = time.time()
        inf_time = time.time()- start
        print(f"inference took: {inf_time}")
        
        #normalize rewards
        _rewards = np.stack(rewards)
        self.rewards_moments.update(_rewards.flatten()) # this is the only place intrinsic rewards are computed
        normalized_rewards = _rewards / self.rewards_moments.var
        
        assert normalized_rewards.shape[0] == rollout_message['shape']

        # normalized_rewards = pickle.dumps(normalized_rewards)

        return normalized_rewards