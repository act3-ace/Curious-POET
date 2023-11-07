import h5py, os
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import random, datetime, time
import matplotlib.pyplot as plt
from central_icm.inference import Rollouts
inf_stack_rollout = Rollouts.stack_rollout
from scipy.stats import betabinom

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, num_frames=4, hist_iter=100000):
        super(MyIterableDataset).__init__()
        self.path = file_path
        random.seed(str(datetime.datetime.now())) 
        self.num_frames = num_frames
        latest_fn = self.get_latest_histogram_npy()
        if not latest_fn is None:
            self.histogram = list(np.load(os.path.join(self.path, '../icm_stats', latest_fn), allow_pickle=True))
        else:
            self.histogram = [0]*100
        #should load the npy histogram on disk at this point!!  #TODO
        print('Initializing iterable dataset')
        self.iter = 0
        self.hist_iter = hist_iter
        self.a = 5
        self.b = 0.1

    def get_latest_histogram_npy(self):
        latest = 0
        latest_fn=None
        try:
            for f in os.listdir(os.path.join(self.path, '../icm_stats')):
                if '.npy' in f:
                    iter_number = int(f[f.rfind('_')+1:-4])
                    if iter_number > latest:
                        latest = iter_number
                        latest_fn = f
                print(f"loading {f} in histogram")
        except FileNotFoundError as e:
            print(f"No Histogram table found.  Starting up without one.")
        return latest_fn

    def update_histogram(self, file_index):
        while file_index > len(self.histogram)-2: # grow as needed
            self.histogram.append(0)
        self.histogram[file_index] += 1

        if self.iter % self.hist_iter == 0:
            dist_filename = os.path.abspath(os.path.join(
                self.files[0].parent,
                    '../icm_stats/', 
                    f"rollout_sampling_dist_iter_{str(self.iter)}"))
            np.save(dist_filename+'.npy', np.asarray(self.histogram))
            f = plt.figure()
            plt.stairs(self.histogram)
            plt.title(f"Sample count vs Rollout File Index (iter: {self.iter})\nw/ Beta Binomial Sampling ($\\alpha$={self.a}, $\\beta$={self.b})")
            plt.savefig(dist_filename+'.png')
            del f



    def __iter__(self):
        while True:
            self.iter += 1
            task_complete = False
            while not task_complete:
                try:
                    p = Path(self.path)  # update file list (this might stress the file system)
                    assert(p.is_dir())
                    self.files = sorted(p.glob('*.h5'))
                    self.files.sort(key=os.path.getctime)  #order by write time (oldest last)
                    if len(self.files) < 1:
                        raise RuntimeError('No hdf5 datasets found')

                    worker_info = torch.utils.data.get_worker_info()
                    # print(f"worker num {worker_info.id}")


                    # file_to_open = self.files[-1]
                    # file_index = len(self.files)-1

                    # if worker_info.id == 0 and len(self.files) > 1:  # all the other workers
                        # open any h5 file but the last one written
                    # file_index = random.randint(0, len(self.files)-1) 
                    # analysis shows (https://git.act3-ace.com/stalwart/curious-poet/-/issues/32)
                    # the beta binomial distribution, when parameterized appropriately, should
                    # result in an approximately uniform distribution for a growing population
                    file_index = betabinom.rvs(n=len(self.files)-1, a=self.a, b=self.b)


                    file_to_open = self.files[file_index]
                    self.update_histogram(file_index)
                    
                    with h5py.File(file_to_open) as h5_file:
                        groups = list(h5_file.keys())
                        group = groups[random.randrange(len(groups))]
                        num_rollouts = len(h5_file[group].keys())
                        rollout = h5_file[group]['data_'+str(random.randrange(num_rollouts))]
                        state, next_state, action = inf_stack_rollout(rollout=rollout)

                    task_complete = True
                except Exception as e:
                    print(f"exception: {e}: filename: {file_to_open}")
                    time.sleep(0.1)
            # these lines validated that data from multiple workers is actually used for training
            # print(f"worker: {worker_info.id}  size: {action.shape[0]}")
            # if worker_info.id==0:
            #     action = np.zeros_like(action)
            # else:
            #     action = np.ones_like(action)
            yield state, next_state, action

