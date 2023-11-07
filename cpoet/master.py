# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import random
from poet_distributed.es import initialize_master_fiber
from poet_distributed.poet_algo import PopulationManager
import os

import yaml
import json
from xvfbwrapper import Xvfb
from utils.coverage_metric import eval_coverage_kjnm as coverage_metric
###############################################################################
## Main Python Functions
###############################################################################
####################
## New POET Run
####################
def run_poet(args):
    '''
    Default Run Function

    This function sets-up and runs the POET algorithm. 
    Currently, it initializes the distributed computing as well, though it 
    doesn't proceed to manage it.

    Parameters
    ----------
    args: argparse.Namespace
        Seems to be like a dictionary. Has all of the parameters within.

    Returns
    -------
        None
    '''
    
    # Initialize fiber
    initialize_master_fiber()

    # If storing gifs, setup directories
    if args.visualize_freq > 0:
        vidpath = os.path.join(args.log_file, "Videos", "Raw")
        logger.info(f"Saving video to {vidpath}")
        os.makedirs(name = vidpath, exist_ok=True)

    # Set master_seed
    #  this is the only way to get reproducibility from neat-python
    random.seed(args.master_seed)

    # Initialize poet population (agent/env pairs) manager
    optimizer_zoo = PopulationManager(args=args)

    # Run POET
    optimizer_zoo.evolve_population(iterations=args.n_iterations,
                    propose_with_adam=args.propose_with_adam,
                    reset_optimizer=True,
                    checkpointing=args.checkpointing,
                    steps_before_transfer=args.steps_before_transfer)


####################
## Restart POET Run
####################
def run_poet_reload(args):
    '''
    Reload Run Function

    This function reloads and continues the POET algorithm.
    Currently, it initializes the distributed computing as well, though it 
    doesn't proceed to manage it.

    Parameters
    ----------
    args: argparse.Namespace
        Seems to be like a dictionary. Has all of the parameters within.

    Returns
    -------
        None
    '''
    # Initialize fiber
    initialize_master_fiber()

    cp_name = args.start_from
    tag = args.logtag
    start_from = os.path.join(args.log_file, cp_name)

    logger.info(f"Starting from {start_from}")

    with open(os.path.join(start_from, "args.json"),"r") as f:
        args.__dict__ = json.load(f)
        if len(tag)>0:
            args.logtag = f"[{tag}:{cp_name}]->"
        else:
            args.logtag = f"[{cp_name}]->"

    # If storing gifs, setup directories
    if args.visualize_freq > 0:
        vidpath = os.path.join(args.log_file, "Videos", "Raw")
        logger.info(f"Saving video to {vidpath}")
        os.makedirs(name = vidpath, exist_ok=True)

    # Set master_seed
    #  this is the only way to get reproducibility from neat-python
    random.seed(args.master_seed)

    # Initialize poet population (agent/env pairs) manager
    #  Immediately load in stored state
    optimizer_zoo = PopulationManager(args=args)
    #print("Args before reload:", )
    optimizer_zoo.reload(start_from)
    print("Args after reload:", optimizer_zoo.args)

    # Tweak to get the reload iteration
    start_iteration = int(start_from.split("-")[-1]) + 1 # Check point at end of iteration
    
    logger.info(f"########### Restarting From Checkpoint At Iter {start_iteration} ############")

    # Run POET
    if args.run_coverage_metric:
        coverage_metric(optimizer = optimizer_zoo, folder=start_from)
    else:    
        optimizer_zoo.evolve_population(iterations=args.n_iterations,
                        propose_with_adam=args.propose_with_adam,
                        reset_optimizer=True,
                        checkpointing=args.checkpointing,
                        steps_before_transfer=args.steps_before_transfer,
                        start_iteration=start_iteration)

    
###############################################################################
## Main CMDLine Function
###############################################################################
def main():
    ''' 
    Main run function. This function parses cmdline arguments and then calls the 
    appropriate :func:`~master.run_poet` or :func:`~master.run_poet_reload` 
    function. 

    Parameters
    ----------
    None

    Returns
    -------
    All information is output to disk.

    '''

    # Init argparser
    parser = ArgumentParser()

    # POET parameters
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--config', default=None)
    parser.add_argument('--niche', default='ES_Bipedal')
    parser.add_argument('--init', default='random')
    parser.add_argument('--visualize_freq', type=int, default=0)
    parser.add_argument('--frame_skip', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.9999)
    parser.add_argument('--lr_limit', type=float, default=0.001)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--noise_decay', type=float, default=0.999)
    parser.add_argument('--noise_limit', type=float, default=0.01)
    parser.add_argument('--l2_coeff', type=float, default=0.01)
    parser.add_argument('--batches_per_chunk', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--eval_batches_per_step', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--n_iterations', type=int, default=200)
    parser.add_argument('--steps_before_transfer', type=int, default=25)
    parser.add_argument('--master_seed', type=int, default=111)
    parser.add_argument('--mc_lower', type=int, default=200)
    parser.add_argument('--mc_upper', type=int, default=340)
    parser.add_argument('--repro_threshold', type=int, default=200)
    parser.add_argument('--max_num_envs', type=int, default=100)
    parser.add_argument('--num_start_envs', type=int, default=1)
    parser.add_argument('--start_envs_type', type=str, default='randAgent',
                        help='Type of environments to start with, one of "randEnv" or "randAgent')
    parser.add_argument('--normalize_grads_by_noise_std', action='store_true', default=False)
    parser.add_argument('--propose_with_adam', action='store_true', default=False)
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--adjust_interval', type=int, default=1)
    parser.add_argument('--returns_normalization', default='normal')
    parser.add_argument('--stochastic', action='store_true', default=True)
    parser.add_argument('--envs', nargs='+')
    parser.add_argument('--start_from', default=None)  # Checkpoint folder to start from
    parser.add_argument('--save_env_params', default=True)
    parser.add_argument('--save_archived_novelty', default=False)
    parser.add_argument('--logtag', default="")
    parser.add_argument('--log_pata_ec', default=True)
    parser.add_argument('--run_coverage_metric', action='store_true', default=False)




    # Parse CMDLine args
    args = parser.parse_args()

    ## Load args from yaml if available
    if args.config:
        with open(args.config, 'r') as f:
            print("Loading config from " + args.config)
            config = yaml.safe_load(f)
            
            for k,v in config.items():
                vars(args)[k] = v
    
    # Log input params
    logger.info(args)

    if args.start_from is not None:
        run_poet_reload(args)
    else:
        run_poet(args)


if __name__ == "__main__":
    with Xvfb() as xvfb:
        main()
