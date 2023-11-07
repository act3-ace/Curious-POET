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


import logging
import time
import numpy as np
from numpy.random import Generator, PCG64DXSM
from collections import namedtuple
from .stats import compute_CS_ranks
from .logger import CSVLogger
import json
import functools
import os
from .iotools import NumpyEncoder, NumpyDecoder

#######################################
#
#       Custom Datatypes
#
#######################################

StepStats = namedtuple('StepStats', [
    'po_returns_mean',
    'po_returns_median',
    'po_returns_std',
    'po_returns_max',
    'po_theta_max',
    'po_returns_min',
    'po_len_mean',
    'po_len_std',
    'noise_std',
    'learning_rate',
    'theta_norm',
    'grad_norm',
    'update_ratio',
    'episodes_this_step',
    'timesteps_this_step',
    'time_elapsed_this_step',
])

EvalStats = namedtuple('StepStats', [
    'eval_returns_mean',
    'eval_returns_median',
    'eval_returns_std',
    'eval_len_mean',
    'eval_len_std',
    'eval_n_episodes',
    'time_elapsed',
    'eval_rollouts'
])

POResult = namedtuple('POResult', [
    'noise_inds',
    'returns',
    'lengths',
    'rollouts',
])
EvalResult = namedtuple('EvalResult', ['returns', 'lengths', 'rollouts'])

logger = logging.getLogger(__name__)

#######################################
#
#       Fiber Utility Functions
#
#######################################

def initialize_master_fiber():
    global noise
    from .noise_module import noise

def initialize_worker_fiber(arg_thetas, arg_niches):
    global noise, thetas, niches
    from .noise_module import noise
    thetas = arg_thetas
    niches = arg_niches

#@functools.lru_cache(maxsize=1000)
def fiber_get_theta(optim_id):
    return thetas[optim_id]

#@functools.lru_cache(maxsize=1000)
def fiber_get_niche(optim_id):
    #logger.info("Getting a niche")
    return niches[optim_id]


#######################################
#
#       Perform Runs
#
#######################################

# def run_eval_batch_fiber(iteration, optim_id, batch_size, rs_seed):
#     global niches, thetas
#     random_state = np.random.RandomState(rs_seed)
#     niche = fiber_get_niche(optim_id)
#     theta = fiber_get_theta(optim_id)

#     returns, lengths = niche.rollout_eval_batch((theta for i in range(batch_size)),
#                                            batch_size, random_state, eval=True)

#     return EvalResult(returns=returns, lengths=lengths)

def run_eval_batch_fiber(iteration, target_env_id, batch_size, rs_seed, source_theta):
    '''
    Parameters:
    source_theta: Numpy array of thetas
    target_env_id: optim_id of the environment
    '''
    global niches
    random_state = Generator(PCG64DXSM(seed=rs_seed))
    niche = fiber_get_niche(target_env_id)

    returns, lengths, rollouts = niche.rollout_eval_batch((source_theta for i in range(batch_size)),
                                           batch_size, random_state, return_rollouts=True)

    return EvalResult(returns=returns, lengths=lengths, rollouts=rollouts)

def run_po_batch_fiber(iteration, optim_id, batch_size, rs_seed, noise_std):
    global noise, niches, thetas
    random_state = Generator(PCG64DXSM(seed=rs_seed))
    niche = fiber_get_niche(optim_id)
    theta = fiber_get_theta(optim_id)

    return(niche.rollout_opt_batch(theta, batch_size, random_state, noise))

def run_single_po_batch_fiber(iteration, optim_id, batch_size, rs_seed, noise_std, theta):
    global noise, niches
    random_state = Generator(PCG64DXSM(seed=rs_seed))
    niche = fiber_get_niche(optim_id)

    return(niche.rollout_opt_batch(theta, batch_size, random_state, noise))


#######################################
#
#       ESOptimizer Class
#
#######################################

class ESOptimizer: 
    """
    Description of __init__

    Parameters
    ----------
    fiber_pool: mp_ctx.Pool
        pool object for the optimizer
    fiber_shared: dict
        Dictionary of niches and thetas
    theta: np.Array
        Stored parameters of the network
    make_niche: Niche function
    niche_kwargs: Dictionary of arguments for  `make_niche`
    learning_rate: float
    batches_per_chunk: int
    batch_size: int
    eval_batch_size: int
    eval_batches_per_step: int
    l2_coeff: float
    noise_std: float
    lr_decay: int 
    lr_limit: float
    noise_decay: int
    noise_limit: float
    normalize_grads_by_noise_std: boolean
    returns_normalization: string
    optim_id: int
    log_file: string
    created_at: int
    is_candidate: boolean
    nolog: boolean

    Methods
    -------
    clean_dicts_before_iter()
    save_to_logger(iteration)
    save_policy(policy_file, reset=False)
    update_dicts_after_transfer(source_optim_id, source_optim_theta, stats, keyword)
    update_dicts_after_es(stats, self_eval_stats)
    add_env(env)
    delete_env(env_name)
    broadcast_theta(theta)
    start_chunk_fiber(runner, batches_per_chunk, batch_size, *args)
    get_chunk(tasks)
    collect_po_results(po_results)
    compute_grads(step_results, theta)
    set_theta(theta, reset_optimizer=True)
    start_step(theta=None)
    get_step(res, propose_with_adam=True, decay_noise=True, propose_only=False)
    start_theta_eval(theta)
    get_theta_eval(res)
    evaluate_theta(theta)
    collect_eval_results(eval_results)
    update_pata_ec(archived_optimizers, optimizers, lower_bound, upper_bound)
    evaluate_transfer(optimizers, evaluate_proposal=True, propose_with_adam=False)
    pick_proposal(checkpointing, reset_optimizer)
    """

    def __init__(self,
                 fiber_pool,
                 fiber_shared,
                 args,
                 make_niche,
                 niche_kwargs,
                 learning_rate,
                 batches_per_chunk,
                 batch_size,
                 eval_batch_size,
                 eval_batches_per_step,
                 l2_coeff,
                 noise_std,
                 lr_decay=1,
                 lr_limit=0.001,
                 noise_decay=1,
                 noise_limit=0.01,
                 normalize_grads_by_noise_std=False,
                 returns_normalization='centered_ranks',
                 optim_id=0,
                 log_file='unname.log',
                 created_at=0,
                 is_candidate=False,
                 nolog=False):
    
        #from .optimizers import Adam, SimpleSGD
        self.nolog = nolog
        logger.debug(f'Creating optimizer {optim_id}...')
        self.fiber_pool = fiber_pool

        self.optim_id = optim_id
        assert self.fiber_pool is not None

        # This randomness controls the seeds sent to batch runs.
        self.random_state = Generator(PCG64DXSM(seed=hash(optim_id) % 2**31 - 1))
    
        self.lr_decay = lr_decay
        self.lr_limit = lr_limit
        self.noise_decay = noise_decay
        self.noise_limit = noise_limit

        self.fiber_shared = fiber_shared
        niches = fiber_shared["niches"]
        niches[optim_id] = make_niche(env_evo_params=niche_kwargs["env_evo_params"],
                            seed=niche_kwargs["seed"],
                            init=niche_kwargs["args"].init,
                            stochastic=niche_kwargs["args"].stochastic,
                            args=niche_kwargs["args"])

        if niche_kwargs["model_params"] is not None:
            self.theta = np.array(niche_kwargs["model_params"])
        else:
            self.theta = niches[optim_id].initial_theta()

        if not self.nolog: logger.debug(f'Optimizer {optim_id} optimizing {len(self.theta)} parameters')
    
        self.batches_per_chunk = batches_per_chunk
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_batches_per_step = eval_batches_per_step
        self.l2_coeff = l2_coeff
        self.noise_std = noise_std
        self.init_noise_std = noise_std

        self.normalize_grads_by_noise_std = normalize_grads_by_noise_std
        self.returns_normalization = returns_normalization

        if is_candidate == False:
            log_fields = [
                f'po_returns_mean_{optim_id}',
                f'po_returns_median_{optim_id}',
                f'po_returns_std_{optim_id}',
                f'po_returns_max_{optim_id}',
                f'po_returns_min_{optim_id}',
                f'po_len_mean_{optim_id}',
                f'po_len_std_{optim_id}',
                f'noise_std_{optim_id}',
                f'learning_rate_{optim_id}',
                f'eval_returns_mean_{optim_id}',
                f'eval_returns_median_{optim_id}',
                f'eval_returns_std_{optim_id}',
                f'eval_len_mean_{optim_id}',
                f'eval_len_std_{optim_id}',
                f'eval_n_episodes_{optim_id}',
                f'theta_norm_{optim_id}',
                f'grad_norm_{optim_id}',
                f'update_ratio_{optim_id}',
                f'episodes_this_step_{optim_id}',
                f'episodes_so_far_{optim_id}',
                f'timesteps_this_step_{optim_id}',
                f'timesteps_so_far_{optim_id}',
                f'time_elapsed_this_step_{optim_id}',

                f'accept_theta_in_{optim_id}',
                f'eval_returns_mean_best_in_{optim_id}',
                f'eval_returns_mean_best_with_ckpt_in_{optim_id}',
                f'eval_returns_mean_theta_from_others_in_{optim_id}',
                f'eval_returns_mean_proposal_from_others_in_{optim_id}',
            ]
             

            if not self.nolog: 
                log_path = log_file + '/' + log_file.split('/')[-1] + "_" +  args.logtag + '.' + optim_id + '.log'
                self.data_logger = CSVLogger(log_path, log_fields + [
                    'time_elapsed_so_far',
                    'iteration',
                ])
                logger.info(f'Optimizer {optim_id} created!')

        self.filename_best = log_file + '/' + log_file.split('/')[-1] + "_" + args.logtag +  '.' + optim_id + '.best.json'
        self.log_data = {}
        self.t_start = time.time()
        self.episodes_so_far = 0
        self.timesteps_so_far = 0

        self.checkpoint_scores = None

        self.self_evals = None   # Score of current parent theta
        self.proposal = None   # Score of best transfer
        self.proposal_theta = None # Theta of best transfer
        self.proposal_source = None # Source of best transfer

        self.created_at = created_at
        self.start_score = None

        self.best_score = None
        self.best_theta = None
        self.recent_scores = []
        self.transfer_target = None
        self.pata_ec = None

        self.iteration = 0

    def __str__(self):
        return(f"ESOptimizer {self.optim_id}")

#######################################
#
#       Checkpointing
#
#######################################
    def checkpoint(self, check_dir):

        ## Set up directory
        opt_dir = os.path.join(check_dir, self.optim_id)
        os.makedirs(opt_dir, exist_ok=True)
        
        ## Save class parameters
        manifest = self.__dict__.copy()

        KEYS_NOT_SAVED = ["fiber_pool","fiber_shared","data_logger","log_data"]
        for k in KEYS_NOT_SAVED:
            del manifest[k]

        # for k,i in manifest.items():
        #     print(k, type(i))

        
        manifest["random_state"] = self.random_state.bit_generator.state
        # manifest["checkpoint_scores"] = self.checkpoint_scores
        # manifest["best_score"] = self.best_score
        # manifest["best_theta"] = self.best_theta.tolist()
        # manifest["theta"] = self.theta.tolist()

        logger.info(f"self.checkpoint_scores: {self.checkpoint_scores}")
        logger.info(f"self.best_score: {self.best_score}")

        with open(os.path.join(opt_dir, "manifest.json"),'w') as f:
            #for key, item in manifest.items():  
                #print(key)
                #json.dump(item,f,cls=NumpyEncoder)
            
            json.dump(manifest,f,cls=NumpyEncoder)

        ## Checkpoint Niche
        niche = self.fiber_shared["niches"][self.optim_id]
        niche.checkpoint(opt_dir)


        pass

    def reload(self, check_dir):
    
        opt_dir = os.path.join(check_dir, self.optim_id)

        ## Load class parameters
        KEYS_NOT_SAVED = ["fiber_pool","fiber_shared","data_logger","log_data"]

        # create new PRNG, will reset when we load it in.
        self.random_state = Generator(PCG64DXSM())

        with open(os.path.join(opt_dir, "manifest.json"),'r') as f:
            manifest = json.load(f, cls=NumpyDecoder)

            SPECIAL_LOAD = ["random_state"]
            self.random_state.bit_generator.state = manifest["random_state"]

            # self.checkpoint_scores = manifest["checkpoint_scores"]
            # self.best_score = manifest["best_score"]
            # self.best_theta = np.array(manifest["best_theta"])
            # self.theta = np.array(manifest["theta"])

            for k,i in manifest.items():
                if k not in SPECIAL_LOAD:
                    #print(k, type(i))
                    self.__dict__[k] = manifest[k]



        ## Reload Niche
        niche = self.fiber_shared["niches"][self.optim_id]
        niche.reload(opt_dir)

        pass

#######################################
#
#       Utility Functions
#
#######################################

    def __del__(self):
        logger.debug(f'Optimizer {self.optim_id} cleanning up workers...')

    def clean_dicts_before_iter(self):
        self.log_data.clear()
        self.self_evals = None
        self.proposal = None
        self.proposal_theta = None
        self.proposal_source = None

    def save_to_logger(self, iteration):
        self.log_data['time_elapsed_so_far'] = time.time() - self.t_start
        self.log_data['iteration'] = iteration
        self.data_logger.log(**self.log_data)

        logger.debug(f'iter={iteration} Optimizer {self.optim_id} best score {self.best_score}')

        #if iteration % 100 == 0:
        #    self.save_policy(self.filename_best+'.arxiv.'+str(iteration))

        self.save_policy(self.filename_best)

    def save_policy(self, policy_file, reset=False):
        if self.best_score is not None and self.best_theta is not None:
            with open(policy_file, 'wt') as out:
                json.dump([self.best_theta.tolist(), self.best_score], out, sort_keys=True, indent=0, separators=(',', ': '))
            if reset:
                self.best_score = None
                self.best_theta = None


    def update_dicts_after_transfer(self, source_optim_id, source_optim_theta, stats, keyword):
        '''Log eval data, returning true is current score is greater than transer target'''

        eval_key = 'eval_returns_mean_{}_from_others_in_{}'.format(keyword,  # noqa
            self.optim_id)
        if eval_key not in self.log_data.keys():
            self.log_data[eval_key] = source_optim_id + '_' + str(stats.eval_returns_mean)
        else:
            self.log_data[eval_key] += '_' + source_optim_id + '_' + str(stats.eval_returns_mean)

        if keyword == 'proposal' and stats.eval_returns_mean > self.transfer_target:
            if  stats.eval_returns_mean > self.proposal:
                self.proposal = stats.eval_returns_mean
                self.proposal_source = source_optim_id + ('' if keyword=='theta' else "_proposal")
                self.proposal_theta = np.array(source_optim_theta)

        return stats.eval_returns_mean > self.transfer_target

    def update_dicts_after_es(self, stats, self_eval_stats):
        '''Updates statistics, including revent scores, transfer targets, and proposal scores. '''

        self.self_evals = self_eval_stats.eval_returns_mean
        if self.start_score is None:
            self.start_score = self.self_evals
        self.proposal = self_eval_stats.eval_returns_mean
        self.proposal_source = self.optim_id
        self.proposal_theta = np.array(self.theta)

        if self.checkpoint_scores is None:
            self.checkpoint_scores = self_eval_stats.eval_returns_mean

        self.episodes_so_far += stats.episodes_this_step
        self.timesteps_so_far += stats.timesteps_this_step

        if self.best_score is None or self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = np.array(self.theta)

        assert len(self.recent_scores) <= 5
        if len(self.recent_scores) == 5:
            self.recent_scores.pop(0)
        self.recent_scores.append(self.self_evals)
        self.transfer_target = max(self.recent_scores)

        self.log_data.update({
            'po_returns_mean_{}'.format(self.optim_id):
                stats.po_returns_mean,
            'po_returns_median_{}'.format(self.optim_id):
                stats.po_returns_median,
            'po_returns_std_{}'.format(self.optim_id):
                stats.po_returns_std,
            'po_returns_max_{}'.format(self.optim_id):
                stats.po_returns_max,
            'po_returns_min_{}'.format(self.optim_id):
                stats.po_returns_min,
            'po_len_mean_{}'.format(self.optim_id):
                stats.po_len_mean,
            'po_len_std_{}'.format(self.optim_id):
                stats.po_len_std,
            'noise_std_{}'.format(self.optim_id):
                stats.noise_std,
            'learning_rate_{}'.format(self.optim_id):
                stats.learning_rate,
            'eval_returns_mean_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_mean,
            'eval_returns_median_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_median,
            'eval_returns_std_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_std,
            'eval_len_mean_{}'.format(self.optim_id):
                self_eval_stats.eval_len_mean,
            'eval_len_std_{}'.format(self.optim_id):
                self_eval_stats.eval_len_std,
            'eval_n_episodes_{}'.format(self.optim_id):
                self_eval_stats.eval_n_episodes,
            'theta_norm_{}'.format(self.optim_id):
                stats.theta_norm,
            'grad_norm_{}'.format(self.optim_id):
                stats.grad_norm,
            'update_ratio_{}'.format(self.optim_id):
                stats.update_ratio,
            'episodes_this_step_{}'.format(self.optim_id):
                stats.episodes_this_step,
            'episodes_so_far_{}'.format(self.optim_id):
                self.episodes_so_far,
            'timesteps_this_step_{}'.format(self.optim_id):
                stats.timesteps_this_step,
            'timesteps_so_far_{}'.format(self.optim_id):
                self.timesteps_so_far,
            'time_elapsed_this_step_{}'.format(self.optim_id):
                stats.time_elapsed_this_step + self_eval_stats.time_elapsed,
            'accept_theta_in_{}'.format(self.optim_id): 'self'
        })

    # def add_env(self, env):
    #     '''On all worker, add env_name to niche'''
    #     logger.debug('Optimizer {} add env {}...'.format(self.optim_id, env.name))

    #     niches = self.fiber_shared["niches"]
    #     niches[self.optim_id].add_env(env)

    def delete_env(self, env_name):
        '''On all worker, delete env from niche'''
        logger.debug(f'Optimizer {self.optim_id} delete env {env_name}...')

        niches = self.fiber_shared["niches"]
        niches[self.optim_id].delete_env(env_name)

#######################################
#
#       Multithread
#
#######################################

    def broadcast_theta(self, theta):
        '''On all worker, set thetas[this optimizer] to theta'''
        logger.debug(f'Optimizer {self.optim_id} broadcasting theta...')

        thetas = self.fiber_shared["thetas"]
        thetas[self.optim_id] = theta
        self.iteration += 1

    def start_chunk_fiber(self, runner, batches_per_chunk, batch_size, *args):
        '''Start runner function on a number of chunks `batches_per_chunk` each of `batch_size` runs
        
        Parameters
        ----------
        runner: function
            Function to run on pool
        batches_per_chunk: int
            Number of chunks to run
        batch_size: int
            Size of each batch to run
        *args: list
            Additional args to pass to function. 

        Returns
        -------
        chunck_tasks: list of async
            List of async objects. Results of function cna be gotten with asyc.get()
        '''

        logger.debug(f'Optimizer {self.optim_id} spawning {batches_per_chunk} batches of size {batch_size}')

        rs_seeds = self.random_state.integers(low=2**31-1, size=batches_per_chunk, dtype=np.int32)

        chunk_tasks = []
        pool = self.fiber_pool
        niches = self.fiber_shared["niches"]
        thetas = self.fiber_shared["thetas"]


        for i in range(batches_per_chunk):
            chunk_tasks.append(
                pool.apply_async(runner, args=(self.iteration,
                    self.optim_id, batch_size, rs_seeds[i])+args))
        return chunk_tasks

    def get_chunk(self, tasks):
        ''' Get the results of the async functions in tasks
        
        Parameters
        ----------
        tasks: list
            List of async pool functions from start_chunk_fiber

        Returns
        -------
        list
            List of function results
        '''
        return [task.get() for task in tasks]
                                                                           
#######################################
#
#       Optimize
#
#######################################
    def set_theta(self, theta, reset_optimizer=True):
        '''Set theta, reseting optimizer parameters (momentum, etc) is specified.'''

        self.theta = np.array(theta)
        if reset_optimizer:
            niches = self.fiber_shared["niches"]
            niche = niches[self.optim_id]
            niche.reset_optimizers()

    def start_single_step(self, theta=None):
        ''' Starts the optimization jobs on the fiber pool for a chunk of runs with random varaiations on 
        the theta value. 
        
        If theta is not supplied use optimizes theta. Random variations are generated in the run_po_batch_fiber function. 

        Parameters
        ----------
        theta = np.Array
            Theta to use to start evaluation runs

        Returns
        -------
        step_results: list
            List of async obejcts that can be called with async.get() to return the result of the run
        theta: np.Array
            Theta used for the runs
        step_t_start: int
            The time step when the jobs begun running
        '''

        step_t_start = time.time()
        if theta is None:
            theta = self.theta

        step_results = self.start_chunk_fiber(
            run_single_po_batch_fiber,
            self.batches_per_chunk,
            self.batch_size,
            self.noise_std,
            theta)

        return step_results, theta, step_t_start

    def start_step(self, theta=None):
        ''' Starts the optimization jobs on the fiber pool for a chunk of runs with random varaiations on 
        the theta value. 
        
        If theta is not supplied use optimizes theta. Random variations are generated in the run_po_batch_fiber function. 

        Parameters
        ----------
        theta = np.Array
            Theta to use to start evaluation runs

        Returns
        -------
        step_results: list
            List of async obejcts that can be called with async.get() to return the result of the run
        theta: np.Array
            Theta used for the runs
        step_t_start: int
            The time step when the jobs begun running
        '''

        step_t_start = time.time()
        if theta is None:
            theta = self.theta
        self.broadcast_theta(theta)

        step_results = self.start_chunk_fiber(
            run_po_batch_fiber,
            self.batches_per_chunk,
            self.batch_size,
            self.noise_std)

        return step_results, theta, step_t_start



    def get_step(self, res, propose_with_adam=True, decay_noise=True, propose_only=False):
        '''Gets and processess the results of the async runs of the environment.

        First, get results of the aync runs of agent through environment. Then collect the final
        scores into po_returns. Compute the gradiants based on the results and pass them off to the
        selected optimization function to adjut the gradient for update. 
        
        Parameters
        ---------- 
        res: list
            list of async functions that return nametuple POResult, theta, step_t_start

        propose_with_adam: boolean
            Only make a proposal of the new thetas and use Adam to do so

        decay_noise: boolean
            Decay the noise function

        propose_only: boolean
            Do not update theta values, propose new thetas only
        
        Returns
        -------
        theta: np.Array
            Computed theta values

        namedtuple StepStats       
            Name tuple containing statistics about the step 
		'''
        niches = self.fiber_shared["niches"]
        niche = niches[self.optim_id]

        step_tasks, theta, step_t_start = res
        step_results = self.get_chunk(step_tasks)

        theta, stepstats = niche.combine_steps(step_results, theta, 
                step_t_start, 
                propose_with_adam=True, 
                decay_noise=True, 
                propose_only=False)

        return(theta, stepstats)

#######################################
#
#       Evaluate
#
#######################################
                                        
    def start_theta_eval(self, theta):
        '''Start theta evaluation in this optimizer's niche
        
        Parameters
        ----------
        theta: np.Array
            Array of theta parameters
        '''
        step_t_start = time.time()
        #self.broadcast_theta(theta)

        eval_tasks = self.start_chunk_fiber(
            run_eval_batch_fiber, self.eval_batches_per_step, self.eval_batch_size, theta)

        return eval_tasks, theta, step_t_start

    def get_theta_eval(self, res):
        '''Get the result of a theta evaualtion, record them, and return the stats on the run. 

        Parameters
        ----------
        res: list
            List of [eval_tasks, theta, step_t_start] produced by start_theta_eval
        
        Returns
        -------
        namedtuple EvalStats
            Named tuple containing statistics of runs.
        '''

        eval_tasks, theta, step_t_start = res
        eval_results = self.get_chunk(eval_tasks)
        eval_returns, eval_lengths, eval_rollouts = self.collect_eval_results(eval_results)
        step_t_end = time.time()

        logger.debug(f'get_theta_eval {self.optim_id} finished running {len(eval_returns)}'
                     f' episodes, {eval_lengths.sum()} timesteps')

        return EvalStats(
            eval_returns_mean=eval_returns.mean(),
            eval_returns_median=np.median(eval_returns),
            eval_returns_std=eval_returns.std(),
            eval_len_mean=eval_lengths.mean(),
            eval_len_std=eval_lengths.std(),
            eval_n_episodes=len(eval_returns),
            time_elapsed=step_t_end - step_t_start,
            eval_rollouts=eval_rollouts
        )

    def evaluate_theta(self, theta):
        '''Evaluate theta in multithreaded way, returning the mean reward.
        
        Parameters
        ----------
        theta: np.Array
            Theta to be evalauted
        
        Returns
        -------
        float
            Mean return
        '''
        self_eval_task = self.start_theta_eval(theta)
        self_eval_stats = self.get_theta_eval(self_eval_task)
        return self_eval_stats.eval_returns_mean

    def collect_eval_results(self, eval_results):
        '''Extracts run results from EvalResult object
        
        Parameters
        ----------
        eval_results: EvalResult
            A EvalResult named tuple. EvalResult objects are created by run_eval_batch_fiber
        
        Returns
        -------
        returns: np.Array
            Array with two scores, one for plus noise one for minus scores
        lengths: np.Array
            Array of run lengths
        '''

        eval_returns = np.concatenate([r.returns for r in eval_results])
        eval_lengths = np.concatenate([r.lengths for r in eval_results])
        try:
            eval_rollouts = np.concatenate([r.rollouts for r in eval_results])
        except ValueError as e:
            eval_rollouts = None
        return eval_returns, eval_lengths, eval_rollouts

    def update_pata_ec(self, archived_optimizers, optimizers, lower_bound, upper_bound):
        '''Evaluate your theta against all other optimizers, both archived and active. Compute the centered
        rank scores and store them in self.pata_ec

        Parameters
        ----------
        archived_optimizers: dict
            Dictionary of ESOptimizers
        optimizers: dict
            Dictionary of ESOptimizers
        lower_bound: int
            Lower bound cutoff for scores
        upper_bound: int
            Upper bound cutoff for scores
        '''
        
        def cap_score(score, lower, upper):
            if score < lower:
                score = lower
            elif score > upper:
                score = upper

            return score

        # setup return objects
        capped_scores = []

        # loop over archived optimizers
        for source_optim in archived_optimizers.values():
            # cap and store
            
            capped_scores.append(cap_score(self.evaluate_theta(source_optim.theta), lower_bound, upper_bound))

        # loop over active optimizers
        for source_id, source_optim in optimizers.items():
            # cap and store
            capped_scores.append(cap_score(self.evaluate_theta(source_optim.theta), lower_bound, upper_bound))

        # update 
        self.pata_ec = compute_CS_ranks(np.array(capped_scores))


#######################################
#
#       Evolve
#
#######################################
    def get_mutated_params(self):
        niche = self.fiber_shared["niches"][self.optim_id]
        return(niche.get_mutated_params())

    def get_env_params(self):
        niche = self.fiber_shared["niches"][self.optim_id]
        return(niche.env_evo_params)

#######################################
#
#       Transfer
#
#######################################

    def evaluate_transfer(self, optimizers, evaluate_proposal=True, propose_with_adam=False):
        '''Evaluate all theta on this optimizer, recording best initial theta. If evaluate_proposal, 
        perform one optimization step and evaluate again. Return the best score and theta.

        Parameters
        ----------
        optimizers: dict
            Disctionary of ESOptimizer obejects
        evaluate_proposal: boolean
            If true, perform one optimization step and evaluate
        propose_with_adam:
            Use adam (ie momentum based optimization) to perform update step

        Returns
        -------
        best_init_score: float
            Best score of all thetas out of all runs
        best_init_theta: np.Array
            Theta value from best run
        '''
        
        best_init_score = 0
        best_init_theta = None

        eval_tasks = {}
        train_tasks = {}
        for source_optim in optimizers.values():
            eval_tasks[source_optim] = self.start_theta_eval(source_optim.theta)

            if evaluate_proposal:
                train_tasks[source_optim] = self.start_single_step(source_optim.theta)
        
        ## Get the one step train tasks and start their eval
        eval_prop_tasks = {}
        if evaluate_proposal:
            for source_optim in optimizers.values():
                proposed_theta, _ = self.get_step(
                    train_tasks[source_optim], propose_with_adam=propose_with_adam, propose_only=True)

                eval_prop_tasks[source_optim] = self.start_theta_eval(proposed_theta)
            
        
        for source_optim in optimizers.values():
            score = self.get_theta_eval(eval_tasks[source_optim]).eval_returns_mean

            if score > best_init_score:
                #print("es.ESOptimizer.evaluate_transfer:: I got a new best score: ", score)
                best_init_score = score
                best_init_theta = np.array(source_optim.theta)

            if evaluate_proposal:
                # task = self.start_step(source_optim.theta)
                # #icm['combine_rewards']=False
                # proposed_theta, _ = self.get_step(
                #     task, propose_with_adam=propose_with_adam, propose_only=True)
                #icm['combine_rewards']=True
                # score = self.evaluate_theta(proposed_theta)
                score = self.get_theta_eval(eval_prop_tasks[source_optim]).eval_returns_mean

                if score > best_init_score:
                    best_init_score = score
                    best_init_theta = np.array(source_optim.theta)

        return best_init_score, best_init_theta

        

    def pick_proposal(self, checkpointing, reset_optimizer):
        '''Compare proposal against previus checkpoints, taking the highest score. Copy all
        statistics from accepted propasal (best_score, self_evals, checkpoints, etc) into 
        optimizer variables.'''

        accept_key = f'accept_theta_in_{self.optim_id}'
        if checkpointing and self.checkpoint_scores > self.proposal:
            self.log_data[accept_key] = 'do_not_consider_CP'
        else:
            self.log_data[accept_key] = f'{self.proposal_source}'
            if self.optim_id != self.proposal_source:
                self.set_theta(
                    self.proposal_theta,
                    reset_optimizer=reset_optimizer)
                self.self_evals = self.proposal

        self.checkpoint_scores = self.self_evals

        if self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = np.array(self.theta)
