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


from genericpath import isdir
from .logger import CSVLogger
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm

import numpy as np
from numpy.random import Generator, PCG64DXSM
import random
from cpoet.poet_distributed.es import ESOptimizer
from cpoet.poet_distributed.es import initialize_worker_fiber
from collections import OrderedDict
from cpoet.poet_distributed.novelty import pata_ec
import json
import os
import time
import importlib
import cpoet.poet_distributed.niches.ES_Bipedal.cppn as cppn
import matplotlib.pyplot as plt

###############################################################################
##
## Main Class
##
###############################################################################
# This is the main POET class, which maintains model/environment pairs, counts 
#  statistics, handles reproduction/transfers/logging. Model/env. optimization 
#  is handled within the niche, as it may be model-structure dependent. 
class PopulationManager:
    def __init__(self, args):
        ## Dynamically load the niche globally within the module
        self.start_time = time.time()

        import sys
        tmp_mod = importlib.import_module("poet_distributed.niches." + args.niche)
        module = sys.modules[__name__]
        logger.info("Training Niche " + args.niche)
        setattr(module, "Niche", tmp_mod.Niche)

        Niche.class_setup(args)

        self.args = args

        # import fiber as mp
        import multiprocessing as mp
        import fiber.config as fiber_config

        fiber_config.backend = "local"

        self.np_random = Generator(PCG64DXSM(seed=args.master_seed))

        mp_ctx = mp.get_context('spawn')
        manager = mp_ctx.Manager()
        self.manager = manager
        self.fiber_shared = {
                "niches": manager.dict(),
                "thetas": manager.dict(),
        }
        self.fiber_pool = mp_ctx.Pool(args.num_workers, initializer=initialize_worker_fiber,
                initargs=(self.fiber_shared["thetas"],
                    self.fiber_shared["niches"]))

        self.ANNECS = 0
        #self.env_registry = OrderedDict()
        #self.env_archive = OrderedDict()
        #self.env_reproducer = Reproducer(args)
        self.optimizers = OrderedDict()
        self.archived_optimizers = OrderedDict()

        # TODO We many want to expose this to the commandline
        self.novelty = pata_ec(args)

        # generate new environment parameters
        params = Niche.new_parameters(args.master_seed)

        # create first optimizer - an agent/environment pair
        self.add_optimizer(env_params=params, seed=args.master_seed)

        # check if we add other/more optimizers
        # Make sure we don't start with more than the maximum number of optimizers
        #  decrement by 1 because we already created 1 optimizer
        nStartOptims = args.max_num_envs if (args.max_num_envs < args.num_start_envs) else args.num_start_envs
        nStartOptims -= 1

        # check if we need to make more optimizers
        if nStartOptims > 1:
            # check which type of environments we want to start
            if args.start_envs_type == 'randAgent':
                # Flat environments with random agents
                # create new optimizers
                for i in range(nStartOptims):
                    # Create and add new optimizer
                    self.add_optimizer(env_params=params,
                                       seed=self.np_random.integers(2**31 - 1), 
                                       optim_id=f'{params.get_name()}_{i}')

            if args.start_envs_type == 'randEnv':
                # Mutated environments with random agents
                #  There are no checks here, so no guarantee that the environments 
                #  fall within MCC. Does that matter?
                # grab first environment
                originalEnv = next(iter(self.optimizers.values()))
                # create new optimizers
                for _ in range(nStartOptims):
                    # create and add new optimizer
                    self.add_optimizer(env_params=originalEnv.get_mutated_params(),
                                       seed=self.np_random.integers(2**31 - 1))


#####################################
##
## Create/Add Optimizers
##
#####################################

    def create_optimizer(self, env_params, seed, optim_id=None, created_at=0, model_params=None, is_candidate=False, nolog=False):
        '''Creates and returns an ESOptimizer object. This object holds a function that generates niches
        with given initial parameters specified in env and theta given by model_params. Each optimizer
        holds the parameters for a single agent
        
        Parameters
        ---------- 
        env_params: niches.box2d.cppn.CppnEnvParam()
            The cppn paramater object to generate the bipedal walker environment
        seed: int
            Random seed for the env.seed parameter
        created_at: int
            Step optimizer created at
        model_params: np.array
            Array of model parameters, ie thetas
        is_candidate:
            (I think...) Indicates that we're actaully recording this optimizer, canidates may be 
            discarded

        Returns
        -------
        es.ESOptimizer 
            Optimizer object for a single agent

        '''

        # # assert env != None
        # # Why is this here?
        # assert env_params != None

        # Get for labeling and check
        optim_id = env_params.get_name() if (optim_id is None) else optim_id
        # optim_id = env_params.get_name()
        

        # # why is this here?
        # assert optim_id not in self.optimizers.keys()

        # for ESOptimizer
        niche_args = {"env_evo_params":env_params,
                      "model_params":model_params,
                      "seed":seed,
                      "args":self.args}

        return ESOptimizer(
            optim_id=optim_id,
            fiber_pool=self.fiber_pool,
            fiber_shared=self.fiber_shared,
            args = self.args,
            make_niche=Niche,
            niche_kwargs=niche_args,
            learning_rate=self.args.learning_rate,
            lr_decay=self.args.lr_decay,
            lr_limit=self.args.lr_limit,
            batches_per_chunk=self.args.batches_per_chunk,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            eval_batches_per_step=self.args.eval_batches_per_step,
            l2_coeff=self.args.l2_coeff,
            noise_std=self.args.noise_std,
            noise_decay=self.args.noise_decay,
            normalize_grads_by_noise_std=self.args.normalize_grads_by_noise_std,
            returns_normalization=self.args.returns_normalization,
            noise_limit=self.args.noise_limit,
            log_file=self.args.log_file,
            created_at=created_at,
            is_candidate=is_candidate,
            nolog=nolog)


    def add_optimizer(self, env_params, seed, optim_id=None, created_at=0, model_params=None, archive=False):
        '''Creates and adds an optimizer to the set of optimiziers, adds the assocated 
        environment parameters to the active registry and the archive. 
        
        Parameters
        ---------- 
        env_params: niches.box2d.cppn.CppnEnvParam()
            The cppn paramater object to generate the bipedal walker environment
        seed: int
            Random seed for the env.seed parameter
        created_at: int
            Step optimizer created at
        model_params: np.array
            Array of model parameters, ie thetas
        '''

        o = self.create_optimizer(env_params=env_params, seed=seed, optim_id=optim_id,
                                  created_at=created_at, model_params=model_params)
        optim_id = o.optim_id

        assert optim_id not in self.optimizers.keys()
        assert optim_id not in self.archived_optimizers.keys()

        if not archive:
            self.optimizers[optim_id] = o
        else:
            self.archived_optimizers[optim_id] = o
        
        if self.args.save_env_params:
            env_params.save(self.args.log_file)







#####################################
##
## Checkpointing
##
#####################################
        
    def checkpoint(self, name = None):
        if name is None:
            name = str(time.time())

        check_dir = os.path.join(self.args.log_file, self.args.logtag + "cp-" + name)
        if os.path.isdir(check_dir):
            logger.info(f"Directory already exists at {check_dir}. Overwriting checkpoint.")
        else:
            logger.info(f"Saving checkpoint to {check_dir}.")

        os.makedirs(check_dir, exist_ok=True)

        ## Create JSON manifest
        manifest = {}
        manifest["random"] = random.getstate()
        manifest["np_random"] = self.np_random.bit_generator.state
        manifest["optimizers"] = [o for o in self.optimizers]
        manifest["archived_optimizers"] = [o for o in self.archived_optimizers]

        with open(os.path.join(check_dir, "manifest.json"),'w') as f:
            json.dump(manifest,f)

        with open(os.path.join(check_dir, "args.json"),"w") as f:
            json.dump(self.args.__dict__, f)

        ## Checkpoint novelty
        self.novelty.checkpoint(check_dir)

        ## Checkpoint Optimizers
        optimizer_index = {"optimizers": [k for k in self.optimizers.keys()],
                           "archived_optimizers": [k for k in self.archived_optimizers.keys()]}

        with open(os.path.join(check_dir, "optimizer_list.json"), 'w') as f:
            json.dump(optimizer_index, f)

        for k, o in self.optimizers.items():
            logger.info(f"Checkpointing {k}")
            o.checkpoint(check_dir)
            
        for k, o in self.archived_optimizers.items():
            o.checkpoint(check_dir)

        return(check_dir)


    def reload(self, check_dir):
        # Notes for reference
        #  https://stackoverflow.com/questions/63081108/how-can-i-store-and-restore-random-state-in-numpy-random-generator-instances

        # create objects that will be set/reset within open():
        rstate = 0
        self.np_random = Generator(PCG64DXSM())

        with open(os.path.join(check_dir, "manifest.json"),'r') as f:
            manifest = json.load(f)
            rstate = manifest["random"]
            rstate[1] = tuple(rstate[1])
            random.setstate(tuple(rstate))
            self.np_random.bit_generator.state = manifest["np_random"]
        
        ## Reload Novelty
        self.novelty.reload(check_dir)

        ## Reload optimizers
        del self.optimizers
        self.optimizers = OrderedDict()

        with open(os.path.join(check_dir, "optimizer_list.json"), 'r') as f:
            opt_idx = json.load(f)

        #env_dir = "/".join(check_dir.split("/")[:-1] + ["saved_envs"])


        for o in opt_idx["optimizers"]:
            params = Niche.new_parameters(self.args.master_seed)
            o_fol = os.path.join(check_dir, o)
            params.reload(o_fol)

            self.add_optimizer(env_params=params, seed=self.args.master_seed, optim_id=o)
            self.optimizers[o].reload(check_dir)

        for o in opt_idx["archived_optimizers"]:
            params = Niche.new_parameters(self.args.master_seed)
            o_fol = os.path.join(check_dir, o)
            params.reload(o_fol)

            self.add_optimizer(env_params=params, seed=self.args.master_seed, optim_id=o, archive=True)
            self.archived_optimizers[o].reload(check_dir)

        #logger.info(f"RANDOM TEST {random.random()}")
        # for k, o in self.archived_optimizers.items():
        #     o.reload(check_dir)
        

        random.setstate(tuple(rstate))


#####################################
##
## Optimize Function
##
#####################################

    def ind_es_step(self, iteration):
        '''Performs a step of runthrough and optimization

        Starts by loading the run_po_batch_fiber function onto the fiber pool. Then loops over the 
        resulting async objects and calles optimizer.get_step to compute the new theta. 

        We then evaluate the new theta and update the statistics tracker. 
        
        Parameters
        ---------- 
        iteration: int 
            Current itteration
		'''
        
        tasks = [o.start_step() for o in self.optimizers.values()]

        for optimizer, task in zip(self.optimizers.values(), tasks):
            #icm['combine_rewards']=True
            optimizer.theta, stats = optimizer.get_step(task)
            #icm['combine_rewards']=False
            self_eval_task = optimizer.start_theta_eval(optimizer.theta)
            self_eval_stats = optimizer.get_theta_eval(self_eval_task)

            logger.info(f'Iter={iteration} Optimizer {optimizer.optim_id} ' 
                        f'theta_mean {self_eval_stats.eval_returns_mean} best po {stats.po_returns_max} '
                        f'iteration spent {iteration - optimizer.created_at}')

            optimizer.update_dicts_after_es(stats=stats,
                self_eval_stats=self_eval_stats)


#####################################
##
## Transfer Agents Between Environments
##
#####################################

    def transfer(self, propose_with_adam, checkpointing, reset_optimizer):
        '''
        For each optimizer, start async perform an eval run of your theta values on each other env. Log the
        mean score with update_dicts_after_transfer, returning true if the mean score is greater than the 
        recent scores of the optimizer, flaggin it as a proposal. 
        
        For each propasal, score the source theta on the target again, storing the result as the targets
        propsal if the resulting score is greater than the last proposal. This ends up with the best proposal
        score being selected to copy theta over. All of these transactionsa are logged. 

        Finally, update all optimizers internal statistics with those of the new theta.
        
        Parameters
        ----------
        propose_with_adam: boolean
            Only make a proposal of the new thetas and use Adam to do so
        checkpointing: boolean
            Checkpoint 
        reset_optimizer: boolean
        '''
        
        logger.info('Computing direct transfers...')
        proposal_targets = {}

        tasks = {}
        for source_optim in self.optimizers.values():
            source_tasks = []
            proposal_targets[source_optim] = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                task = target_optim.start_theta_eval(
                    source_optim.theta)
                source_tasks.append((task, target_optim))

            tasks[source_optim] = source_tasks

        for source_optim in self.optimizers.values():
            for task, target_optim in tasks[source_optim]:
                stats = target_optim.get_theta_eval(task)

                try_proposal = target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                    source_optim_theta=source_optim.theta,
                                    stats=stats, keyword='theta')
                if try_proposal:
                    proposal_targets[source_optim].append(target_optim)


        logger.info('Computing proposal transfers...')
        ## The code below has been replaced with paralellized code. However, 
        ## the parallelized code places all evalaution jobs in the fiber queue
        ## before adding the optimization jobs, resulting in a different ordering
        ## or calls to each optimizers random number generator, resulting in 
        ## different evaluation runs than the POET v1 code. If you want to
        ## restore the origional order for compairision uncomment this code. 
        
        # for source_optim in self.optimizers.values():
        #     source_tasks = []
        #     for target_optim in [o for o in self.optimizers.values()
        #                             if o is not source_optim]:
        #         if target_optim in proposal_targets[source_optim]:
        #             task = target_optim.start_step(source_optim.theta)
        #             source_tasks.append((task, target_optim))

        #     for task, target_optim in source_tasks:
        #         proposed_theta, _ = target_optim.get_step(
        #             task, propose_with_adam=propose_with_adam, propose_only=True)

        #         proposal_eval_task = target_optim.start_theta_eval(proposed_theta)
        #         proposal_eval_stats = target_optim.get_theta_eval(proposal_eval_task)

        #         target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
        #             source_optim_theta=proposed_theta,
        #             stats=proposal_eval_stats, keyword='proposal')


        tasks = {}
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                if target_optim in proposal_targets[source_optim]:
                    task = target_optim.start_single_step(source_optim.theta)
                    source_tasks.append((task, target_optim))
            
            tasks[source_optim] = source_tasks



        for source_optim in self.optimizers.values():
            for task, target_optim in tasks[source_optim]:
                proposed_theta, _ = target_optim.get_step(
                    task, propose_with_adam=propose_with_adam, propose_only=True)

                proposal_eval_task = target_optim.start_theta_eval(proposed_theta)
                proposal_eval_stats = target_optim.get_theta_eval(proposal_eval_task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                    source_optim_theta=proposed_theta,
                    stats=proposal_eval_stats, keyword='proposal')

        logger.info('Considering transfers...')
        for o in self.optimizers.values():
            o.pick_proposal(checkpointing, reset_optimizer)

#####################################
##
## Environmnet Reproduction
##
#####################################
# Function are in the order that they are called in adjust_envs_niches
# Any other aux functions are directly above the function that calls them.

    def check_optimizer_status(self):
        '''Check optimizers and return list of candidates for reproduction

        Returns
        -------
        repro_candidates: list
            IDs of optimizers to that are candidates for reproduction
		'''

        # initialize loop objects
        repro_candidates = []
        repro_thresh = self.args.repro_threshold

        # Loop through all active niches
        #  This always loops in the same order - orderedDict()
        for optim_id, optim in self.optimizers.items():
            # log
            logger.info(f"niche {optim_id} created at {optim.created_at} "
                        f"start_score {optim.start_score} current_self_evals {optim.self_evals}")
            # check if ready to reproduce
            if optim.self_evals >= repro_thresh:
                repro_candidates.append(optim_id)

        return repro_candidates


    def write_pata_ec(self, optim, csvfile):
        #print("Pata_EC: ", np.array(optim.pata_ec).shape)
        if len(np.array(optim.pata_ec).shape)>0:
            #print("Pata_EC", optim.pata_ec)
            x_str = str(optim.optim_id) + ", " + ",".join(np.char.mod('%f', optim.pata_ec)) + "\n"
            csvfile.write(x_str)


    def get_new_env(self, list_repro):
        '''Uses the environment reproducer in self.reproducer to pick a parent env, and then clone it, 
        producing a mutated cppn and a mutated enviroment parameter set. 
        
        Parameters
        ----------
        list_repro: list
            List of IDs for possible parent optimizers
        
        Return
        ------
        child_env_params: CppnEnvParams
            Inherited, mutated CppnEnvParams from parent
        seed: int
            Random seed
        optim_id: int
            Optimizer id
        '''

        # Grab ID of parent environment
        optim_id = self.np_random.choice(a=list_repro, shuffle=False)

        # grab parent optimizer from dictionary by ID
        optim_parent = self.optimizers[optim_id]

        # generate child environment
        child_env_params = optim_parent.get_mutated_params()
        # generate seed for child
        child_env_seed = self.np_random.integers(2**31 - 1)

        # Log info
        logger.info(f"we pick to mutate: {optim_id} and we got {child_env_params.get_name()} back")

        logger.debug("parent")
        logger.debug(optim_parent.get_env_params())
        logger.debug("child")
        logger.debug(child_env_params)

        # return new env, seed, and parent ID
        return child_env_params, child_env_seed, optim_id


    def get_child_list(self, parent_list, max_children, iteration):
        '''Returns a list of random mutations from the parent_list, including their novelty scores. 

        Parameters
        ----------
        parent_list: list
            List of IDs of ESOptimizers
        max_children: int
            Maximium number of potential children to attempt - these attempts may not pass MCC checks

        Returns
        -------
        child_list: list
            List of (env_params, seed, parent_id, novelty_score) for new child optimizers. 
        '''

        self.novelty.update_novelty(self.archived_optimizers, self.optimizers, iteration)

        # setup return list to hold viable children
        #  These are defined as the potential children than pass MC checks.
        child_list = []
        potential_children = {}

        mutation_trial = 0
        while mutation_trial < max_children:
            new_env_params, seed, parent_optim_id = self.get_new_env(parent_list)
            mutation_trial += 1
            # check for duplicate naming
            if new_env_params.get_name() in self.optimizers.keys():
                logger.debug("active env already. reject!")
            else: 
                o = self.create_optimizer(new_env_params, seed, is_candidate=True)
                score = o.evaluate_theta(self.optimizers[parent_optim_id].theta)


                if self.pass_mc(score):
                    potential_children[o] = (new_env_params, seed, parent_optim_id, score)

                    # novelty_score = self.compute_novelty(
                    #                     archived_optimizers=self.archived_optimizers,
                    #                     optimizers=self.optimizers, opt=o, k=5,
                    #                     low=self.args.mc_lower, high=self.args.mc_upper)



                    # logger.debug(f"{score} passed mc, novelty score {novelty_score}")
                    # child_list.append((new_env_params, seed, parent_optim_id, novelty_score))
                else:
                    del o

        # log if we're returning no children
        if len(potential_children.keys()) == 0:
            logger.info("mutation to reproduce env FAILED!!!")
        else:
            ## Compute the novelty scores:
            novelties = self.novelty.novelty(self.archived_optimizers, 
                                             self.optimizers,
                                             list(potential_children.keys()))

            for o, v in potential_children.items():
                novelty_score = novelties[o]
                score = v[3]
                logger.debug(f"{score} passed mc, novelty score {novelty_score}")

                child_list.append(v[:3] + (novelty_score,))
        
        #sort child list according to novelty for high to low
        child_list = sorted(child_list,key=lambda x: x[3], reverse=True)

        # return child list
        return child_list

    def pass_mc(self, score):
        '''Check if score in MCC range. 

        Parameters
        ----------
        score:
            double or None

        Returns
        -------
        boolean:
            Whether or not the score passed
        '''

        # safety if None
        if score is None:
            return False

        # return checks
        #  1) check that score is above lower
        #  2) check that score is below upper
        return (self.args.mc_lower <= score) and (score <= self.args.mc_upper)



    def archive_optimizer(self, optim_id):
        '''
        Removes the optimizer and env parameters from the active regestries.
        Updates ANNECS if appropriate.
        
        Parameters
        ---------- 
        optim_id: int 
            The id of the optimizer and env parameters to archive

        Returns
        -------
        None
		'''

        # Get optimizer by ID, and remove from active list at the same time
        o = self.optimizers.pop(optim_id)

        # add optimizer to archive by ID
        self.archived_optimizers[optim_id] = o

        # check for ANNECS
        aBool = o.transfer_target > self.args.repro_threshold
        if aBool:
            self.ANNECS += 1

        # log action
        logger.info(f'Archived {optim_id} - added to ANNECS: {aBool}')

        # no return


    def remove_oldest(self, num_removals):
        '''
        Grabs oldest optimizers from dictionary, passes them to archive_optimizer 
        for removal and archiving.

        This will fail if num_removals begins less than 0.
        
        Parameters
        ---------- 
        num_removals: int 
            Number of optimizers/environments to archive.

        Returns
        -------
        None
		'''

        # setup list to pass down
        list_delete = []

        # loop through active optimizers
        for optim_id in self.optimizers.keys():
            # check if there are more to archive
            #  else break
            if num_removals:
                # put ID in list and decrement number to remove
                list_delete.append(optim_id)
                num_removals-=1
            else:
                break

        # loop through optimizers to delete and archive them.
        for optim_id in list_delete:
            self.archive_optimizer(optim_id)  

        # no return        


    def adjust_envs_niches(self, iteration, max_num_envs=8, max_children=8, max_admitted=1):
        '''Try to evolve new environmental niches from old ones.

        Check if it's time to evolve. If so, get list of candidate optimizers for reproduction. Score each optimizer
        on each env, clip and rank the scores. Based on those scores, get a list of children up to max_children,
        loop through the potetntial children evaluating each until we find max_admitted suitable candidates

        Parameters
        ----------
        iteration: int
            current global iteration number
        max_num_envs: int
            Maximum number of environments to keep active, older envs will be archived
        max_children: int
            Maximum number of mutations to attempt, attempted mutations may not pass mc
        max_admitted: int
            How many mutations to keep
        '''
        logger.info("poet_algo.optimize:: ################# Env Evolution Step ###############")
        list_repro = self.check_optimizer_status()

        if len(list_repro) == 0:
            logger.info("no suitable niches to reproduce")
            return

        logger.info("list of niches to reproduce")
        logger.info(list_repro)

        # get list of potential children
        #  These children have passed initial MCC check, and are ranked by novelty
        child_list = self.get_child_list(list_repro, max_children, iteration)

        admitted = 0
        for child in child_list:
            new_env_params, seed, _, _ = child
            # targeted transfer
            o = self.create_optimizer(new_env_params, seed, is_candidate=True)
            score_child, theta_child = o.evaluate_transfer(optimizers=self.optimizers)
            del o


            # Check MCC
            if self.pass_mc(score_child):
                # Pass checks!
                #  add to current population  
                self.add_optimizer(env_params=new_env_params, seed=seed,
                                   created_at=iteration, model_params=theta_child)
                # increment number admitted
                admitted += 1
                # break loop if finished
                if admitted >= max_admitted:
                    break



        if len(self.optimizers) > max_num_envs:
            num_removals = len(self.optimizers) - max_num_envs
            self.remove_oldest(num_removals)



#####################################
##
## Main Loop
##
#####################################

    def evolve_population(self, iterations=200,
                        steps_before_transfer=25,
                        propose_with_adam=False,
                        checkpointing=False,
                        reset_optimizer=True,
                        start_iteration=0):

        '''Main optimization loop. 

        For each step of the iteration evolves the env's, optimizing the fucntions, and evalutes the
        transfer potential.

        Parameters
        ---------- 
        iterations: int
            Number of iterations to run
        steps_before_transfer: int
            Steps before transfer is attempted
        propose_with_adam: boolean
            Makes the transfer step propose a theta update instead of actually commiting to one?
        reset_optimizer
            Resets momentum or other parameters in the optimizer
        '''

        # Loop Constants
        nMut = self.args.adjust_interval * steps_before_transfer
        # Main Loop
        #  Algorithm 2 in POET paper (https://doi.org/10.48550/arXiv.1901.01753)
        for iteration in range(start_iteration, iterations):
            #logger.info(f"Random Test: {np.random.random()} {random.random()}")
            
            # Environment Evolution Step
            Niche.iteration_update(iteration)

            # Reproduction Step
            if (iteration > 0) and (iteration % nMut == 0):
                self.adjust_envs_niches(iteration=iteration, max_num_envs=self.args.max_num_envs)

            for o in self.optimizers.values():
                o.clean_dicts_before_iter()

            # Optimization Step
            self.ind_es_step(iteration=iteration)

            # Transfer Step
            if len(self.optimizers) > 1 and iteration % steps_before_transfer == 0:
                logger.info("--- %s seconds --- ################# Transfer Step ###############" % (time.time() - self.start_time))
                self.transfer( 
                    propose_with_adam=propose_with_adam,
                    checkpointing=checkpointing,
                    reset_optimizer=reset_optimizer)

            if iteration > 0 and iteration % steps_before_transfer == 0:
                check_dir = self.checkpoint(str(iteration))

                #self.reload(check_dir)
                for o in self.optimizers.values():
                    o.save_to_logger(iteration)

