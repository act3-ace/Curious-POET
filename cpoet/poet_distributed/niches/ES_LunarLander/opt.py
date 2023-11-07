import numpy as np
from ... import es
from ...es import StepStats
import logging 
from .optimizers import Adam, SGD
logger = logging.getLogger(__name__)
import time
from ...stats import compute_weighted_sum, compute_CS_ranks
import json
import os

class OptimizationControler:
    def __init__(self, args, model):
        self.l2_coeff = args.l2_coeff
        self.lr_decay = args.lr_decay
        self.noise_decay = args.noise_decay
        self.lr_limit = args.lr_limit
        self.noise_limit = args.noise_limit
        self.noise_std = args.noise_std
        self.init_noise_std = self.noise_std
        self.optimizer = Adam(model.get_zeroed_model_params(), args.learning_rate)
        self.sgd_optimizer = SGD(model.get_zeroed_model_params(), args.learning_rate)

        self.returns_normalization = args.returns_normalization
        self.normalize_grads_by_noise_std=args.normalize_grads_by_noise_std

    def __str__(self) -> str:
        return("Optimization Controler")

    def checkpoint(self, folder):
        json_dict = self.__dict__.copy()
        del json_dict["optimizer"]
        del json_dict["sgd_optimizer"]

        with open(os.path.join(folder, "_opt.json"), 'w') as f:
            json.dump(json_dict, f)

        self.optimizer.checkpoint(folder)
        self.sgd_optimizer.checkpoint(folder)

    def reload(self, folder):
        with open(os.path.join(folder, "_opt.json"), 'r') as f:
            json_dict = json.load(f)
            for k in json_dict.keys():
                self.__dict__[k] = json_dict[k]
        
        self.optimizer.reload(folder)
        self.sgd_optimizer.reload(folder)

    def combine_steps(self, step_results, theta, step_t_start, propose_with_adam=True, decay_noise=True, propose_only=False):
        #po_noise_inds = np.concatenate([r.noise_inds for r in step_results])
        po_returns = np.concatenate([r.returns for r in step_results])
        po_lengths = np.concatenate([r.lengths for r in step_results])
        
        episodes_this_step = len(po_returns)
        timesteps_this_step = po_lengths.sum()

        #logger.debug(
        #    'Optimizer {} finished running {} episodes, {} timesteps'.format(
        #        self.optim_id, episodes_this_step, timesteps_this_step))

        #### This needs to be placed in the nitch
        

        grads, po_theta_max = self.compute_grads(step_results, theta)
        if not propose_only:
            update_ratio, theta = self.optimizer.update(
                theta, -grads + self.l2_coeff * theta)

            self.optimizer.stepsize = max(
                self.optimizer.stepsize * self.lr_decay, self.lr_limit)
            if decay_noise:
                self.noise_std = max(
                    self.noise_std * self.noise_decay, self.noise_limit)

        else:  #only make proposal
            if propose_with_adam:
                update_ratio, theta = self.optimizer.propose(
                    theta, -grads + self.l2_coeff * theta)
            else:
                update_ratio, theta = self.sgd_optimizer.compute(
                    theta, -grads + self.l2_coeff * theta)  # keeps no state
        #logger.debug(
        #    'Optimizer {} finished computing gradients'.format(
        #        self.optim_id))

        step_t_end = time.time()

        return theta, StepStats(
            po_returns_mean=po_returns.mean(),
            po_returns_median=np.median(po_returns),
            po_returns_std=po_returns.std(),
            po_returns_max=po_returns.max(),
            po_theta_max=po_theta_max,
            po_returns_min=po_returns.min(),
            po_len_mean=po_lengths.mean(),
            po_len_std=po_lengths.std(),
            noise_std=self.noise_std,
            learning_rate=self.optimizer.stepsize,
            theta_norm=np.square(theta).sum(),
            grad_norm=float(np.square(grads).sum()),
            update_ratio=float(update_ratio),
            episodes_this_step=episodes_this_step,
            timesteps_this_step=timesteps_this_step,
            time_elapsed_this_step=step_t_end - step_t_start,
        )

    def reset_optimizers(self):
        self.optimizer.reset()
        self.noise_std = self.init_noise_std


    def compute_grads(self, step_results, theta):
        '''Computes and returns gradients for the thetas based on run results

        Function takes the run results E, normalized them, and computes the weighted sum of the 
        noise vectors to construct the gradient G = sum_n E(theta + epsilon)epsilon
        
        Parameters
        ---------- 
        step_results: list of POResult
            A list of POResult objects for each run in the chunk
        
        theta: np.Array
            Array of theta values

        Returns
        -------
        grads: np.Array
            Gradiant vector
        
        po_theta_max: np.Array
            TODO: Maximum possible addiition to theta? Unsure what this is used for besides reproting.
		'''

        noise_inds, returns, _ = self.collect_po_results(step_results)

        theta_len = len(theta)
        pos_row, neg_row = returns.argmax(axis=0)
        noise_sign = 1.0
        po_noise_ind_max = noise_inds[pos_row]

        if returns[pos_row, 0] < returns[neg_row, 1]:
            noise_sign = -1.0
            po_noise_ind_max = noise_inds[neg_row]

        po_theta_max = theta + noise_sign * self.noise_std * es.noise.get(po_noise_ind_max, theta_len)

        if self.returns_normalization == 'centered_ranks':
            proc_returns = compute_CS_ranks(returns)
        elif self.returns_normalization == 'normal':
            proc_returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        else:
            raise NotImplementedError(
                'Invalid return normalization `{}`'.format(
                    self.returns_normalization))

        grads, _ = compute_weighted_sum(
            weights = proc_returns[:, 0] - proc_returns[:, 1],
            vec_generator = (es.noise.get(idx, theta_len) for idx in noise_inds),
            theta_size = theta_len)

        grads /= len(returns)
        if self.normalize_grads_by_noise_std:
            grads /= self.noise_std
        return grads, po_theta_max
    

    def collect_po_results(self, po_results):
        '''Extracts run results from POResult object
        
        Parameters
        ----------
        po_results: POResult
            A POResult named tuple. POResults objects are created by run_po_batch_fiber
        
        Returns
        -------
        noise_inds: np.Array
            Array where each row is the noise for one run
        returns: np.Array
            Array with two scores, one for plus noise one for minus scores
        lengths: np.Array
            Array of run lengths
        '''

        noise_inds = np.concatenate([r.noise_inds for r in po_results])
        returns = np.concatenate([r.returns for r in po_results])
        lengths = np.concatenate([r.lengths for r in po_results])
        return noise_inds, returns, lengths