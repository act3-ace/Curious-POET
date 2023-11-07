from copyreg import pickle
from .env import BipedalWalkerCustom
from collections import OrderedDict
from collections import namedtuple
from ...es import StepStats
from .model import Model
from .opt import OptimizationControler
from .cppn import CppnEnvParams
import numpy as np
import datetime
import imageio
import logging
import pickle
import json
import dill
import time
import os

logger = logging.getLogger(__name__)

### TODO: These should maybe be set as class constants
final_mode = False
render_mode = False
RENDER_DELAY = False
record_video = False
MEAN_MODE = False

### TODO: This should maybe live in the env, or deleted entirly since 
### they're redundant to the cppn.

CreateGym = BipedalWalkerCustom

POResult = namedtuple('POResult', [
    'noise_inds',
    'returns',
    'lengths',
    'rollouts',
])

Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size',
                           'output_size', 'layers', 'activation', 'noise_bias',
                           'output_noise'])

bipedhard_custom = Game(env_name='BipedalWalkerCustom-v0',
                        input_size=24,
                        output_size=4,
                        time_factor=0,
                        layers=[40, 40],
                        activation='tanh',
                        noise_bias=0.0,
                        output_noise=[False, False, False],
                        )

class Niche:
    @staticmethod
    def new_parameters(seed):
        return(CppnEnvParams(seed))

    @staticmethod
    def class_setup(args):
        pass

    @staticmethod
    def iteration_update(iteration):
        pass

    def __init__(self, env_evo_params, seed, init='random', stochastic=False, args=None):
        self.args = args
        # if not isinstance(env_configs, list):
        #     env_configs = [env_configs]
        # self.env_configs = OrderedDict()
        # for env in env_configs:
        #     self.env_configs[env.name] = env
        self.seed = seed
        self.stochastic = stochastic

        self.env_evo_params = env_evo_params
        self.env = CreateGym()
        self.model = Model(bipedhard_custom, seed)
        self.opt_control = OptimizationControler(args, self.model)
        #logger.info('Created OptimizationControler ' + str(self.opt_control) )
        self.init = init
        ## Remember: If you add anything here you need to add it to get and set state!
        ## otherwise it won't recover from being shared with fiber
    def __getstate__(self):
        return {#"env_configs": self.env_configs,
                "env_evo_params": self.env_evo_params,
                "seed": self.seed,
                "stochastic": self.stochastic,
                "init": self.init,
                "args": self.args,
                }

    def __setstate__(self, state):
        self.args = state["args"]
        self.seed = state["seed"]
        self.env = CreateGym()
        self.model = Model(bipedhard_custom, self.seed, args = self.args)
        self.opt_control = OptimizationControler(self.args, self.model)
        #self.env_configs = state["env_configs"]
        self.env_evo_params = state["env_evo_params"]
        self.stochastic = state["stochastic"]
        self.init = state["init"]

    def checkpoint(self, opt_dir):
        json_dict = {}
        for k, i in self.__dict__.items():
            if k != "env":
                json_dict[k] = i
        
        pickle.dump(json_dict, open(os.path.join(opt_dir, "opt.pkl"), 'wb'))
        self.env_evo_params.checkpoint(opt_dir)
        
        #dill.dump(self.__dict__, open(os.path.join(opt_dir, "opt.pkl"), 'wb'))
        return

        manifest = {"seed": self.seed,
                    "stochastic": self.stochastic,
                    "init": self.init}

        with open(os.path.join(opt_dir, "niche_manifest.json"), 'w') as f:
            json.dump(manifest,f)

        self.env_evo_params.checkpoint(opt_dir)
        #self.env.checkpoint()
        #self.model.checkpoint()
        self.opt_control.checkpoint(opt_dir)

        pass

    def reload(self, opt_dir):
        json_dict = pickle.load(open(os.path.join(opt_dir, "opt.pkl"), 'rb'))

        for k in self.__dict__.keys():
            if k != "env":
                self.__dict__[k] = json_dict[k]


        return
        with open(os.path.join(opt_dir, "niche_manifest.json"), 'r') as f:
            manifest = json.load(f)
            self.seed = manifest["seed"]
            self.stochastic = manifest["stochastic"]
            self.init = manifest["init"]

        self.env_evo_params.reload(opt_dir)
        self.opt_control.reload(opt_dir)
        pass

    def initial_theta(self):
        if self.init == 'random':
            return self.model.get_random_model_params()
        elif self.init == 'zeros':
            return self.model.get_zeroed_model_params()
        else:
            raise NotImplementedError(
                'Undefined initialization scheme `{}`'.format(self.init))

    def rollout_opt_batch(self, theta, batch_size, random_state, noise):
        noise_inds = np.asarray([noise.sample_index(random_state, len(theta))
                             for i in range(batch_size)],
                            dtype='int')

        returns = np.zeros((batch_size, 2))
        lengths = np.zeros((batch_size, 2), dtype='int')

        returns[:, 0], lengths[:, 0], rollouts_0 = self._rollout_opt_batch(
            (theta + self.args.noise_std * noise.get(noise_idx, len(theta))
                for noise_idx in noise_inds), batch_size, random_state)

        returns[:, 1], lengths[:, 1], rollouts_1 = self._rollout_opt_batch(
            (theta - self.args.noise_std * noise.get(noise_idx, len(theta))
                for noise_idx in noise_inds), batch_size, random_state)

        return POResult(
            returns=returns, 
            noise_inds=noise_inds, 
            lengths=lengths, 
            rollouts = rollouts_0 + rollouts_1,
            )
        
    def _rollout_opt_batch(self, thetas, batch_size, random_state, eval=False):  
        
        ## If your optimization step is diferent than your eval step, that can be recorded here
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype='int')

        for i, theta in enumerate(thetas):
            returns[i], lengths[i] = self._rollout(
                theta, random_state=random_state, eval=eval)

        return returns, lengths, []

    def rollout_eval_batch(self, thetas, batch_size, random_state, return_rollouts=False):
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype='int')

        for i, theta in enumerate(thetas):
            returns[i], lengths[i] = self._rollout(
                theta, random_state=random_state, eval=True)

        return returns, lengths, None

    def _rollout(self, theta, random_state, eval=False):
        self.model.set_model_params(theta)
        total_returns = 0
        total_length = 0

        returns, lengths = self._simulate(
            self.model, self.env, random_state, train_mode=not eval, num_episode=1, 
                env_evo_params=self.env_evo_params)
        total_returns += returns[0]
        total_length += lengths[0]
        return total_returns, total_length, None

    def get_mutated_params(self):
        return(self.env_evo_params.get_mutated_params())

    def reset_optimizers(self):
        self.opt_control.reset_optimizers()

    def combine_steps(self, step_results, theta, step_t_start, propose_with_adam=True, decay_noise=True, propose_only=False):
        return(self.opt_control.combine_steps(step_results, theta, step_t_start, propose_with_adam=True, decay_noise=True, propose_only=False))

    def _simulate(self, model, env, random_state, train_mode=False, render_mode=False, num_episode=5,
             max_len=-1, env_config_this_sim=None, env_evo_params=None):
        '''Perform one full round of simulation.

        Parameters
        ----------
        model: Model
            Model class
        seed: int
            Random seed
        train_mode: boolean
            If train_mode is true than add random noise to each computation?
        render_mode: str
            Render mode for env.render, if exists
        num_episode: int
            Number of times to run simulation
        max_len: int
            Maximum episode length, but only happens when train_mode is set to true?
        env_config_this_sim: Env_config
            Set a new env config for the 
        env_params: cppn.CppnEnvParams
            Set new cppn parameters for this runthough
        TODO

        Returns
        -------
        reward_list: list
            List of total rewards, one for each num_episode
        t_list: list
            List of timesteps, one for each episode
        '''

        reward_list = []
        t_list = []

        # TODO - Why is this hardcoded?
        max_episode_length = 2000

        save_dir = None

        if train_mode and max_len > 0:
            if max_len < max_episode_length:
                max_episode_length = max_len

        env.seed(random_state.integers(2**31-1, dtype=int))

        sample = 0

        if model.args.visualize_freq>0:
            if model.args.visualize_freq == 1:
                sample = 1
            else:
                sample = random_state.integers(low=1, high=model.args.visualize_freq,
                                               endpoint=True)

        if sample == 1:
            render_mode = 'rgb_array'
            logger.info(f'{self.env_evo_params.get_name()} saving gif...')

        # if env_config_this_sim:
        #     env.set_env_config(env_config_this_sim)

        if env_evo_params:
            env.augment(env_evo_params)

        for _ in range(num_episode):

            if model.rnn_mode:
                model.reset()

            obs = env.reset()
            if obs is None:
                obs = np.zeros(model.input_size)

            total_reward = 0.0

            frames = []
            for t in range(max_episode_length):

                if render_mode:
                    if render_mode == 'rgb_array':
                        if t % model.args.frame_skip == 0:
                            frames.append(env.render(render_mode))

                    else:
                        env.render(render_mode) 
                    if RENDER_DELAY:
                        time.sleep(0.01)


                if model.rnn_mode:
                    model.update(obs, t)
                    action = model.get_action()
                else:
                    if MEAN_MODE:
                        action = model.get_action(
                            obs, t=t, mean_mode=(not train_mode))
                    else:
                        action = model.get_action(obs, t=t, mean_mode=False)

                obs, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    break

            if render_mode:
                if render_mode == 'rgb_array':
                    if len(frames) > 0:
                        temp = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
                        save_dir = os.path.join(model.args.log_file, "Videos")
                        filename = os.path.join(save_dir, f"opt_{self.env_evo_params.get_name()}_{temp}.gif")
                        imageio.mimwrite(filename, frames, fps=int(60/model.args.frame_skip))
                        logger.debug(F"capturing gif: {filename}")

            #print("reward", total_reward, "timesteps", t)
            reward_list.append(total_reward)
            t_list.append(t)

        return reward_list, t_list