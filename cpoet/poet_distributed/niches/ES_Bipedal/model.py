import numpy as np
from numpy.random import Generator, PCG64DXSM
import json
import time
import os
import datetime
import imageio
import logging
logger = logging.getLogger(__name__)

def make_model(game):
    # can be extended in the future.
    model = Model(game)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def passthru(x):
    return x

# useful for discrete actions


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Model:
    ''' simple feedforward model '''

    def __init__(self, game, seed, args = None):
        self.args = args
        self.np_random = Generator(PCG64DXSM(seed=seed))

        self.output_noise = game.output_noise
        self.env_name = game.env_name

        self.rnn_mode = False  # in the future will be useful
        self.time_input = 0  # use extra sinusoid input
        self.sigma_bias = game.noise_bias  # bias in stdev of output
        self.sigma_factor = 0.5  # multiplicative in stdev of output
        if game.time_factor > 0:
            self.time_factor = float(game.time_factor)
            self.time_input = 1
        self.input_size = game.input_size
        self.output_size = game.output_size

        self.shapes = [(self.input_size + self.time_input, game.layers[0])]
        self.shapes += [(game.layers[i], game.layers[i+1]) for i in range(len(game.layers)-1)]
        self.shapes += [(game.layers[-1], self.output_size)]

        self.critic_shapes = [(self.input_size + self.time_input, game.layers[0])]
        self.critic_shapes += [(game.layers[i], game.layers[i+1]) for i in range(len(game.layers)-1)]
        self.critic_shapes += [(game.layers[-1], 1)]

        self.sample_output = False
        if game.activation == 'relu':
            self.activations = [relu, relu, passthru]
        elif game.activation == 'sigmoid':
            self.activations = [np.tanh, np.tanh, sigmoid]
        elif game.activation == 'softmax':
            self.activations = [np.tanh, np.tanh, softmax]
            self.sample_output = True
        elif game.activation == 'passthru':
            self.activations = [np.tanh, np.tanh, passthru]
        else:
            self.activations = [np.tanh, np.tanh, np.tanh]

        self.actor_weight = []
        self.actor_bias = []
        self.actor_bias_log_std = []
        self.actor_bias_std = []
        self.actor_param_count = 0

        idx = 0
        for shape in self.shapes:
            self.actor_weight.append(np.zeros(shape=shape))
            self.actor_bias.append(np.zeros(shape=shape[1]))
            self.actor_param_count += (np.product(shape) + shape[1])
            if self.output_noise[idx]:
                self.actor_param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.actor_bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.actor_bias_std.append(out_std)
            idx += 1

        self.render_mode = False

    def __repr__(self):
        return "{}".format(self.__dict__)

    def get_action(self, x, t=0, mean_mode=False):
        # if mean_mode = True, ignore sampling.
        h = np.array(x).flatten()
        if self.time_input == 1:
            time_signal = float(t) / self.time_factor
            h = np.concatenate([h, [time_signal]])
        num_layers = len(self.actor_weight)
        for i in range(num_layers):
            w = self.actor_weight[i]
            b = self.actor_bias[i]
            h = np.matmul(h, w) + b
            if (self.output_noise[i] and (not mean_mode)):
                out_size = self.shapes[i][1]
                out_std = self.actor_bias_std[i]
                output_noise = self.np_random.standard_normal(size=out_size) * out_std
                h += output_noise
            h = self.activations[i](h)

        if self.sample_output:
            h = np.argmax(self.np_random.multinomial(n=1, pvals=h, size=1))

        return h

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.actor_weight[i] = chunk[:s_w].reshape(w_shape)
            self.actor_bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s
            if self.output_noise[i]:
                s = b_shape
                self.actor_bias_log_std[i] = np.array(
                    model_params[pointer:pointer + s])
                self.actor_bias_std[i] = np.exp(
                    self.sigma_factor * self.actor_bias_log_std[i] + self.sigma_bias)
                if self.render_mode:
                    print("bias_std, layer", i, self.actor_bias_std[i])
                pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return self.np_random.standard_normal(size=self.actor_param_count) * stdev

    def get_zeroed_model_params(self):
        return np.zeros(self.actor_param_count)