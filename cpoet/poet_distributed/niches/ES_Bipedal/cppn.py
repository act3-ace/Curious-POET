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


import datetime
from collections import deque
import json
import neat
from neat.six_util import iteritems, iterkeys
import numpy as np
from numpy.random import Generator, PCG64DXSM
import pickle
import time
import os
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt


class PrettyGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)

    def __str__(self):
        connections = [c for c in self.connections.values() if c.enabled]
        connections.sort()
        s = "Fitness: {0}\nNodes:".format(self.fitness)
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nConnections:"
        for c in connections:
            s += "\n\t" + str(c)
        return s


class CppnEnvParams:
    x = np.array([(i - 200 / 2.0) / (200 / 2.0) for i in range(200)])
    def __init__(self, seed, cppn_config_path='config-cppn', genome_path=None):
        self.parent = None
        self.seed = seed
        self.np_random = Generator(PCG64DXSM(seed=seed))
        self.cppn_config_path = os.path.dirname(__file__) + '/' + cppn_config_path
        self.genome_path = genome_path
        self.hardcore = False
        self.cppn_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.cppn_config_path)
        self.cppn_genome = None
        self.altitude_fn = lambda x: x
        if genome_path is not None:
            self.cppn_genome = pickle.load(open(genome_path, 'rb'))
        else:
            start_cppn_genome = PrettyGenome('0')
            start_cppn_genome.configure_new(self.cppn_config.genome_config)
            start_cppn_genome.nodes[0].activation = 'identity'
            self.cppn_genome = start_cppn_genome
        self.reset_altitude_fn()

    def get_name(self):
        #logger.info(str(hash(str(self.cppn_genome))))
        #logger.info(str(self.cppn_genome))
        return(str(hash(str(self.cppn_genome))))

    def reset_altitude_fn(self):
        net = neat.nn.FeedForwardNetwork.create(self.cppn_genome, self.cppn_config)
        self.altitude_fn = net.activate

    def get_mutated_params(self):
        is_valid = False
        while not is_valid:
            mutated = copy_genome(self.cppn_genome)
            mutated.nodes[0].response = 1.0
            mutated.key = datetime.datetime.utcnow().isoformat()
            mutated.mutate(self.cppn_config.genome_config)
            is_valid = is_genome_valid(mutated) & (self.cppn_genome.distance(mutated, self.cppn_config.genome_config) > 0)
            if not is_valid:
                continue
            net = neat.nn.FeedForwardNetwork.create(mutated, self.cppn_config)
            y = np.array([net.activate((xi, )) for xi in self.x])
            y -= y[0] # normalize to start at altitude 0
            threshold_ = np.abs(np.max(y))
            is_valid = (threshold_ > 0)
            if not is_valid:
                continue
            if threshold_ < 0.25:
                mutated.nodes[0].response = (self.np_random.random() / 2 + 0.25) / threshold_
            if threshold_ > 16:
                mutated.nodes[0].response = (self.np_random.random() * 4 + 12) / threshold_
            res = CppnEnvParams(self.np_random.integers(low=2**32-1, dtype=int))
            res.cppn_genome = mutated
            res.reset_altitude_fn()
            res.parent = self.get_name()

            return res

    def xy(self):
        net = neat.nn.FeedForwardNetwork.create(self.cppn_genome, self.cppn_config)
        y = np.array([net.activate((xi, )) for xi in self.x])
        return({'x': self.x.tolist(), 'y': y.tolist()})

    def save_xy(self, folder='/tmp'):
        #with open(folder + '/' + self.cppn_genome.key + '_xy.json', 'w') as f:
        with open(os.path.join(folder,'_xy.json'), 'w') as f:
            net = neat.nn.FeedForwardNetwork.create(self.cppn_genome, self.cppn_config)
            y = np.array([net.activate((xi, )) for xi in self.x])
            f.write(json.dumps({'x': self.x.tolist(), 'y': y.tolist()}))

            plt.clf()
            plt.plot(self.x.tolist(), y.tolist())
            plt.savefig(os.path.join(folder,"terrain.png"))

    def to_json(self):
        return json.dumps({
            'cppn_config_path': self.cppn_config_path,
            'genome_path': self.genome_path,
            'parent': self.parent,
            'np_random': self.np_random.bit_generator.state
        })

    def save_genome(self, file_path=None):
        if file_path is None:
            file_path = '/tmp/genome_{}_saved.pickle'.format(time.time())

        pickled = pickle.dump(self.cppn_genome, open(file_path, 'wb'))

    def save(self, folder):
        folder = os.path.join(folder, "saved_envs", self.get_name())
        self._save(folder)

    def _save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.save_genome(os.path.join(folder, "_genome.pkl"))
        self.save_xy(folder)

        with open(os.path.join(folder, '_config.json'), 'w') as f:
            f.write(self.to_json())

        with open(os.path.join(folder, 'genome.txt'), 'w') as f:
            f.write(str(self.cppn_genome))
    
    def checkpoint(self, folder):
        self._save(os.path.join(folder,"cppn"))

    def reload(self, folder):

        dir = os.path.join(folder,"cppn")

        self.cppn_genome = pickle.load(open(os.path.join(dir, "_genome.pkl"), 'rb'))
        
        # create prng, set state within open()
        self.np_random = Generator(PCG64DXSM())

        with open(os.path.join(dir, "_config.json")) as f:
            json_dict = json.load(f)

        self.cppn_config_path = json_dict["cppn_config_path"]
        self.genome_path = json_dict["genome_path"]
        self.parent = json_dict["parent"]
        self.np_random.bit_generator.state = json_dict["np_random"]
        self.reset_altitude_fn()


def copy_genome(genome):
    file_path = '/tmp/genome_{}.pickle'.format(time.time())
    pickled = pickle.dump(genome, open(file_path, 'wb'))
    return pickle.load(open(file_path, 'rb'))

def is_genome_valid(g):
    graph = {}
    for key in g.connections.keys():
        if key[0] not in graph:
            graph[key[0]] = []
        graph[key[0]].append(key[1])
    q = deque([-1])
    while len(q) > 0:
        cur = q.popleft()
        if cur == 0:
            return True
        if cur not in graph:
            continue
        for node in graph[cur]:
            q.append(node)
    return False
