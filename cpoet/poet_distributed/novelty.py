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


import numpy as np
from .stats import compute_CS_ranks
from .iotools import NumpyDecoder, NumpyEncoder, saveKeyedTuple, loadKeyedTuple
import os
import json

def cap_score(score, lower, upper):
    if score < lower:
        score = lower
    elif score > upper:
        score = upper

    return score

class pata_ec():
    def __init__(self, args):
        ## Archive pata_ec scores of archives envs
        self.mc_lower = args.mc_lower
        self.mc_upper = args.mc_upper
        self.archived_pata_ec = {}
        self.pata_list = None
        self.k = 5
        self.save_archived_novelty = args.save_archived_novelty
        self.log_pata_ec = args.log_pata_ec
        self.log_file = args.log_file

    def checkpoint(self, folder):
        if len(list(self.archived_pata_ec.keys()))>0:
            saveKeyedTuple(self.archived_pata_ec, os.path.join(folder, 'archived_pata_ec.csv'))
        
        archived_pata_ec = self.archived_pata_ec
        del self.__dict__["archived_pata_ec"]

        with open(os.path.join(folder, 'pata_ec.json'), 'w') as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder)

        self.__dict__["archived_pata_ec"] = archived_pata_ec

    def reload(self, folder):
        with open(os.path.join(folder, 'pata_ec.json'), 'r') as f:
            dct = json.load(f, cls=NumpyDecoder)
            dct["archived_pata_ec"] = loadKeyedTuple(os.path.join(folder, 'archived_pata_ec.csv'))
            self.__dict__ = dct

    def novelty(self, arc_opts,  opts, opt_list):
        all_opt = (*arc_opts.values(), *opts.values())

        pata_ec = {}
        tasks = {}

        for s in all_opt:
            for t in opt_list:
                task = t.start_theta_eval(s.theta)
                tasks[(t, s)] = task

        for (t,s), task in tasks.items():
            pata_ec[(t, s)] = cap_score(t.get_theta_eval(task).eval_returns_mean, 
                                        self.mc_lower, 
                                        self.mc_upper)

        new_patas = {}
        for t in opt_list:
            capped_scores = []
            for s in all_opt:
                capped_scores.append(pata_ec[(t, s)])

            new_patas[t] = compute_CS_ranks(np.array(capped_scores))

        novelty = {}
        
        for o,v in new_patas.items():
            ### Compute the patasec for o against self.pata_list. 
            distances = [euclidean_distance(v, c) for c in self.pata_list]

            distances = np.array(distances)
            top_k_indicies = (distances).argsort()[:self.k]
            top_k = distances[top_k_indicies]
            novelty[o] = top_k.mean()


        return(novelty)

    def update_novelty(self, arc_opts, opts, iteration):
        tasks = {}
        pata_ec = {}

        all_opt = (*arc_opts.values(), *opts.values())

        ## s - Source, the agent. This is fixed iff niche is archived
        ## t - Target, the enironmnet. Always fixed after creation

        ## Place eval tasks in pool, but for archived ones only if they're new
        for t in all_opt:
            for s in all_opt:
                if not (t.optim_id,s.optim_id) in self.archived_pata_ec.keys():
                    #print(f"novelty:: placing {t.optim_id}, {s.optim_id}, {hash(s.theta.data.tobytes())}")
                    task = t.start_theta_eval(s.theta)

                    tasks[(t, s)] = task

        ## Get eval tasks from pool. Construct pata_ec record. 
        for s in arc_opts.values():
            #print("novelty:: ++++++++++++++++I NEVER GET CALLED!!!!++++++++++++++++")
            for t in all_opt:
                if (t.optim_id,s.optim_id) in self.archived_pata_ec.keys():
                    #print("novelty:: I'm retreiving an old score!")
                    pata_ec[(t, s)] = self.archived_pata_ec[(t.optim_id, s.optim_id)]
                else: 
                    assert (t,s) in tasks.keys()
                    task = tasks[(t, s)]
                    pata_ec[(t, s)] = cap_score(t.get_theta_eval(task).eval_returns_mean, 
                                                self.mc_lower, 
                                                self.mc_upper)
                    if self.save_archived_novelty:
                        self.archived_pata_ec[(t.optim_id, s.optim_id)] = pata_ec[(t, s)]

        
        for t in all_opt:
            for s in opts.values():
                task = tasks[(t, s)]
                pata_ec[(t, s)] = cap_score(t.get_theta_eval(task).eval_returns_mean, 
                                            self.mc_lower, 
                                            self.mc_upper)
                #print(f"novelty:: Novelty report: {t.optim_id},{s.optim_id}: {pata_ec[(t, s)]}")

        pata_list = []
        for t in all_opt:
            capped_scores = []
            for s in all_opt:
                capped_scores.append(pata_ec[(t, s)])

            pata_list.append(compute_CS_ranks(np.array(capped_scores)))

        #print(pata_ec)
        #print(pata_list)

        self.pata_list = pata_list

        if self.log_pata_ec:
            dir = os.path.join(self.log_file,"pata_ec")
            os.makedirs(dir,exist_ok=True)

            with open(os.path.join(dir, f'pata_ec_{iteration}.csv'), 'w') as csvfile:
                
                # Output header
                for o in all_opt:
                    csvfile.write("," + str(o.optim_id))
                
                csvfile.write("\n")

                for score, o in zip(self.pata_list,all_opt):
                    if type(score)==int:score = [score]
                    
                    csvfile.write(str(o.optim_id)+",")
                    csvfile.write(",".join([str(i) for i in score]))
                    csvfile.write("\n")

            with open(os.path.join(dir, f'pata_ec_raw{iteration}.csv'), 'w') as csvfile:
                
                # Output header
                for o in all_opt:
                    csvfile.write("," + str(o.optim_id))
                
                csvfile.write("\n")

                for t in all_opt:
                    csvfile.write(str(t.optim_id))
                    for s in all_opt:
                        csvfile.write("," + str(pata_ec[(t, s)]))
                    
                    csvfile.write("\n")

    


def euclidean_distance(x, y):
    if type(x) is not list:
        #print("novelty.euclidean_distance:: inputs not of type list but of type", type(x))
        x = np.array([x])
        y = np.array([y])
    
    n, m = len(x), len(y)

    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)


# def compute_novelty(archived_optimizers, optimizers, opts, k, low, high):
#     '''Compute novelty of all optimizers on a single env. 
    
#     Score each theta on the niche env and then compute the eculdiean distance between the centered 
#     normalized ranking niches own centered normalized ranking. Return the mean of the top k difference
#     values. 

#     Parameters
#     ----------
#     archived_optimizers - dict
#         Dictionary of archvied optimizers
#     optimizers - dict
#         Dictionary of active optimizers
#     niche - ESOptimizer
#         Optimizer object containing env to evaluate on
#     k - int
#         Number of top scoring objects to return
#     low - float
#         Low end cutoff for scores
#     high - float
#         High end cutoff for scores

#     Returns
#     -------
#     float:
#         Mean distance to top k farthest other envs. 
#     '''

#     # setup distances list
#     distances = []

#     # calculate PATA-EC on this niche
#     niche.update_pata_ec(archived_optimizers, optimizers, low, high)

#     # Loop through archived optimizers, calculate distance between PATA-ECs
#     for point in archived_optimizers.values():
#         distances.append(euclidean_distance(point.pata_ec, niche.pata_ec))

#     # Loop through active optimizers, calculate distance between PATA-ECs
#     for point in optimizers.values():
#         distances.append(euclidean_distance(point.pata_ec, niche.pata_ec))


#     # Pick k nearest neighbors
#     #  20220921 JB
#     #  why not use numpy.ndarray.sort on distances? It would sort in place, and then we take first k
#     #  Even better, numpy.ndarray.partition. This does in-place partial partitioning, 
#     #   which is fine since we need the average of k things, not the perfect ording of 
#     #   k things.
#     #   https://numpy.org/doc/stable/reference/generated/numpy.ndarray.partition.html#numpy.ndarray.partition
#     #   https://stackoverflow.com/questions/37125495/how-to-get-the-n-maximum-values-per-row-in-a-numpy-ndarray
    
#     # distances = np.array(distances)
#     # np.ndarray.partition(distances, -k)
#     # res1 =  distances[-k:].mean()

#     distances = np.array(distances)
#     top_k_indicies = (distances).argsort()[:k]
#     top_k = distances[top_k_indicies]
#     res2 = top_k.mean()


#     return res2
