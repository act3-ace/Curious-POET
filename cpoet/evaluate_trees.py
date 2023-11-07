import networkx as nx
import imageio as io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pygraphviz as pgv
import numpy as np
import pydot
import glob
import json 
import os
import re
import cv2
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xlim", type=float, required=False, nargs='+', help="--xlim=0.0 25.0")
    parser.add_argument("--ylim", type=float, required=False, nargs='+', help="--ylim=0.0 25.0")
    parser.add_argument("--xlim1", type=float, required=False, nargs='+', help="--xlim1=0.0 25.0")
    parser.add_argument("--ylim1", type=float, required=False, nargs='+', help="--ylim1=0.0 25.0")
    parser.add_argument("--prefix", type=str, required=False, default=None)
    args=parser.parse_args()
    return args

def imageio_gif(datapath, run_name):
    files = sorted(glob.glob(os.path.join(datapath, run_name, 'plots/trees/*.png')), key=lambda x:float(re.findall("(\d+)",x)[-1]))
    max_dims = [max(k) for k in zip(*[Image.open(f).size for f in files])]

    with io.get_writer(os.path.join(datapath, run_name, f'plots/trees/{run_name}_gif.gif'), mode='I', duration=1) as writer:
        for filename in files:
            image = io.imread(filename)
            wh_ratio = max_dims[0]/max_dims[1]
            h, w, _ = image.shape 
            if w/h > wh_ratio:
                # add h
                padw = 0
                padh = round(w/wh_ratio - h) 
            else: 
                # add w
                padh = 0
                padw = round((h*wh_ratio - w)/2)
            # padw = int((max_dims[0] - w)/2)
            # padh = max_dims[1] - h
            image = np.pad(image, pad_width=((0, padh),(padw, padw),(0,0)), constant_values=255)
            new_ratio = image.shape[1]/image.shape[0]
            image = cv2.resize(image, dsize=(int(max_dims[0]*0.5), int(max_dims[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            writer.append_data(image)

def main():
    args = parse_args()
    datapath = args.output_dir
    run_name = args.run_name
    with open(os.path.join(datapath, run_name, 'logs/phylo_tuples.json')) as tuples:
        tuple_data = json.load(tuples)

    if not os.path.exists(os.path.join(datapath, run_name, 'plots')):
        os.mkdir(os.path.join(datapath, run_name, 'plots'))
    if not os.path.exists(os.path.join(datapath, run_name, 'plots','trees')):
        os.mkdir(os.path.join(datapath, run_name, 'plots','trees'))

    

    it_vec = sorted(set([x['iteration'] for x in tuple_data]))
    phylo_tuples = []
    annecs = np.zeros((len(it_vec), 1))
    num_envs = np.zeros((len(it_vec), 1))
    total_transfers = np.zeros((len(it_vec), 1))
    longest_phylo_path = np.zeros((len(it_vec), 1))
    proportion_nodes_w_gt1_leaf = np.zeros((len(it_vec), 1))


    for i, it in enumerate(it_vec): 
        phylo_tuples = phylo_tuples + [(x['parent'], x['child']) for x in tuple_data if (x['link_type']=='env_evolution' and x['iteration']==it)]
        transfers_it = [(x['parent'], x['child']) for x in tuple_data if (x['link_type']=='agent_transfer' and x['iteration']==it)]
        annecs[i] = max(x['annecs'] for x in tuple_data if x['iteration']==it)


        # Plot transfers on fixed pydot graph
        if i % 10 == 0: 
            tree = pgv.AGraph(directed=True, splines=True)
            tree.add_edges_from(phylo_tuples)
            tree.layout(prog='dot')
            tree.add_edges_from(transfers_it, color='red')
            tree_filename = os.path.join(datapath, run_name, 'plots','trees', f'{run_name}_tree_wtransfers_pydot{it}.png')
            if not os.path.exists(tree_filename):
                tree.draw(tree_filename, args='')

        # Metrics - compute these on graph without transfers 
        tree = nx.DiGraph()
        tree.add_edges_from(phylo_tuples)
        num_envs[i] = len(tree.nodes)

        tree_undir = nx.Graph(tree)
        total_transfers[i] = len(transfers_it)
        #transfer_lengths = [nx.shortest_path_length(tree_undir,u,v) for (u,v) in transfers]
        #max_transfer_lengths = max(transfer_lengths)

        longest_phylo_path[i] = max((dict(nx.shortest_path_length(tree))['flat']).values())
        degrees = [d for n, d in tree.degree()]
        #max_leaves = max(degrees)-1
        proportion_nodes_w_gt1_leaf[i] = sum([1 for d in degrees if d > 2])/num_envs[i]

    imageio_gif(datapath, run_name)

    to_plot = [(num_envs, 'Number of environments created', 'num_envs'), (total_transfers, 'Number of transfers at iteration', 'total_transfers'), 
                (longest_phylo_path, 'Longest path in tree', 'longest_phylo_path'), (annecs, 'ANNECS', 'annecs'),
                (proportion_nodes_w_gt1_leaf, 'Proportion of nodes with more than 1 leaf', 'proportion_nodes_w_gt1_leaf')]

    plt.figure()
    for data, label, _ in to_plot[:4]:
        plt.plot(it_vec, data, label=label)
    plt.legend()
    plt.grid(color = 'lightgrey')
    plt.xlabel('iteration')
    plt.title(f"POET run: '{run_name}'")
    if args.xlim: plt.xlim(args.xlim)
    if args.ylim: plt.ylim(args.ylim)
    plt.savefig(os.path.join(datapath, run_name, 'plots', f"{args.prefix if args.prefix else ''}{run_name}_metrics.png"))

    plt.figure()
    plt.plot(it_vec, to_plot[4][0], label=to_plot[4][1])
    plt.legend()
    plt.grid(color = 'lightgrey')
    plt.xlabel('iteration')
    plt.title(f"POET run: '{run_name}'")
    if args.xlim1: plt.xlim(args.xlim1)
    if args.ylim1: plt.ylim(args.ylim1)
    plt.savefig(os.path.join(datapath, run_name, 'plots', f"{args.prefix if args.prefix else ''}{run_name}_{to_plot[4][2]}.png"))

if __name__ == "__main__":
    main()
