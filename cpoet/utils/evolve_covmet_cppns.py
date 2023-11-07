import os, json
import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

import poet_distributed.niches.ES_Bipedal.cppn as cppn

#####################################
##
## Create Coverage Metric CPPN Genomes
##
#####################################

def load_cppns(folder, num_cppns=10000, max_depth=100, debug=False, mode='normal'):
    folder = os.path.abspath(os.path.join(folder, '../../..'))
    fn = os.path.join(folder, f"{num_cppns}cppns_k{max_depth}.pkl")
    with open(fn, 'rb') as f:
        envs = pickle.load(f)
    if debug:
        return envs[:10]
    if mode == 'only_last':
        return envs[max_depth-1::max_depth]  # return only the last env in each evolutionary series
    elif isinstance(mode, int):
        return envs[mode-1::mode] # every nth environment
    elif isinstance(mode, dict):
        if 'index' in mode:
            indices = [x+mode['index']-1 for x in list(range(0, num_cppns, max_depth))]
        if 'specific_env' in mode:
            indices = [mode['specific_env']]
        return [envs[idx] for idx in indices]
    else: return envs

def open_evolve_cppns(dest_folder, num_cppns=10000, max_depth=100):
    logger.info(f"free evolving {num_cppns} cppns w/ max depth {max_depth}")
    envs = []
    n = cppn.CppnEnvParams(0)
    
    count = 0
    for i in range(num_cppns):
        envs.append(n)
        count += 1
        if count < max_depth:
            n = n.get_mutated_params()
        else:
            n = cppn.CppnEnvParams(0)
            count=0
    
    fn=os.path.join(dest_folder, f"{num_cppns}cppns_k{max_depth}.pkl")
    with open(fn, 'ab') as f:
        pickle.dump(envs, f)

    logger.info(f"Done evolving cppns.  Saved to {fn}")


def convert_to_json(dest_folder, num_cppns=10000, max_depth=100):
    logger.info(f"writing freely evolved set of {num_cppns} cppns w/ max depth {max_depth} to json...")
    in_fn=os.path.join(dest_folder, f"{num_cppns}cppns_k{max_depth}.pkl")
    out_fn=os.path.join(dest_folder, f"{num_cppns}cppns_k{max_depth}.json")

    with open(in_fn, 'rb') as f:
        envs = pickle.load(f)

    out = []
    for e in envs:
        out.append([y[0] for y in e.xy()['y']])

    with open(out_fn, 'w') as f:
        json.dump(out, f)

    logger.info(f"Done")


def load_coverage_data(data_file_path, min_cov_val = 230, min_solves=0, max_solves=18):
    cov_data = np.load(data_file_path)[0,0,:,:] 
    env_ids = np.array(range(len(cov_data)))
    num_solves = np.sum(cov_data > min_cov_val, axis=1)
    solve_mask = np.bitwise_and(num_solves>min_solves,num_solves<max_solves) # filter out the 0 and all success envs
    return cov_data[solve_mask], env_ids[solve_mask]


def terrain_fft(dest_folder, num_cppns=10000, max_depth=100):
    in_fn = os.path.join(dest_folder, f"{num_cppns}cppns_k{max_depth}.json")
    out_folder = os.path.join(dest_folder, 'terrain_clustering')

    with open(in_fn, 'r') as f:
        envs = json.load(f)

    # pull in not-too-easy & not too hard environments 
    _, env_ids = load_coverage_data(
        data_file_path='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_30nov/cp-2000/fixedset_coverage_metric_N10000_wRollouts/kjnm_databrick.npy',
    )

    interesting_envs = []
    for i, env in enumerate(envs):
        if i in env_ids:
            interesting_envs.append(env)

    envs = interesting_envs

    TRUNCATED_BINS = 20
    MAX_VALUE = 100
    x = range(200)
    xf = range(int(len(x)/2))
   

    # for k in [0, 10, 20, 50, 70, 99]:
    #     print(f"k={k}")
    #     fig, axs = plt.subplots(3,10)
    #     fig.set_size_inches(40, 8)
    #     for i in range(10):
    #         y = envs[i*100 + k]
    #         y -= np.mean(y)
    #         # y /= np.std(y)
    #         yf = np.real( np.fft.fft(y)[:int(len(x)/2)] )
    #         max_value = np.max(np.absolute(yf))

    #         axs[0,i].plot(x,y)
    #         axs[0,i].set_title(f"Terrain y vs x")

    #         axs[1,i].stem(xf,yf)
    #         axs[1,i].set_title(f"Fourier transform of Terrain y' vs x/2")
    #         axs[1,i].set_ylim((max_value, -max_value))

    #         max_value = np.max(np.absolute(yf[1:TRUNCATED_BINS]))
    #         axs[2,i].stem(xf[1:TRUNCATED_BINS],yf[1:TRUNCATED_BINS])
    #         axs[2,i].set_title(f"y' (first {TRUNCATED_BINS} bins)")
    #         axs[2,i].set_xticks(list(range(1,TRUNCATED_BINS)))
    #         axs[2,i].set_ylim((max_value, -max_value))

    #     fig.tight_layout()
    #     fig.savefig(os.path.join(out_folder, f"y_vs_spectral_k_{k}.png"))

    def list_duplicates_of(seq, item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item,start_at+1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs
    
    # zero mean each environment
    # collect spectral features for each terrain
    spec_feats = []
    for i, env in enumerate(envs):
        envs[i] -= np.mean(env)
        yf = np.fft.fft(env)[:int(len(x)/2)]
        # yf -= np.mean(yf)
        spec_feats.append(np.real(yf[1:TRUNCATED_BINS]))

    overall_tops=[]  # find the max and min for all the plots in this row
    overall_bottoms = []
    for env in envs:
        overall_tops.append(max(env))
        overall_bottoms.append(min(env))
    # from sklearn.cluster.contrib
    from sklearn.cluster import DBSCAN
    

    # print('\nDBSCAN\n')
    # max_num_examples = 10
    # max_clusters_plot = 20
    # METRICS = ['euclidean', 'cosine']
    # epsss = ([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    # for metric_index, metric in enumerate(METRICS):
    #     epss = epsss[metric_index]
    #     for e_index, e in enumerate(epss):
    #         db = DBSCAN(eps=e, metric=metric).fit(X=spec_feats) ###################################
    #         num_clusters = len(set(db.labels_))
    #         print(f'for eps={e}, {num_clusters} clusters discovered')
    #         num_clusters_plot = min(num_clusters, max_clusters_plot)
    #         fig, axs = plt.subplots(num_clusters_plot, max_num_examples) 
    #         fig.set_size_inches(10,8)
            
    #         # set all plots to invisible at first, to keep un-drawn ones from being visible
    #         for row in range(axs.shape[0]):
    #             for col in range(axs.shape[1]):
    #                 axs[row, col].set_visible(False)

    #         rows = [-1]+list(range(num_clusters-1))
    #         if len(rows) > num_clusters_plot:
    #             rows = rows[:num_clusters_plot]
    #         for plot_row, label in enumerate(rows):
                
    #             env_indices = list_duplicates_of(db.labels_.tolist(), label)
    #             trun_env_indices = env_indices[:min(len(env_indices), max_num_examples)]
    #             for col, env_index in enumerate(trun_env_indices):
    #                 env = envs[env_index]
    #                 axs[plot_row, col].plot(range(len(env)), env,)
    #                 axs[plot_row, col].set_visible(True)
    #                 axs[plot_row, col].xaxis.set_tick_params(labelbottom=False)
    #                 if col == 0:
    #                     axs[plot_row, col].set_ylabel(f"{rows[plot_row]} ({len(env_indices)})", rotation=0)
    #                 else:
    #                     axs[plot_row, col].xaxis.set_tick_params(labelleft=False)
    #                 axs[plot_row, col].set_xticks([])
    #                 axs[plot_row, col].set_yticks([])
    #             fig.suptitle(f"DBSCAN w/ {metric} metric over truncated terrain spectrum \nClusters (rows) with eps={e}\n {num_clusters} clusters found")  #################
    #         fig.savefig(os.path.join(out_folder, 'DBSCAN', f"DBSCAN_w_{metric}_trun_terrain_spectrum_clustering_{e_index}_eps_{e}.png"))  ###############
    #     print('Done with DBSCAN')
    
    def plot_noise_envs(count, env_indices, envs, mode, eps_index, eps, all=False):
        if all:
            dim = int(np.ceil(np.sqrt(count)))
            fig, axs = plt.subplots(dim, dim)
            fig.set_size_inches(dim*2, dim*2)
            row=0
            col=0
            for env_idx in env_indices:
                axs[row, col].plot( range(len(envs[env_idx])), envs[env_idx])
                axs[row, col].tick_params(bottom=False, left=False, right=False)
                axs[row, col].set_yticks([])
                axs[row, col].set_xticks([])
                col += 1
                if col > dim-1:
                    row += 1
                    col = 0
            fig.suptitle(f"Cluster -1 (Noise) environments") 
            fig.tight_layout()
            fig.savefig(os.path.join(out_folder, 'DBSCAN', f"DBSCAN_Euclidean_metric_over_env_{mode}_clustering_{eps_index}_eps_{eps}_noise_envs.png"))  
            print('')

        else:
            scales = [5, 2, 1, 0.5, 0.2, 0.1]
            counts = [0]*len(scales)
            fig, axs = plt.subplots(len(scales), 1) 
            fig.set_size_inches(6, len(scales)*1)

            label = -1

            for idx in env_indices:
                env = envs[idx]
                row = list(max(env) > scales).index(True)-1
                axs[row].plot(range(len(env)), env,)
                counts[row] += 1

            for row, scale in enumerate(scales):
                axs[row].set_ylim((-scale, scale))
                axs[row].set_title(f"{counts[row]} terrains")
                axs[row].tick_params(bottom=False)

            fig.suptitle(f"Cluster -1 (Noise) environments at various scales") 
            fig.tight_layout()
            fig.savefig(os.path.join(out_folder, 'DBSCAN', f"DBSCAN_Euclidean_metric_over_env_{mode}_clustering_{eps_index}_eps_{eps}_noise_envs.png"))  
            print('')

    
    import hdbscan
    import random

    print('\nDBSCAN\n')

    output = {}
    modes = [('spectra', [0.001, 0.002, 0.005, 0.01, .02, 0.05]), ('terrain', [0.001, 0.002, 0.005, 0.01, 0.013, 0.015, 0.017, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5])]#

    for (mode, epss) in modes:
        print(f"cluster mode: {mode}")
        output[mode]={}
        plot_num_clusters=[]
        for eps_index, eps in enumerate(epss):
            if mode == 'spectra':  data = spec_feats
            if mode == 'terrain':  data = envs
            
            hdb=DBSCAN(
                    eps=eps, 
                    metric='cosine',
                    # min_samples=1,
                    # cluster_selection_epsilon=0.01
                ).fit(X=data)
            output[mode][eps] = hdb.labels_
            num_clusters = len(set(hdb.labels_))
            plot_num_clusters.append(num_clusters)
            print(f'for eps={eps}, {num_clusters} clusters discovered')

            subplot_dim = int(np.ceil(np.sqrt(num_clusters)))
            fig, axs = plt.subplots(subplot_dim, subplot_dim) 
            fig.set_size_inches(subplot_dim*2, subplot_dim*2)
            
            # set all plots to invisible at first, to keep un-drawn ones from being visible
            for row in range(axs.shape[0]):
                for col in range(axs.shape if len(axs.shape)==1 else axs.shape[1]):
                    axs[row, col].set_visible(False)

            labels = [-1]+list(range(num_clusters-1))
            clusters = {}
            for label in labels:  # sort by cluster size
                a=list_duplicates_of(hdb.labels_.tolist(), label)
                clusters[label]= (len(a), a,)
            clusters = sorted(clusters.items(), key=lambda x: x[1][0], reverse=True)
            col=0
            row=0
            for label, (count, env_indices) in clusters:
                if label == -1: 
                    try:
                        plot_noise_envs(count, env_indices, envs, mode, eps_index, eps, all=True)
                        ...
                    except:
                        ...
                tops=[]  # find the max and min for all the plots in this cluster
                bottoms = []
                for idx in env_indices:
                    tops.append(max(envs[idx]))
                    bottoms.append(min(envs[idx]))

                for idx in env_indices:
                    env = envs[idx]
                    axs[row, col].plot(range(len(env)), env,)
                    if label == -1:
                        axs[row, col].set_ylim((-5, 5))
                    else:    
                        axs[row, col].set_ylim((min(bottoms)*1.2, max(tops)*1.2))
                    
                    axs[row, col].set_visible(True)
                    axs[row, col].tick_params(bottom=False)
                    axs[row, col].set_xticks([])
                    # axs[row, col].set_yticks([])
                    axs[row, col].set_title(f"{label} ({count})")
                col += 1
                if col > subplot_dim-1:
                    col = 0
                    row += 1

            fig.suptitle(f"DBSCAN w/ Euclidean metric over {mode} \nClusters with eps={eps}\n {num_clusters} clusters found") 
            fig.tight_layout()
            fig.savefig(os.path.join(out_folder, 'DBSCAN', f"DBSCAN_Euclidean_metric_over_env_{mode}_clustering_{eps_index}_eps_{eps}.png"))  
            print('')

        fig, ax = plt.subplots()
        ax.plot(epss, plot_num_clusters, 'o-')
        for i in range(len(epss)):
            ax.text(epss[i]+0, plot_num_clusters[i]+1, plot_num_clusters[i])
        ax.set_xlabel('DBSCAN EPSs')
        ax.set_ylabel('Number of clusters')
        ax.set_title(f"DBSCAN Environment Clustering on {mode}")   #####################################

        ax.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(out_folder, 'DBSCAN', f"num_clusters_vs_eps__{mode}.png")) #####################################

        output_fn = os.path.join(out_folder, 'DBSCAN', f"cluster_labels_vs_eps__{mode}.pkl") #############################
        with open(output_fn, 'wb') as f:
            pickle.dump(output, f)
    print('Done with DBSCAN')

    return 



def terrain_vendi_score(dest_folder, num_cppns=10000, max_depth=100):
    in_fn=os.path.join(dest_folder, f"{num_cppns}cppns_k{max_depth}.json")
    out_folder = os.path.join(dest_folder, 'terrain_clustering', 'vendi_score_analysis')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    with open(in_fn, 'r') as f:
        envs = json.load(f)

    TRUNCATED_BINS = 20
    MAX_VALUE = 100
    x = range(200)
    xf = range(int(len(x)/2))
   
    
    # zero mean each environment
    # collect spectral features for each terrain
    spec_feats = []
    for i, env in enumerate(envs):
        envs[i] -= np.mean(env)
        yf = np.fft.fft(env)[:int(len(x)/2)]
        # yf -= np.mean(yf)
        spec_feats.append(np.real(yf[1:TRUNCATED_BINS]))


  
    from vendi_score import vendi
    import tqdm

    def cos_sim(a, b):
        from numpy import dot 
        from numpy.linalg import norm

        cs = dot(a, b)/(norm(a) * norm(b))
        return cs

    configs = [
        (envs,          cos_sim,    'cosine similarity of terrain'),
        (spec_feats,    cos_sim,    'cosine similarity of truncated terrain spectra'),
    ]

    fig, axs = plt.subplots(1,1)
    # vendi score for 10k cov met envs

    for plot_num, (data, sim_met, desc) in enumerate(configs):
        vendi_vs_k = []
        print(f"starting {desc}")
        for k in tqdm.tqdm(range(100)):
            env_indices = list(range(10000))[k::100]
            envs_at_k = [data[x] for x in env_indices]
            vs_at_k = vendi.score(envs_at_k, sim_met)
            vendi_vs_k.append(vs_at_k)
            ...
        
        print(vendi_vs_k)
        axs.plot(range(len(vendi_vs_k)), vendi_vs_k, label=desc)
        axs.set_title(f"Vendi Score vs Random Evolution Step\n (10k coverage metric environments)")
        axs.grid(True)
        axs.set_ylabel(f"Vendi Score")
        axs.set_xlabel(f"Random Evolution Steps")
        axs.legend()
        print(f"done with {desc}\n\n")
        fig.tight_layout()
        fig.savefig(os.path.join(out_folder, f"Vendi_score_vs_evolution_depth_k.png"))


    # pull in not-too-easy & not too hard environments 
    _, env_ids = load_coverage_data(
        data_file_path='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/icm_gamma10.0_wCM_30nov/cp-2000/fixedset_coverage_metric_N10000_wRollouts/kjnm_databrick.npy',
    )

    interesting_envs = []
    interesting_spec_feats = []
    for i, env in enumerate(envs):
        if i in env_ids:
            interesting_envs.append(env)
            interesting_spec_feats.append(spec_feats[i])


    vs_interesting_envs = vendi.score(interesting_envs, cos_sim)
    print(f"VS for interesting envs terrain : {vs_interesting_envs:2.2f}")

    vs_interesting_spec_feats = vendi.score(interesting_spec_feats, cos_sim)
    print(f"VS for interesting envs spectrum : {vs_interesting_spec_feats:2.2f}")

    vs_spec_feats = vendi.score(spec_feats, cos_sim)
    print(f"VS for all 10k envs spectrum: {vs_spec_feats:2.2f}")

    vs_envs = vendi.score(envs, cos_sim)
    print(f"VS for all 10k envs: {vs_envs:2.2f}")



    # vs_scores=[]
    # for k in range(100):



    print('Done with Vendi scores.')

    return 


def look_at_cppns(folder = f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/10000cppns_k100.pkl"):

    with open(folder, 'rb') as f:
        cppns = pickle.load(f)

    def count_connections(cppn):
        count = 0
        for k, v in cppn.cppn_genome.connections.items():
            if v.enabled:
                count += 1
        return count

    fig, axs = plt.subplots()
    cons = np.zeros((100,100))

    for thread_idx in range(0,10000,100):
        for ev_step in range(100):
            num_connections = count_connections(cppns[thread_idx+ev_step])
            cons[int(thread_idx/100), ev_step] = num_connections
            # print(thread_idx, ev_step, thread_idx+ev_step, num_connections)

        axs.plot(range(100), cons[int(thread_idx/100)], color='gray')
    axs.plot(range(100), np.mean(cons, axis=0), color='red')
    axs.set_title(f"number of active connections")
    fig.savefig(f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/cppn_dims.png")
    print('hi')


if __name__ == '__main__':
    # terrain_vendi_score(
    #     dest_folder='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM',
    #     num_cppns=10000,
    #     max_depth=100,
    # )

    # terrain_fft(
    #     dest_folder='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM',
    #     num_cppns=10000,
    #     max_depth=100,
    # )

    # open_evolve_cppns(
    #     dest_folder='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM',
    #     num_cppns=10000,
    #     max_depth=100,

    # )
    look_at_cppns()