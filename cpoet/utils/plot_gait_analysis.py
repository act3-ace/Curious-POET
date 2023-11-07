import os, pickle
import argparse
import shutil
from matplotlib import pyplot as plt
import numpy as np
import tqdm

EXCLUDE_RUN_NAMES = []

def main(args):
    for this_checkpoint in args.checkpoints:
        def make_folder(results_folder):
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

        agg_gait_chars_fn = os.path.join(args.agg_gait_chars_pickle_folder, f"agg_gait_chars_cp_{this_checkpoint}.pkl")
        if not os.path.exists(agg_gait_chars_fn):
            print(f"loading gait char files..")
            aggregated_gait_chars=[]
            for fn in tqdm.tqdm(os.listdir(args.agg_gait_chars_pickle_folder)):
                if '.pkl' in fn and str(this_checkpoint) in fn:
                    _fn = os.path.join(args.agg_gait_chars_pickle_folder, fn)
                    # print(f"loading {_fn}")
                    with open(_fn, 'rb') as f:
                        aggregated_gait_chars.append(pickle.load(f))
                    if 4 in args.tasks: 
                        sa_fn = os.path.join(args.agg_gait_chars_pickle_folder, fn[:fn.rfind('000')+3]+'_stateactions.npy')
                        print(f"loading state actions np: {sa_fn}")
                        aggregated_gait_chars[-1]['stateactions_var'] = np.var(np.load(sa_fn))
                    # print('hi')
            print(f"Done loading individual data.  Writing new aggregated file to save time next time: {agg_gait_chars_fn}")
            with open(agg_gait_chars_fn, 'wb') as f:
                pickle.dump(aggregated_gait_chars, f)
        else:        
            print(f"aggregated gait characteristics file found.   Loading...  {agg_gait_chars_fn}")
            with open(agg_gait_chars_fn, 'rb') as f:
                aggregated_gait_chars = pickle.load(f)
            print(f"file loaded.")


        for d in aggregated_gait_chars:  # extract the integer checkpoint from the messy checkpoint string
            cp = d['checkpoint']
            d['cp_int'] = int(cp[cp[:cp.find('000')].rfind('-')+1 :])

        for d in aggregated_gait_chars:  # extract the gamma value
            s = d['run_name']
            start = s.find('gamma') + 5
            end=s[9:].find('_')
            d['gamma'] = float(s[start:start+end])

        unique_run_names = []
        for d in aggregated_gait_chars:
            if d['run_name'] not in unique_run_names and d['run_name'] not in EXCLUDE_RUN_NAMES:
                unique_run_names.append(d['run_name'])

        unique_gamma_values = []
        for d in aggregated_gait_chars:
            if d['gamma'] not in unique_gamma_values:
                unique_gamma_values.append(d['gamma'])
        unique_gamma_values.sort()

        labels=aggregated_gait_chars[0]['gait_char_desc']

        def reshape_into_np(checkpoint):
            # make into one gait rect.  add the occasional zero to fill in where there are a different number of strides left and right
            
            # first find the total length of a looooong concatenation of all these vectors
            num_features = len(aggregated_gait_chars[0]['gait_chars'][0])
            total_length = 0
            for sequence in checkpoint['gait_chars']:
                feat_length = 0
                for feature in sequence:
                    if len(feature) > feat_length:
                        feat_length = len(feature)
                total_length += feat_length

            np_long_sequence = np.zeros((num_features, total_length))
            last_index = 0
            for sequence in checkpoint['gait_chars']:
                
                feat_length = 0
                # get rect size
                for feature in sequence:
                    if len(feature) > feat_length:
                        feat_length = len(feature)                   
                np_rect = np.zeros((num_features, feat_length))
                #now copy the feats into this array
                for idx, feature in enumerate(sequence):
                    np_rect[idx, 0:len(feature)] = np.array(feature, dtype=np.float64 )
                # now copy that array into the long array
                np_long_sequence[:, last_index:last_index+feat_length] = np_rect
                last_index += np_rect.shape[1]
            return np_long_sequence

        def get_gait_stats(checkpoint):
            # make into one gait rect.  add the occasional zero to fill in where there are a different number of strides left and right
            
            # first find the total length of a looooong concatenation of all these vectors
            num_features = len(aggregated_gait_chars[0]['gait_chars'][0])

                

            return np_long_sequence

        # for each unique run name, plot gait characteristics stats vs checkpoint iterations
        if 1 in args.tasks:
            results_folder = args.results_folder +'/mode1'
            make_folder(results_folder)

            for u in unique_run_names:
                fig, axs = plt.subplots()
                fig.set_size_inches(10,10)
                checkpoints = []
                for d in aggregated_gait_chars:
                    if d['run_name'] == u:
                        checkpoints.append(d)
                #plot for each checkpoint
                for checkpoint in checkpoints:
                    np_long_sequence = reshape_into_np(checkpoint)
                    
                    # gait_means = np.mean(gait_chars)#, axis=0)
                    gait_var = np.var(np_long_sequence)#, axis=0)
                    # axs.bar(list(range(len(labels))), height=gait_means, yerr=gait_stds, label=checkpoint['run_name'], capsize=10.0)
                    axs.bar(x=checkpoint['cp_int'], height=gait_var, width = 10.0, label=u)

                axs.set_title(f"Population Gait Characteristics Var \n {u}")
                # axs.set_xticks(range(len(labels)))
                # axs.set_xticklabels(labels, rotation=-45)
                axs.grid()
                fig.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(results_folder, f"{u}.png"))


        # plot gait variance vs gamma  (with single variance value)
        if 2 in args.tasks: 
            results_folder = args.results_folder + '/mode2'
            make_folder(results_folder)

            gait_vars={}
            for g in unique_gamma_values:
                gait_vars[g] = []

            for checkpoint in aggregated_gait_chars:
                if checkpoint['cp_int']==this_checkpoint:
                    np_long_sequence = reshape_into_np(checkpoint)
                    gait_vars[checkpoint['gamma']].append(
                        ( np.var(np_long_sequence), np.cov(np_long_sequence), checkpoint['run_name'] )
                        )

            # plot variances
            fig, axs = plt.subplots()
            # fig.set_size_inches(10,10)
            TEXT_OFFSET = 0.1
            y=[]
            for gamma, values in gait_vars.items():
                xx=[gamma]*len(values)
                yy=[x[0] for x in values]
                rr=[x[2][x[2].rfind('wCM_')+4:] for x in values]
                axs.scatter(x=xx, y=yy, color='grey')
                for t in range(len(xx)):
                    axs.text(x=xx[t]+TEXT_OFFSET, y=yy[t], s=f"{yy[t]:.0f} ({rr[t]})", color='grey')
                mean = np.mean(np.array([x[0] for x in values]))
                axs.scatter(x=[gamma], y=mean, color='red')
                y.append(mean)
            # trendline
            x = list(gait_vars.keys())
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axs.plot(x, p(x), "r--")

            axs.set_title(f"Variance of Population Gait Characteristics vs Gamma\n at cPOET iteration {this_checkpoint}\n Multiple populations (grey) w/ Trendline (red) \n Trendline coefs:{z}")
            axs.set_xticks(unique_gamma_values)
            axs.set_xticklabels([str(x) for x in unique_gamma_values], rotation=-45)
            # axs.set_ylim([0,140])
            axs.grid()
            # fig.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(results_folder, f"pop_gait_variance_vs_gamma_cp{this_checkpoint}.png"))


            # plot covariances
            # first find number of rows
            num_rows=1
            for k, v in gait_vars.items():
                if len(v) > num_rows:
                    num_rows = len(v)
                    print(num_rows, len(v))
            
            fig, axs = plt.subplots(max(num_rows,2), len(list(gait_vars.keys())))

            # set all plot to invisible at first, to keep un-drawn ones from being visible
            for row in range(axs.shape[0]):
                for col in range(axs.shape[1]):
                    axs[row,col].set_visible(False)

            # find the lowest and highest values in the dataset and define min and max for IMshow
            # num_rows=1
            # for k, v in gait_vars.items():
            #     if len(v) > num_rows:
            #         num_rows = len(v)
            #         print(num_rows, len(v))


            MIN=0
            MAX=350
            for col, (k,v) in enumerate(gait_vars.items()):
                # print(col, k, v)
                for row, var_covs in enumerate(v):
                    # print(col, row)
                    im=axs[row, col].imshow(var_covs[1], vmin=MIN, vmax=MAX)
                    axs[row, col].axis('off')
                    axs[row, col].set_visible(True)
                    # axs[row, col].set_title(f"{row} {col}")
                    # print(v[row][1])
                    # print('hi')
            fig.suptitle(f"Gait Characteristics Covariance Matrices \n Populations (columns) vs Gamma ({str(unique_gamma_values)})")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            fig.savefig(os.path.join(results_folder, f"covariance_matrices_{this_checkpoint}.png"))





        # plot individual gait characteristic variances vs gamma 

        if 3 in args.tasks: 
            results_folder = args.results_folder + '/mode3'
            make_folder(results_folder)
            num_features = len(aggregated_gait_chars[0]['gait_chars'][0])
            for ind in range(num_features):
                gait_vars={}
                for g in unique_gamma_values:
                    gait_vars[g] = []

                for checkpoint in aggregated_gait_chars:
                    if checkpoint['cp_int']==this_checkpoint:
                        np_long_sequence = reshape_into_np(checkpoint)
                        gait_vars[checkpoint['gamma']].append((np.var(np_long_sequence[ind,:]), checkpoint['run_name']))

                # fig, axs = plt.subplots()
                # fig.set_size_inches(10,10)

                # y=[]
                # for gamma, values in gait_vars.items():
                #     axs.scatter(x=[gamma]*len(values), y=values, color='grey')
                #     mean = np.mean(np.array(values))
                #     axs.scatter(x=[gamma], y=mean, color='red')
                #     y.append(mean)
                # # trendline
                # x = list(gait_vars.keys())
                # z = np.polyfit(x, y, 1)
                # p = np.poly1d(z)
                # axs.plot(x, p(x), "r--")

                fig, axs = plt.subplots()
                # fig.set_size_inches(10,10)
                TEXT_OFFSET = 0.1
                y=[]
                for gamma, values in gait_vars.items():
                    xx=[gamma]*len(values)
                    yy=[x[0] for x in values]
                    rr=[x[1][x[1].rfind('wCM_')+4:] for x in values]
                    axs.scatter(x=xx, y=yy, color='grey')
                    for t in range(len(xx)):
                        axs.text(x=xx[t]+TEXT_OFFSET, y=yy[t], s=f"{yy[t]:.0f} ({rr[t]})", color='grey')
                    mean = np.mean(np.array([x[0] for x in values]))
                    axs.scatter(x=[gamma], y=mean, color='red')
                    y.append(mean)
                # trendline
                x = list(gait_vars.keys())
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axs.plot(x, p(x), "r--")

                axs.set_title(f"({ind}) coef:{z} \nVariance of Population Gait Characteristics vs Gamma\n at cPOET iteration {this_checkpoint}\n Multiple populations (grey) w/ Trendline (red)")
                axs.set_xticks(unique_gamma_values)
                axs.set_xticklabels([str(x) for x in unique_gamma_values], rotation=-45)
                # axs.set_ylim([0,250])
                axs.grid()
                # fig.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(results_folder, f"pop_gait_variance_vs_gamma_cp{this_checkpoint}_{ind}.png"))

        # plot state-action variance vs gamma  (with single variance value)
        if 4 in args.tasks: 
            results_folder = args.results_folder + '/mode4'
            make_folder(results_folder)
            sa_vars={}
            for g in unique_gamma_values:
                sa_vars[g] = []

            for checkpoint in aggregated_gait_chars:
                if checkpoint['cp_int']==this_checkpoint:
                    sa_vars[checkpoint['gamma']].append(
                        checkpoint['stateactions_var']
                        )

            # plot variances
            fig, axs = plt.subplots()
            # fig.set_size_inches(10,10)

            y=[]
            for gamma, values in sa_vars.items():
                print(gamma, values)
                axs.scatter(x=[gamma]*len(values), y=values, color='grey')
                mean = np.mean(np.array(values))
                axs.scatter(x=[gamma], y=mean, color='red')
                y.append(mean)
            # trendline
            x = list(sa_vars.keys())
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axs.plot(x, p(x), "r--")

            axs.set_title(f"Sample Variance of State-actions vs Gamma\n at cPOET iteration {this_checkpoint}\n Multiple populations (grey) w/ Trendline (red) \n Trendline coefs:{z}")
            axs.set_xticks(unique_gamma_values)
            axs.set_xticklabels([str(x) for x in unique_gamma_values], rotation=-45)
            # axs.set_ylim([0,140])
            axs.grid()
            # fig.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(results_folder, f"pop_stateaction_variance_vs_gamma_cp{this_checkpoint}.png"))


        # Perform one time computation reducing agg_gait_chars.pkl to intermediate gait statistics file mode5 folder
        if 5 in args.tasks: 
            results_folder = args.results_folder + '/mode5'
            make_folder(results_folder)

            intermediate_results_fn = os.path.join(results_folder, 'intermediate_results.pkl')
            if os.path.exists(intermediate_results_fn):
                print(f"Computing gait characteristics intermediate values pkl...")
                gait_stats={}
                for g in unique_gamma_values:
                    gait_stats[g] = []

                def compute_checkpoint_gait_feature_stats(checkpoint):
                    traj_feat_stats = []
                    for traj in tqdm.tqdm(checkpoint['gait_chars'], leave=False):
                        feat_stats = []
                        for feat in traj:
                            feat_stats.append(np.mean(feat))
                            feat_stats.append(np.std(feat))
                            feat_stats.append(np.max(feat))
                            feat_stats.append(np.min(feat))

                        traj_feat_stats.append(feat_stats)
                    return traj_feat_stats

                
                for checkpoint in tqdm.tqdm(aggregated_gait_chars, leave=False):
                    if checkpoint['cp_int'] == this_checkpoint:
                        envs_gait_stats = compute_checkpoint_gait_feature_stats(checkpoint)
                        gait_stats[checkpoint['gamma']].append( ( envs_gait_stats, checkpoint['run_name'] ) )

                print(f"Saving Computed gait characteristics {intermediate_results_fn}")
                with open(intermediate_results_fn, 'wb') as f:
                    pickle.dump(gait_stats, f)

        # Perform one time computation reducing agg_gait_chars.pkl to intermediate gait statistics file mode6 folder
        if 6 in args.tasks: 
            results_folder = args.results_folder + '/mode6'
            make_folder(results_folder)

            intermediate_results_fn = os.path.join(results_folder, f"intermediate_results_cp_{this_checkpoint}.pkl")
            print(f"Computing (more expressive) gait characteristics intermediate values pkl...")
            gait_stats={}
            for g in unique_gamma_values:
                gait_stats[g] = []

            def compute_checkpoint_gait_feature_stats(checkpoint, desired_length):
                traj_feat_stats = []
                for traj in tqdm.tqdm(checkpoint['gait_chars'], leave=False):
                    feat_stats = []
                    for feat in traj:
                        # make the features all the same length: desired length
                        agg_feat = feat
                        while len(agg_feat) < desired_length:
                            agg_feat += feat
                        feat_stats += agg_feat[:desired_length]


                    traj_feat_stats.append(feat_stats)
                return traj_feat_stats

            
            for checkpoint in tqdm.tqdm(aggregated_gait_chars, leave=False):
                if checkpoint['cp_int'] == this_checkpoint:
                    envs_gait_stats = compute_checkpoint_gait_feature_stats(checkpoint=checkpoint, desired_length=100)
                    gait_stats[checkpoint['gamma']].append( ( envs_gait_stats, checkpoint['run_name'] ) )

            print(f"Saving Computed gait characteristics {intermediate_results_fn}")
            with open(intermediate_results_fn, 'wb') as f:
                pickle.dump(gait_stats, f)


        # plot Vendi Score from similarity measure vs gamma
def cluster_and_vendi_gait_feat_stats(args): 
        from vendi_score import vendi
        # import dbscan

        results_folder = args.results_folder + '/mode5'
        intermediate_results_fn = os.path.join(results_folder, f"intermediate_results_cp_{this_checkpoint}.pkl")

        print(f"Loading Previously Computed gait characteristics pkl {intermediate_results_fn}")
        with open(intermediate_results_fn, 'rb') as f:
            gait_stats = pickle.load(f)

        # each entry under each gamma value is of size population x 10k

        def cos_sim(a, b):
            from numpy import dot 
            from numpy.linalg import norm

            cs = dot(a, b)/(norm(a) * norm(b))
            return cs

        fig, axs = plt.subplots()
        for gamma, v in gait_stats.items():
            for pop_id in range(len(v)):
                gait_stats[gamma][pop_id] = list(gait_stats[gamma][pop_id])
                # print(gamma, pop_id)

                individual_gait_stats=gait_stats[gamma][pop_id][0]
                pop_gait_vendi_scores=[]
                for i in range(int(len(individual_gait_stats)/10000)):
                    start = i*10000
                    end = start + 10000

                    X = np.array(individual_gait_stats[start:end])  # samples by features matrix
                    
                    # vs =        vendi.score(X, cos_sim) # calls similarity function, populates sim matrix K
                    # X_vs =      vendi.score_X(X)        # 
                    dual_vs =   vendi.score_dual(X)

                    # now find the distribution of stability with each env held out
                    # stability = []
                    # for j in tqdm.tqdm(range(10000), leave=False):
                    #     data = np.concatenate((X[:j], X[j+1:]))
                    #     stability.append(vendi.score_dual(data) - dual_vs)
                    # print(dual_vs)
                    axs.scatter(gamma, dual_vs)
                    pop_gait_vendi_scores.append(dual_vs)
                
                gait_stats[gamma][pop_id].append(pop_gait_vendi_scores)
        axs.grid(True)
        axs.set_title(f"Vendi Score over gait feature statistics \n for individual agents over CM environments")
        fig.savefig(os.path.join(results_folder, f"vds_vs_gamma.png"))

        # Now look at the VS over gait features of all agents over a single environment


        env_specific_vendi_scores = []
        for env_index in tqdm.tqdm(range(10000)):  # this is the specific environment index
                    
            feature_vectors = []
            for gamma, v in gait_stats.items():
                for pop_id in range(len(v)):
                    gait_stats[gamma][pop_id] = list(gait_stats[gamma][pop_id])
                    # print(gamma, pop_id)

                    individual_gait_stats=gait_stats[gamma][pop_id][0]
                    for i in range(int(len(individual_gait_stats)/10000)):
                        start = i*10000
                        addr = start + env_index
                        feature_vectors.append(individual_gait_stats[addr])


            X = np.array(feature_vectors)  # samples by features matrix
            dual_vs =   vendi.score_dual(X)
            env_specific_vendi_scores.append(dual_vs)
        fig, axs = plt.subplots()
        # print(dual_vs)
        axs.scatter(range(10000), env_specific_vendi_scores)
      
        axs.grid(True)
        axs.set_title(f"Vendi Scores of all agents (gait feature statistics) \n vs specific CM environments")
        fig.savefig(os.path.join(results_folder, f"all_agents_vs_vs_envs.png"))
        print(f"Done.")




        # plot Vendi Score from similarity measure vs gamma
def cluster_and_vendi_gait_feat_stats_more_expressive_feats(args): 
    from vendi_score import vendi

    results_folder = args.results_folder + '/mode6'
    for this_checkpoint in args.checkpoints:
        intermediate_results_fn = os.path.join(results_folder, f"intermediate_results_cp_{this_checkpoint}.pkl")

        print(f"Loading Previously Computed gait characteristics pkl {intermediate_results_fn}")
        with open(intermediate_results_fn, 'rb') as f:
            gait_stats = pickle.load(f)

        # each entry under each gamma value is of size population x 10k

        fig, axs = plt.subplots()
        for gamma, v in tqdm.tqdm(gait_stats.items(), leave=False):
            for pop_id in tqdm.tqdm(range(len(v)), leave=False):
                gait_stats[gamma][pop_id] = list(gait_stats[gamma][pop_id])
                # print(gamma, pop_id)

                individual_gait_stats=gait_stats[gamma][pop_id][0]
                pop_gait_vendi_scores=[]
                for i in tqdm.tqdm(range(int(len(individual_gait_stats)/10000))):
                    start = i*10000
                    end = start + 10000

                    X = np.array(individual_gait_stats[start:end])  # samples by features matrix

                    dual_vs = vendi.score_dual(X)

                    axs.scatter(gamma, dual_vs)
                    pop_gait_vendi_scores.append(dual_vs)
                
                gait_stats[gamma][pop_id].append(pop_gait_vendi_scores)
        axs.grid(True)
        axs.set_title(
            f"Vendi Score over gait feature trajectories \n for individual agents over CM set of environments\n vs Gamma\n" + \
                f"(at POET checkpoint {this_checkpoint})"
            )
        axs.set_ylabel(f"Vendi Score")
        axs.set_xlabel(f"Gamma")
        fig.tight_layout()
        fig.savefig(os.path.join(results_folder, f"vds_vs_gamma_cp{this_checkpoint}.png"))

        # now straighten out the plotting so agents from the same run/pop are the same color
        fig, axs = plt.subplots()
        fig.set_size_inches(10,8)
        for gamma, v in gait_stats.items():
            for pop_id in range(len(v)):

                individual_gait_stats=gait_stats[gamma][pop_id][0]
                pop_gait_vendi_scores=[]
                for i in range(int(len(individual_gait_stats)/10000)):
                    start = i*10000
                    end = start + 10000
                vs_scores= gait_stats[gamma][pop_id][2]
                axs.scatter([gamma]*len(vs_scores), vs_scores)
                

        axs.grid(True)
        axs.set_title(
            f"Vendi Score over gait feature trajectories \n for individual agents over CM set of environments\n vs Gamma\n" + \
                f"(at POET checkpoint {this_checkpoint})"
            )
        axs.set_ylabel(f"Vendi Score")
        axs.set_xlabel(f"Gamma")
        fig.tight_layout()
        fig.savefig(os.path.join(results_folder, f"_vds_vs_gamma_cp{this_checkpoint}.png"))



        print(f"Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agg_gait_chars_pickle_folder',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/aggregated_gait_chars'
    )
    parser.add_argument(
        '--results_folder',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/agg_gait_chars_results'
    )
    parser.add_argument('--tasks', nargs='+', default=[6])
    parser.add_argument('--checkpoints', nargs='+', default=[2000])
    args = parser.parse_args()

    # main(args)
    # cluster_and_vendi_gait_feat_stats(args)
    cluster_and_vendi_gait_feat_stats_more_expressive_feats(args)