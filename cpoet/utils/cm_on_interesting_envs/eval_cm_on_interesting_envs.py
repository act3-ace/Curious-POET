import json, os
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})

def get_gamma_list(args):
    gamma_list = []
    for run, _ in args.run_names:
        if 'baseline' in run:
            gamma_list.append(str(0.0))
        else:
            s0 = run[run.find('gamma')+5: run.find('_wCM')]
            s1 = run[run.find('gamma')+5: run.find('_start')]
            if len(s0) < len(s1):
                gamma_list.append(s0)
            else:
                gamma_list.append(s1)
    return gamma_list


def get_agg_gammas(gammas):
    # return a list of unique gamma strings in ascending numerical order
    # given ["0.0", "0.0", "5.0", "7.5", "10.0", "10.0", "10.0", ]
    # return ["0.0", "5.0", "7.5", "10.0"]
    ret = []
    for g in gammas:
        if not g in ret:
            ret.append(g)
    return ret

N_ENVS=100
BP_SOLVE=230.0

# open up all the kjnm_databricks for appropriate run/checkpoint/coveragemetric folders,
# down-select only those environments that are 'interesting', i.e., solved some, but not all agents.
# then compute a coverage metric score for each population (at cp 2000) using this set of environments.
# make a plot of mean cm score vs zeta (used to be gamma)



def main(args):
    # first down select only interesting envs and make a mask
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, f"stdout_{int(args.too_hard_fraction*100)}_{int(args.too_easy_fraction*100)}.txt"), 'w+') as g:
        env_solve_counts=[0]*N_ENVS
        env_solve_matrix = np.zeros([len(args.run_names), N_ENVS])
        num_agents = 0

        for run_idx, (name, cp) in enumerate(args.run_names):
            fn = os.path.join(args.experiments_folder, name, cp, args.coverage_metric_name, f"kjnm_databrick.npy")
            results = np.load(fn)[0][0][:N_ENVS][:]  # addressing specific to last 100?
            num_agents += results.shape[1]
            for i in range(N_ENVS):
                env_solve_counts[i] += sum(results[i,]>BP_SOLVE)
                env_solve_matrix[run_idx,i] += sum(results[i,]>BP_SOLVE)
        print(f"There are {num_agents} total agents across all {len(args.run_names)} experiment populations(runs).", file=g)
        frac_agents_solved = [0.0]*N_ENVS
        
        for i in range(len(frac_agents_solved)):
            frac_agents_solved[i] = env_solve_counts[i]/num_agents
        NUM_BINS=50
        counts, bins = np.histogram(frac_agents_solved, bins=[(1/NUM_BINS)*x for x in range(NUM_BINS+1)])
        fig, axs = plt.subplots(1,1)
        plt.hist(bins[:-1], bins, weights=counts)
        axs.set_xlabel(f"By this fraction of {num_agents} total agents")
        axs.set_ylabel(f"Num of Envs solved")
        axs.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_folder, 'env_solution_histogram.png'))

        print(f"{sum(np.array(frac_agents_solved)<= args.too_hard_fraction)} of {N_ENVS} environments are solved by <={args.too_hard_fraction*100}% of agents", file=g)
        print(f"{sum(np.array(frac_agents_solved)>= args.too_easy_fraction)} of {N_ENVS} environments are solved by >={args.too_easy_fraction*100}% of agents", file=g)

        # now ocmpute a coverage metric for each population, and each super-population with the same zeta
        # and for that we need a masked array to mask out the un-interesting envs
        mask = []
        mask_too_hard = []
        mask_too_easy = []
        for i in range(N_ENVS):
            mask.append(frac_agents_solved[i]<= args.too_hard_fraction or frac_agents_solved[i]>= args.too_easy_fraction)
            mask_too_easy.append(frac_agents_solved[i]>= args.too_easy_fraction)
            mask_too_hard.append(frac_agents_solved[i]<= args.too_hard_fraction)

        if args.use_this_mask is not None:
            with open(args.use_this_mask, 'rb') as f:
                mask = pickle.load(f)
                print(f"disregard above.  loading interesting environment mask: {args.use_this_mask}", file=g)
        num_int_envs = N_ENVS - sum(np.array(mask))
        print(f"There are {num_int_envs} interesting/remaining environments.", file=g)


        zetas = get_gamma_list(args)
        unique_zetas = get_agg_gammas(zetas)
        cm_scores = []
        for name, cp in args.run_names:
            print(f"{name},  {cp}", file=g)
            fn = os.path.join(args.experiments_folder, name, cp, args.coverage_metric_name, f"kjnm_databrick.npy")
            results = np.load(fn)[0][0][:N_ENVS][:]  # addressing specific to last 100?
            # interesting cm score is the fraction of interesting envs solved by at least one agent
            int_env_solved_count = 0
            for i in range(N_ENVS):
                if not mask[i]:   # remember True = masked out (not interesting)
                    int_env_solved_count += int(sum(results[i,]>BP_SOLVE)>1)
            cm_scores.append(int_env_solved_count/num_int_envs)
            print(f"{int_env_solved_count/num_int_envs}", file=g)

        agg_cm_scores = []
        agg_cm_stds = []
        for uz in unique_zetas:
            acc = []
            for idx, z in enumerate(zetas):
                if "gamma"+uz+"_" in args.run_names[idx][0] or \
                    ("baseline" in args.run_names[idx][0] and uz == "0.0"):
                    acc.append(cm_scores[idx])
            agg_cm_scores.append(np.mean(acc))
            agg_cm_stds.append(np.std(acc))

        fig, axs = plt.subplots(1,1)
        plt.errorbar(unique_zetas[0], agg_cm_scores[0], yerr=agg_cm_stds[0], capsize=10, color='blue', marker='o')
        plt.errorbar(unique_zetas[1:], agg_cm_scores[1:], yerr=agg_cm_stds[1:], capsize=10, color='green', marker='o')

        axs.set_xlabel(f"$\zeta$")
        axs.set_ylabel(f"Coverage Metric Score")
        axs.set_ylim([0, 1.1])
        axs.grid()
        axs.legend(['ePOET', 'Curious POET'], reverse=True)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_folder, f"cm_vs_zeta_{int(args.too_hard_fraction*100)}_{int(args.too_easy_fraction*100)}.png"))

        print(f"Aggregate scores vs gamma", file=g)
        print(agg_cm_scores, file=g)
        print(f"Aggregate stds vs gamma", file=g)
        print(agg_cm_stds, file=g)
        print(f"now normalized by baseline:", file=g)
        for i in agg_cm_scores:
            print(f"{i/agg_cm_scores[0]:0.2f}", file=g)

        # now make a nice plot of which envs are interesting
        with open(args.terrain_object_filename, 'rb') as f:
            envs = pickle.load(f)
        last_envs = envs[99:len(envs)+1:100]
        
        fig, axs = plt.subplots(10,10)
        fig.set_size_inches(10,10)
        index=0
        for i in range(10):
            for j in range(10):
                terrain = last_envs[index].xy()['y']
                terrain = (terrain-np.mean(terrain)) / np.std(terrain)
                axs[i][j].plot(range(200), terrain)
                # axs[i][j].set_ylim([-100, 100])
                axs[i][j].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
                axs[i][j].tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left=False,      # ticks along the bottom edge are off
                    right=False,         # ticks along the top edge are off
                    labelleft=False) # labels along the bottom edge are off
                if mask[index]:
                    if mask_too_easy[index]: axs[i][j].set_facecolor('lightgreen')
                    if mask_too_hard[index]: axs[i][j].set_facecolor('lightcoral')
                    
                index += 1
        # fig.tight_layout()
        fig.savefig(os.path.join(args.output_folder, 
            f"interesting_envs_{int(args.too_hard_fraction*100)}_{int(args.too_easy_fraction*100)}.png")
            )

        # now make a bar graph plotting and comparing who beat what envs
        # first let's simplify env_solve_matrix by combining like zeta values together
        simp_env_solve_matrix = np.zeros((len(unique_zetas), N_ENVS))
        for uz_idx, uz in enumerate(unique_zetas):
            accum = np.zeros((N_ENVS))
            for z_idx, z in enumerate(zetas):   
                if z == uz: 
                    accum += env_solve_matrix[z_idx,:]
            simp_env_solve_matrix[uz_idx,:] = accum

        # now remove all masked out envs (leaving interesting ones)
        # temp = np.zeros((len(unique_zetas), N_ENVS-sum(mask)))
        # last_idx = 0
        # for i in range(N_ENVS):
        #     if not mask[i]:
        #         temp[:,last_idx] = simp_env_solve_matrix[:,i]
        #         last_idx += 1
        # simp_env_solve_matrix = temp
        #neither solved
        #only gamma solved
        #baseline & gamma solved
        #only baseline solved
        baseline_only = []
        baseline_and_zeta = []
        only_zeta = []
        neither = []
        l = simp_env_solve_matrix.shape[1]
        for z in range(1, len(unique_zetas)):
            baseline_only.append(sum([list(simp_env_solve_matrix[0,]>0)[i] and list(simp_env_solve_matrix[z,]==0)[i] for i in range(l)]))
            baseline_and_zeta.append(sum([list(simp_env_solve_matrix[0,]>0)[i] and list(simp_env_solve_matrix[z,]>0)[i] for i in range(l)]))
            only_zeta.append(sum([list(simp_env_solve_matrix[0,]==0)[i] and list(simp_env_solve_matrix[z,]>0)[i] for i in range(l)]))
            neither.append(sum([list(simp_env_solve_matrix[0,]==0)[i] and list(simp_env_solve_matrix[z,]==0)[i] for i in range(l)]))

        weight_counts = {
            'ePOET solved':baseline_only,
            'both solved':baseline_and_zeta,
            'Curious POET solved':only_zeta,
            'neither solved':neither,
        }

        fig, ax = plt.subplots()
        bottom = np.zeros((len(unique_zetas)-1))    

        for name, weight_count in weight_counts.items():
            p = ax.bar(unique_zetas[1:], weight_count, 0.5, label=name, bottom=bottom)
            bottom += weight_count
        ax.legend(loc="upper right")
        ax.set_xlabel(r"$\zeta$")
        ax.set_ylabel(f"Proportion of Envs")
        ax.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_folder, 
            f"partition_by_solution_group.png")
            )

        print(weight_counts, file=g)
        with open(os.path.join(args.output_folder, f"mask_{int(args.too_hard_fraction*100)}_{int(args.too_easy_fraction*100)}.pkl"), 'wb') as f:
            pickle.dump(mask, f)
        print('done')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=f"/data/petabyte/poet/curious_poet_paper_dataset/centralized_ICM/_cm_on_interesting_envs_2k_lookback/")
    parser.add_argument('--experiments_folder', default=f"/data/petabyte/poet/curious_poet_paper_dataset/centralized_ICM/")
    parser.add_argument('--use_this_mask', type=str, default="/data/petabyte/poet/curious_poet_paper_dataset/centralized_ICM/_cm_on_interesting_envs/mask_5_95.pkl")
    parser.add_argument("--coverage_metric_name", default=f"fixedset_coverage_metric_only_100th_wRollouts")
    parser.add_argument("--run_names", type=str, required=False, nargs='+', default=[

        ('icm_gamma0.0_wCM_17nov', 'cp-2000'),
        ('icm_gamma0.0_wCM_17feb', 'cp-2000'),
        ('icm_gamma0.0_wCM_21feb', 'cp-2000'),

        ('icm_gamma5.0_wCM_19nov', '[cp-1800]->cp-2000'),
        ('icm_gamma5.0_wCM_28nov', 'cp-2000'),
        ('icm_gamma5.0_wCM_29nov', 'cp-2000'),

        ('icm_gamma7.5_wCM_21nov', '[[cp-250]->cp-1300]->cp-2000'),
        ('icm_gamma7.5_wCM_2dec', 'cp-2000'),
        ('icm_gamma7.5_wCM_30nov_a', 'cp-2000'),
        ('icm_gamma7.5_wCM_21feb', 'cp-2000'),
        ('icm_gamma7.5_wCM_7mar', '[cp-1150]->cp-2000'),

        ('icm_gamma10.0_wCM_8nov', '[[cp-1300]->cp-1550]->cp-2000'),
        ('icm_gamma10.0_wCM_28nov',  'cp-2000'),
        ('icm_gamma10.0_wCM_30nov',  'cp-2000'),  
        ('icm_gamma10.0_wCM_27jan',  'cp-2000'),
        ('icm_gamma10.0_wCM_8mar',  'cp-2000'),  # This sample appears to have been left out of original submission
        ('icm_gamma10.0_wCM_8marA',  'cp-2000'),

        ('icm_gamma15.0_wCM_28nov', 'cp-2000'),
        ('icm_gamma15.0_wCM_1dec', 'cp-2000'),
        ('icm_gamma15.0_wCM_2dec_a', 'cp-2000'),

        ('icm_gamma20.0_wCM_8nov', 'cp-2000'),
        ('icm_gamma20.0_start100_19nov', 'cp-2000'),
        ('icm_gamma20.0_wCM_2dec', 'cp-2000'),
        ])
    
    parser.add_argument("--terrain_object_filename", type=str, required=False, 
        default='/data/petabyte/poet/curious_poet_paper_dataset/centralized_ICM/10000cppns_k100.pkl'
        )
    parser.add_argument("--too_easy_fraction", type=float, default = 0.95)
    parser.add_argument("--too_hard_fraction", type=float, default = 0.05)


    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_args()
    main(
        args=args,

    )
