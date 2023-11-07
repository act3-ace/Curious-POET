import os, pickle
import argparse
import shutil
from matplotlib import pyplot as plt
import numpy as np

MODE=3
EXCLUDE_RUN_NAMES = [

]

def main(args):

    if MODE == 1:
        args.results_folder += '/mode1'
    if MODE == 2:
        args.results_folder += '/mode2'
        CHECKPOINT=2000
    if MODE == 3:
        args.results_folder += '/mode3'
        CHECKPOINT=2000
    if MODE == 4:
        args.results_folder += '/mode4'
        CHECKPOINT=2000

    if os.path.exists(args.results_folder):
        shutil.rmtree(args.results_folder)
    os.makedirs(args.results_folder)
    with open(args.agg_gait_chars_pickle_fn, 'rb') as f:
        aggregated_gait_chars = pickle.load(f)
    
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

    labels=[
            'mean_left_stride_time',
            'std_left_stride_time',
            'mean_right_stride_time',
            'std_right_stride_time',
            'mean_left_stride_x',
            'std_left_stride_x',
            'mean_right_stride_x',
            'std_right_stride_x',
            'lr_stride_ratio',
        ]
    # for each unique run name, plot gait characteristics stats vs checkpoint iterations
    if MODE == 1: 

        for u in unique_run_names:
            fig, axs = plt.subplots()
            fig.set_size_inches(10,10)
            checkpoints = []
            for d in aggregated_gait_chars:
                if d['run_name'] == u:
                    checkpoints.append(d)
            #plot for each checkpoint
            for checkpoint in checkpoints:
                gait_chars = np.array(checkpoint['gait_chars'])
                gait_means = np.mean(gait_chars)#, axis=0)
                gait_var = np.var(gait_chars)#, axis=0)
                # axs.bar(list(range(len(labels))), height=gait_means, yerr=gait_stds, label=checkpoint['run_name'], capsize=10.0)
                axs.bar(x=checkpoint['cp_int'], height=gait_var, width = 10.0, label=u)

            axs.set_title(f"Population Gait Characteristics Var \n {u}")
            # axs.set_xticks(range(len(labels)))
            # axs.set_xticklabels(labels, rotation=-45)
            axs.grid()
            fig.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.results_folder, f"{u}.png"))


    # plot gait variance vs gamma  (with single variance value)
    if MODE == 2: 
        gait_vars={}
        for g in unique_gamma_values:
            gait_vars[g] = []

        for i in aggregated_gait_chars:
            if i['cp_int']==CHECKPOINT:
                gait_vars[i['gamma']].append([np.var(np.array(i['gait_chars']))])

        fig, axs = plt.subplots()
        # fig.set_size_inches(10,10)

        y=[]
        for gamma, values in gait_vars.items():
            axs.scatter(x=[gamma]*len(values), y=values, color='grey')
            mean = np.mean(np.array(values))
            axs.scatter(x=[gamma], y=mean, color='red')
            y.append(mean)
        # trendline
        x = list(gait_vars.keys())
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axs.plot(x, p(x), "r--")

        axs.set_title(f"Variance of Population Gait Characteristics vs Gamma\n at cPOET iteration {CHECKPOINT}\n Multiple populations (grey) w/ Trendline (red) \n Trendline coefs:{z}")
        axs.set_xticks(unique_gamma_values)
        axs.set_xticklabels([str(x) for x in unique_gamma_values], rotation=-45)
        axs.set_ylim([0,140])
        axs.grid()
        # fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.results_folder, f"pop_gait_variance_vs_gamma_cp{CHECKPOINT}.png"))


    # plot individual gait characteristic variances vs gamma 
    # taekaway: The stdev's are quite flat and tend to not be discriminitive
    if MODE == 3: 
        for ind in range(9):
            gait_vars={}
            for g in unique_gamma_values:
                gait_vars[g] = []

            for i in aggregated_gait_chars:
                if i['cp_int']==CHECKPOINT:
                    gait_vars[i['gamma']].append([np.var(np.array(i['gait_chars'])[:,ind])])

            fig, axs = plt.subplots()
            # fig.set_size_inches(10,10)

            y=[]
            for gamma, values in gait_vars.items():
                axs.scatter(x=[gamma]*len(values), y=values, color='grey')
                mean = np.mean(np.array(values))
                axs.scatter(x=[gamma], y=mean, color='red')
                y.append(mean)
            # trendline
            x = list(gait_vars.keys())
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axs.plot(x, p(x), "r--")

            axs.set_title(f"({ind}) coef:{z} \nVariance of Population Gait Characteristics vs Gamma\n at cPOET iteration {CHECKPOINT}\n Multiple populations (grey) w/ Trendline (red)")
            axs.set_xticks(unique_gamma_values)
            axs.set_xticklabels([str(x) for x in unique_gamma_values], rotation=-45)
            axs.set_ylim([0,250])
            axs.grid()
            # fig.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.results_folder, f"pop_gait_variance_vs_gamma_cp{CHECKPOINT}_{ind}.png"))

    # plot gait variance vs gamma  (for selected indices of the gait characteristic)
    # discard standard deviations as they are flat, and not discriminative
    if MODE == 4: 
        gait_vars={}
        for g in unique_gamma_values:
            gait_vars[g] = []

        for i in aggregated_gait_chars:
            if i['cp_int']==CHECKPOINT:
                arr = np.array(i['gait_chars'])
                # gait_vars[i['gamma']].append([np.var(arr[:,[0,2,4,6]])])  # discard everything but means of stride (time and x)
                gait_vars[i['gamma']].append([np.var(arr[:,[0,2,3,5,6]])])  # discard everything with trendline slope magnitude below 0.5

        fig, axs = plt.subplots()
        # fig.set_size_inches(10,10)

        y=[]
        for gamma, values in gait_vars.items():
            axs.scatter(x=[gamma]*len(values), y=values, color='grey')
            mean = np.mean(np.array(values))
            axs.scatter(x=[gamma], y=mean, color='red')
            y.append(mean)
        # trendline
        x = list(gait_vars.keys())
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axs.plot(x, p(x), "r--")

        axs.set_title(f"Variance of Population Gait Characteristics vs Gamma\n at cPOET iteration {CHECKPOINT}\n Multiple populations (grey) w/ Trendline (red) \nTrendline coeffs:{z}")
        axs.set_xticks(unique_gamma_values)
        axs.set_xticklabels([str(x) for x in unique_gamma_values], rotation=-45)
        # axs.set_ylim([0,140])
        axs.grid()
        # fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.results_folder, f"pop_gait_variance_vs_gamma_cp{CHECKPOINT}.png"))

    print('done')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agg_gait_chars_pickle_fn',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/aggregated_gait_chars.pkl'
    )
    parser.add_argument(
        '--results_folder',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/agg_gait_chars_results'
    )
    args = parser.parse_args()
    main(args)