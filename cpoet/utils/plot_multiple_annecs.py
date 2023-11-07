import os, json
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
    parser.add_argument("--run_names", type=str, required=False, nargs='+', default=[
        # 'baseline_wCM_14nov_seed24582923',
        'baseline_wCM_17nov_seed24582924',
        # 'icm_gamma0.0_wCM_17nov',

        # 'icm_gamma5.0_wCM_19nov',
        # 'icm_gamma5.0_wCM_28nov',
        # 'icm_gamma5.0_wCM_29nov',

        # 'icm_gamma7.5_wCM_21nov',
        # 'icm_gamma7.5_wCM_2dec',
        # 'icm_gamma7.5_wCM_30nov_a',

        # 'icm_gamma10.0_wCM_8nov',
        # 'icm_gamma10.0_wCM_28nov',
        'icm_gamma10.0_wCM_30nov',

        # 'icm_gamma15.0_wCM_28nov',
        # 'icm_gamma15.0_wCM_1dec',
        # 'icm_gamma15.0_wCM_2dec_a',

        # 'icm_gamma20.0_wCM_8nov',
        # 'icm_gamma20.0_start100_19nov',
        # 'icm_gamma20.0_wCM_2dec',
        
        # 'icm_gamma25.0_wCM_28nov',
        # 'icm_gamma25.0_wCM_4dec',
        # 'icm_gamma25.0_wCM_4dec_a',

        # 'icm_gamma30.0_wCM_16nov',
        # 'icm_gamma30.0_wCM_4dec',
        # 'icm_gamma30.0_wCM_4dec_a',

        ])
    parser.add_argument("--annecs_data_filename", type=str, required=False, default='annecsVsIterations.json')
    parser.add_argument("--output_figure_filename", type=str, required=False, default='multiAnnecsVsIterations.png')
    args=parser.parse_args()
    return args

def main():
    """
    Given a list of run folders with annecsVsIterations.json files, plot them against one another for comparison. 

    """
    args = parse_args()
    if args.run_names == 'all':
        run_names = os.listdir(args.output_dir)
        for run_name in run_names:
            if not os.path.isdir(os.path.join(args.output_dir, run_name)):
                run_names.remove(run_name)
    else:
        run_names = args.run_names

    run_names.sort()
    labels = [r"baseline ePOET ($\zeta$=0.0)", r"Curious POET   ($\zeta$=10.0)"]
    import matplotlib 
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams.update({'font.family': 'serif'})


    fig, axs = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(15, 7)
    for label_idx, run_name in enumerate(run_names):
        print(f"opening run name: {run_name}")
        with open(os.path.join(args.output_dir, run_name,  args.annecs_data_filename), 'r') as f:
            data = json.load(f) 

        # plot individually
        
        axs.plot(data['iteration'], data['annecs'], label=labels[label_idx])
    # axs.set_title(f"ANNECS vs Training Iterations")
    axs.grid(True)
    axs.legend()
    # axs.set_xlim((0,4000))
    # axs.set_ylim((0, 20))
    

    plt.savefig(os.path.join(args.output_dir, args.output_figure_filename))
    print('plot created')


    print('done')

        
if __name__ == "__main__":
    main()