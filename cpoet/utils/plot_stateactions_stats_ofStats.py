import os, pickle
import argparse
import shutil
from matplotlib import pyplot as plt

def main(args):

    if os.path.exists(args.results_folder):
        shutil.rmtree(args.results_folder)
    os.makedirs(args.results_folder)
    with open(args.stateactions_stats_pickle_fn, 'rb') as f:
        state_action_list = pickle.load(f)
    
    for d in state_action_list:  # extract the integer checkpoint from the messy checkpoint string
        cp = d['checkpoint']
        d['cp_int'] = int(cp[cp[:cp.find('000')].rfind('-')+1 :])

    for d in state_action_list:  # extract the gamma value
        s = d['run_name']
        start = s.find('gamma') + 5
        end=s[9:].find('_')
        d['gamma'] = float(s[start:start+end])

    unique_run_names = []
    for d in state_action_list:
        if d['run_name'] not in unique_run_names and not '25jan' in d['run_name'] and not 'diverged' in d['run_name']:
            unique_run_names.append(d['run_name'])

    # for each unique run name, plot stats vs gamma
    fig, axs = plt.subplots()
    fig.set_size_inches(10,10)
    for u in unique_run_names:
        checkpoints = []
        for d in state_action_list:
            if d['run_name'] == u:
                checkpoints.append(d)
        #plot for each checkpoint
        axs.plot([x['cp_int'] for x in checkpoints], [x['sa_var_agg'] for x in checkpoints], label=checkpoints[0]['run_name'])
        print(f"{d['run_name']}")
    
    fig.legend()
    # fig.tight_layout()
    fig.savefig(os.path.join(args.results_folder, 'test.png'))
    print('done')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stateactions_stats_pickle_fn',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/state_action_stats_data.pkl'
    )
    parser.add_argument(
        '--results_folder',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/state_action_stats_results'
    )
    args = parser.parse_args()
    main(args)