import os, pickle, json
import argparse
import shutil
from matplotlib import pyplot as plt
import numpy as np

CHECKPOINT='2000'
EXCLUDE_RUN_NAMES = [

]

def main(args):

    # def make_folder(results_folder):
        # if os.path.exists(results_folder):
        #     shutil.rmtree(results_folder)
        # os.makedirs(results_folder)

    run_names = []
    for fn in os.listdir(args.input_folder):
        if CHECKPOINT in fn:
            icm_name = fn[fn.find('ICM_')+4:fn.find('_inferring')]
            pop_name = fn[fn.find('AGENTS_')+7:fn.rfind('_')]

            if icm_name not in run_names:
                run_names.append(icm_name)
            if pop_name not in run_names:
                run_names.append(pop_name)
    run_names.sort()
    run_names.sort(key=(lambda x: len(x[:x.find('.')])))

    cm_means = np.zeros((len(run_names), len(run_names))) # shape is icm x pop
    cm_stds = np.zeros((len(run_names), len(run_names))) # shape is icm x pop

    for fn in os.listdir(args.input_folder):
        icm_name = fn[fn.find('ICM_')+4:fn.find('_inferring')]
        pop_name = fn[fn.find('AGENTS_')+7:fn.rfind('_')]    
        icm_name_index = run_names.index(icm_name)
        pop_name_index = run_names.index(pop_name)
        with open(os.path.join(args.input_folder, fn), 'r') as f:
            data = json.load(f)

        cm_means[icm_name_index, pop_name_index]=data['mean_ri']
        cm_stds[icm_name_index, pop_name_index]=data['std_ri']

    # list of gammas and dates
    labels=[]
    for run_name in run_names:
        labels.append(
            run_name[run_name.rfind('_wCM_')+5:] + ' ' + \
            run_name[run_name.find('gamma')+5 : run_name.find('.')+2]
        )


    # confusion matrix of means-----------------------------------------------
    fig, axs = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(10,10)
    plt.imshow(cm_means, cmap='plasma')
    axs.set_xticks(np.arange(len(run_names)), labels=labels)
    axs.set_yticks(np.arange(len(run_names)), labels=labels)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    axs.set_ylabel('ICM')
    axs.set_xlabel(f"Population (Checkpoint {CHECKPOINT})")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = axs.text(j, i, int(cm_means[i, j]),
                        ha="center", va="center", color="r")
    
    axs.set_title("Mean Intrinsic Reward confusion matrix")
    plt.colorbar(shrink=0.76)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'ICM_inference_confustion_matrix.png'))


        # confusion matrix of stds-------------------------------------------
    fig, axs = plt.subplots(ncols=1, nrows=1)
    plt.imshow(cm_stds, cmap='plasma')
    axs.set_xticks(np.arange(len(run_names)), labels=labels)
    axs.set_yticks(np.arange(len(run_names)), labels=labels)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    axs.set_ylabel('ICM')
    axs.set_xlabel(f"Population (Checkpoint {CHECKPOINT})")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = axs.text(j, i, int(cm_stds[i, j]),
                        ha="center", va="center", color="r")
    
    axs.set_title("StDev of Intrinsic Rewards confusion matrix")
    plt.colorbar(shrink=0.76)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'std_ICM_inference_confustion_matrix.png'))
    print('done')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/aggregated_ICM_inference_results'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/aggregated_ICM_inference_plots'
    )
    parser.add_argument('--tasks', nargs='+', default=[3])
    args = parser.parse_args()
    main(args)