import os
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
    parser.add_argument("--run_name", type=str, required=False, default = "icm_gamma10.0_wCM_8marA")
    parser.add_argument("--log_filenames", type=str, nargs='+', default=['logs/1_train_icm_stdout.log'])
    parser.add_argument("--output_figure_folder", type=str, required=False, default='logs')
    parser.add_argument("--solved_threshold", type=float, required=False, default=230.0)
    args=parser.parse_args()
    return args


def main():
    """
    Parse the stdout from a curious POET ICM trainer, gathering and plotting Rextrinsic and Rintrinsic 
    also plot the icm training losses in a separate figure.
    """
    args = parse_args()
    output_folder = os.path.join(args.output_dir, args.run_name,  args.output_figure_folder)
    iterations = []
    rext = []
    rint = []
    forward_loss = []
    inverse_loss = []
    overall_loss = []
    # this will need to handle overlapping iterations between log files
    for fn in args.log_filenames:
        filename = os.path.join(args.output_dir, args.run_name, fn)
        with open(filename, 'r') as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'Sent an inference response!  ' in line:
                rint.append(float(line[line.find('Ri = ')+5: line.find(', ')]))
                rext.append(float(line[line.find('Re=')+3: line.find('@')-1]))
                iterations.append(int(line[line.find('POET iter: ')+11:-1]))
            if 'step: ' in line:
                forward_loss.append(float(line[line.find('forward: ')+9: line.find(' inverse')]))
                inverse_loss.append(float(line[line.find('inverse: ')+9: line.find(' loss')]))
                overall_loss.append(float(line[line.find('loss: ')+6:-1]))


    # plot losses

    assert len(overall_loss) == len(inverse_loss) == len(forward_loss)
    fig, axs = plt.subplots(1,1)
    axs.plot(range(len(inverse_loss)), inverse_loss, label='inverse loss')
    axs.plot(range(len(forward_loss)), forward_loss, label='forward loss')
    # axs.plot(range(len(overall_loss)), overall_loss)
    axs.set_yscale('log')
    axs.set_xlabel('ICM Training Steps')
    axs.set_title(f"ICM training losses \n Run name: {args.run_name}")
    axs.grid(True)

    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, 'icm_losses.png'))
    print('Losses plot created')

    if len(rint) > 0:
        # plot rewards
        num_plots=2
        if max(rint[200:])> 10:
            num_plots = 3

        fig, axs = plt.subplots(num_plots,1)
        axs[0].plot(iterations, rext)
        axs[0].set_title(f"Population Extrinsic Reward vs POET Iterations \n Run name: {args.run_name}")
        axs[0].grid(True)


        axs[1].plot(iterations, rint)
        axs[1].set_ylim([0, 1.2*max(rint[200:])])
        axs[1].set_title(f"Population Intrinsic Reward vs POET Iterations \n Run name: {args.run_name}")
        axs[1].grid(True)

        if num_plots == 3:
            axs[2].plot(iterations, rint)
            axs[2].set_ylim([0, 10])
            axs[2].set_title(f"Population Intrinsic Reward vs POET Iterations \n Run name: {args.run_name}")
            axs[2].grid(True)


        fig.tight_layout()
        plt.savefig(os.path.join(output_folder, 'rewards.png'))
        print(' rewards plot created')



        
if __name__ == "__main__":
    main()