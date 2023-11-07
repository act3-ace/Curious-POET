import os
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
    parser.add_argument("--run_names", type=str, nargs='+', required=False, default = ["all"])
    parser.add_argument("--log_filenames", type=str, nargs='+', default=['run.log'])
    parser.add_argument("--output_figure_filename", type=str, required=False, default='population_extrinsic_reward.png')
    parser.add_argument("--solved_threshold", type=float, required=False, default=230.0)
    args=parser.parse_args()
    return args


def main():
    """
    Parse the stdout from a curious POET run gathering and plotting Rextrinsic 

    """
    args = parse_args()

    if 'all' in args.run_names[0]:
        args.run_names = []
        for f in os.listdir(args.output_dir):
            if 'icm_gamma' in f:
                if not os.path.exists(os.path.join(args.output_dir, f, 'population_extrinsic_reward.png')):
                    args.run_names.append(f)

    for run_name in args.run_names:

        output_filename = os.path.join(args.output_dir, run_name,  args.output_figure_filename)
        iteration = 0
        rext = []
        # this will need to handle overlapping iterations between log files
        for fn in args.log_filenames:
            filename = os.path.join(args.output_dir, run_name, fn)
            with open(filename, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if 'Iter=' in line and 'theta_mean' in line:
                        start = line.find('theta_mean ') + len('theta_mean') + 1
                        score = float(line[start:start+8])

                        iter_start = line.find('Iter=') + len('Iter=')
                        iter_stop = line.find(' ', iter_start)
                        iteration = int(line[iter_start:iter_stop])

                        if len(rext) < iteration:
                            while len(rext) < iteration:
                                rext.append(0)

                        if len(rext) == iteration:
                            rext.append(score)
                        else:
                            if score > rext[iteration]:
                                rext[iteration] = score
                    

        # find first training iteration to succeed
        def index_of_first(lst, threshold):
            for i, v in enumerate(lst):
                if v > threshold:
                    return i
            return None

        success_iter = index_of_first(rext, args.solved_threshold)

        # now plot
        
        fig, axs = plt.subplots(2,1)
        axs[0].plot(rext)
        axs[0].set_title(f"Population (max) Extrinsic Reward vs Training Iterations \n Run name: {run_name}")
        axs[0].grid(True)
        axs[0].set_ylim([0, 400])
        for i in range(500, len(rext), 500):
                axs[0].plot(i, rext[i], 'ro')
                axs[0].text(i, 350, f"{int(rext[i])}", ha='center')


        if success_iter:
            axs[1].plot(rext)
            axs[1].set_xlim([0,success_iter+100])
            axs[1].grid(True)
            axs[1].plot(success_iter, args.solved_threshold, 'ro')
            axs[1].text(success_iter - 5, args.solved_threshold+ 20 , f"Rext >= {int(args.solved_threshold)} at iteration {success_iter}", ha='right')
        else:
            axs[1].text(.10, .5 , f"Population hasn't yet achieved score {args.solved_threshold}")
            print("Population hasn't yet solved environment.")    
        plt.savefig(output_filename)
        print('plot created')

        
if __name__ == "__main__":
    main()