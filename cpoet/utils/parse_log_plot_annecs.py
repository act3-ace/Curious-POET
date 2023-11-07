import os, json
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False, default="/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM")
    parser.add_argument("--run_name", type=str, required=False, default = "icm_gamma10.0_wCM_8marA")  #'all' to run on all runs_folders
    parser.add_argument("--latest_log_filename", type=str, default=None) 
    parser.add_argument("--output_data_filename", type=str, required=False, default='annecsVsIterations.json')
    parser.add_argument("--output_figure_filename", type=str, required=False, default='annecsVsIterations.png')
    args=parser.parse_args()
    return args

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_longest_log_filename(args, run_name) -> str:
    listdir = os.listdir(os.path.join(args.output_dir, run_name))
    logs = [a for a in listdir if 'run.log' in a]
    logs.sort(key=len)
    return logs[-1]

def recursively_extract_log_filenames(args, run_name) -> [str]:
    # in retrospect, this could have been done by simply sorting all the log filenames by length...
    listdir = os.listdir(os.path.join(args.output_dir, run_name))

    ret = ['run.log']
    if not args.latest_log_filename is None:
        log = args.latest_log_filename
    else:
        log = get_longest_log_filename(args, run_name)

    if log == ret[0]:
        return ret
    starts = find(log, '[')
    starts.reverse()
    ends = find(log, ']')
    assert len(starts) == len(ends)

    for i in range(len(starts)):
        inside_brackets = log[starts[i]+1:ends[i]]
        fn = inside_brackets+".resume_run.log"
        ret.append(fn)
        assert fn in listdir
        # print(ret)
    ret.append(log)
    return ret

def main():
    """
    Given the last run log from a series of resumes, open all the logs (starting with run.log) in sequence
    and parse out the ANNECS vs iteration values.   

    """
    args = parse_args()
    if args.run_name == 'all':
        run_names = []
        for run_name in os.listdir(args.output_dir):
            if os.path.isdir(os.path.join(args.output_dir, run_name)) and 'gamma' in run_name:
                run_names.append(run_name)
    else:
        run_names = [args.run_name]
    run_names.sort()

    for run_name in run_names:
        print(f"processing run name: {run_name}")
        logs = recursively_extract_log_filenames(args, run_name)
        print(f"log files")
        print(logs)
    
        iteration = []
        annecs = []
        latest_annecs_value = 0
        # this will need to handle overlapping iterations between log files
        for fn in logs:
            print(fn)
            filename = os.path.join(args.output_dir, run_name, fn)
            with open(filename, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if 'added to ANNECS:' in line and 'True' in line:
                        latest_annecs_value += 1
                        print(f"latest_annecs_value: {latest_annecs_value}")
                    if 'Iter=' in line:
                        start = line.find('Iter=')+5
                        end = line.find(' ', start)
                        latest_iter = int(line[start:end])

                            
                        if latest_iter == 0 or latest_iter != iteration[-1]: #new iter number(sometimes there are repeats)
                            iteration.append(latest_iter)
                            

                            annecs.append(latest_annecs_value)
        assert len(iteration) == len(annecs)
        with open(os.path.join(args.output_dir, run_name,  args.output_data_filename), 'w+') as f:
            json.dump({'iteration': iteration, 'annecs':annecs}, f) 

        # plot individually
        fig, axs = plt.subplots(1,1)
        axs.plot(iteration, annecs)
        axs.set_title(f"ANNECS vs Training Iterations \n Run name: {run_name}")
        axs.grid(True)
    
        plt.savefig(os.path.join(args.output_dir, run_name,  args.output_figure_filename))
        print('plot created')


    print('done')

        
if __name__ == "__main__":
    main()