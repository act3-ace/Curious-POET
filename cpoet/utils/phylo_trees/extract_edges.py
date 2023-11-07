import os
import argparse, json

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--latest_log_filename", type=str, default=None) 
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
    
    listdir = os.listdir(os.path.join(args.output_dir, run_name))

    ret = ['run.log']
    if args.latest_log_filename is None:
        log = get_longest_log_filename(args, run_name)
    else:
        log = args.latest_log_filename

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
        ret.append(fn)
    return ret


def optimizer_created(lines: list, optim_id: str):
    for idx, line in enumerate(lines):
        if 'created!' in line:
            space_idxs = [pos for pos, char in enumerate(line) if char == ' ']
            begin = space_idxs[-2]+1
            end = space_idxs[-1]
            child = line[begin:end]

            if child == optim_id:
                return True
    return False

def get_iteration(line):
    import re
    space_indices = [i.start() for i in re.finditer(' ', line)]
    try: iteration = int(line[line[:space_indices[0]].rfind('=')+1 : space_indices[0]])
    except: iteration = int(line[space_indices[1]+1:line.rfind(',')]) # format for transfer lines
    # annecs = int(line[space_indices[3]+1:-1])
    return iteration

def main():
    """
    Parse the stdout from a curious POET run and create a list of tuples used to generate a phylo tree

    Evolved environments create a tuple (parent, child) where the agent is copied from the parent to the child
    (The parent's weights *may* be modified by an iter or two of training, or they may be copied.  have to check)

    Transfers create tuples too where the agent is again copied from a donor optimizer, replacing the agent in the 
    target optimizer.
    """
    args = parse_args()
    logs = recursively_extract_log_filenames(args, args.run_name)
    for log in logs:
        print(f"parsing logfile: {log}")
        filename = os.path.join(args.output_dir, args.run_name, f"{log}")
        output_filename = os.path.join(args.output_dir, args.run_name,  f"logs/phylo_tuples.json")

        tuples = []
        iteration = 0
        annecs = 0
        with open(filename, 'r') as f:
            lines = f.readlines()
            # first find all tuples from regular env evolution
            for idx, line in enumerate(lines):
                if 'Iter=' in line:
                    iteration = get_iteration(line)
                if "we pick to mutate:" in line:
                    space_idxs = [pos for pos, char in enumerate(line) if char == ' ']
                    begin = line.find("we pick to mutate:")+len("we pick to mutate: ")
                    assert begin-1 in space_idxs
                    end = space_idxs[space_idxs.index(begin-1) + 1]
                    parent = line[begin:end]
                    assert parent in line

                    end = space_idxs[-1]
                    begin = space_idxs[-2]+1
                    child = line[begin:end]
                    assert child in line

                    if optimizer_created(lines, child):
                        tuples.append({
                            'parent':parent, 
                            'child':child, 
                            'link_type':'env_evolution', 
                            'iteration':iteration, 
                            'annecs':annecs
                            })
                        print(f"parent: {parent}, child: {child}, line: {idx}")

            print(f"\nNow find transfers\n\n")
            # now find all tuples from agent transfers
            iteration = 0
            annecs = 0
            for idx, line in enumerate(lines):
                if "'iteration':" in line:
                    iteration = get_iteration(line)
                if "accept_theta_in_" in line and 'do_not_consider_CP' not in line and 'self' not in line:
                    space_idxs = [pos for pos, char in enumerate(line) if char == ' ']
                    begin = space_idxs[-1]+2
                    assert begin-2 in space_idxs
                    if 'proposal' in line:
                        end = -12  # assumes line ends with "_proposal'\n"
                    else:
                        end = -3
                    parent = line[begin:end]
                    assert parent in line
                    assert len(parent)>0

                    end = space_idxs[-1]-2
                    begin = line.find("'accept_theta_in_")+len("'accept_theta_in_")
                    child = line[begin:end]
                    assert child in line
                    assert len(child)>0
                    if not parent == child:
                        new_entry = {
                            'parent':parent, 
                            'child':child, 
                            'link_type':'agent_transfer',
                            'iteration':iteration, 
                            'annecs':annecs
                            }
                        if new_entry not in tuples:
                            tuples.append(new_entry)
                            print(f"parent: {parent}, child: {child}, line: {idx}")
                        # else:
                        #     print('not appending duplicate entry')
                        
                    
                    
    print('Parse complete')
    print(f"Saving to {output_filename}")
    with open(output_filename, 'w+') as f:
        json.dump(tuples, f)


if __name__ == "__main__":
    main()