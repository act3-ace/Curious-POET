import time, os
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import json
import poet_distributed.niches.ES_Bipedal.cppn as cppn
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from utils.evolve_covmet_cppns import load_cppns
import pickle

#####################################
##
## eval_coverage_NvK Metric
##
#####################################
def eval_coverage_NvK(self, threshold=-25, num_cppns=10, max_depth=10):
    '''
    Initial draft of a coverage metric as described here in mattermost: 
    https://chat.git.act3-ace.com/act3/pl/e45kc6gdsjryfqkqeuh67cjecw

    "Given a random environment uniformly drawn from this space of 
    environments, what is the probability that any agent in a population
    of agents gets a score above T." -Castanon

    Create a group of randomly evolved cppn environments of size N.  
    Limit the depth of an evolution sequence to k (max_depth).

    For each environment, evaluate against each agent in both the 
    active and archived populations.  Keep track of whether each environment
    is ever solved.

    Return the ratio of solved environments to total environments. 

    '''

    logger.info('************************ Computing Coverage Metric ************************')
    num_cppns_list = [100, 200, 500, 1000, ]  # overrides for looping experiment
    threshold = 200
    k_list = [10, 20, 35, 50, 75, 100]
    num_iters = 10
    metric = np.zeros((len(k_list), num_iters))
    for num_cppns in num_cppns_list:
        folder = os.path.join(self.args.log_file, f"coverage_metric_N{str(num_cppns)}")
        os.makedirs(folder, exist_ok=True)
        for k_index, max_depth in enumerate(k_list):
            for j in range(num_iters):
                start = time.time()
                if not hasattr(self, 'cppns'):
                    self.cppns = self.open_evolve_cppns(folder=folder, num_cppns=num_cppns, max_depth=max_depth, iter=j)
                logger.info('starting eval...')
                results = np.zeros((num_cppns, len(self.optimizers)))
                eval_tasks = []
                for cppn_index, cppn_params in tqdm(enumerate(self.cppns)):
                    ephemoral_optimizer = self.create_optimizer(cppn_params, seed=0, nolog=True)
                    for optim_index, optim in enumerate(self.optimizers.values()):
                        task = ephemoral_optimizer.start_theta_eval(optim.theta)
                        assert all(task[1] == optim.theta)
                        eval_tasks.append((task, cppn_index, optim_index))
                logging.info('getting eval results....')        
                for task, cppn_index, optim_index in tqdm(eval_tasks):
                    stats = ephemoral_optimizer.get_theta_eval(task)
                    results[cppn_index, optim_index] = stats.eval_returns_mean
                
                # compute the number of cppns that are solved by any active agent
                num_solved_cppns = 0
                for i in range(num_cppns):
                    if any(results[i, :] > threshold):
                        num_solved_cppns += 1


                metric = num_solved_cppns / num_cppns
                logger.info(f"************************** Coverage Metric Result: {metric} : {time.time()-start}secs")
                metric[k_index, j] = metric
                np.save(os.path.join(folder, f"coverage_metric.npy"), metric)
                del self.cppns
        f = plt.figure()        
        plt.plot(k_list, metric, 'o', color='lightgray')
        plt.errorbar(x=k_list, y=np.mean(metric, axis=1), yerr=np.std(metric, axis=1), fmt='bo', capsize=4)
        plt.grid()
        plt.xlabel('K (max CPPN evolution depth)')
        plt.ylabel('Coverage Metric Score')
        plt.title(f"Coverage Metric Score vs Max CPPN Evolution Depth\n ({num_iters} samples w/ N={num_cppns} CPPNs)")
        plt.ylim([0.32, 1.05])
        f.savefig(os.path.join(folder, f"score_vs_k_n{num_cppns}.png"))
    os.exit()
    return metric


#####################################
##
## Get Coverage Metric
##
#####################################
def eval_coverage_TvK(self, folder=None, threshold=-25, num_cppns=10, max_depth=10):
    '''
    Initial draft of a coverage metric as described here in mattermost: 
    https://chat.git.act3-ace.com/act3/pl/e45kc6gdsjryfqkqeuh67cjecw

    "Given a random environment uniformly drawn from this space of 
    environments, what is the probability that any agent in a population
    of agents gets a score above T." -Castanon

    Create a group of randomly evolved cppn environments of size N.  
    Limit the depth of an evolution sequence to k (max_depth).

    For each environment, evaluate against each agent in both the 
    active and archived populations.  Keep track of whether each environment
    is ever solved.

    Return the ratio of solved environments to total environments. 

    '''
    begin_time = time.time()
    logger.info('************************ Computing Coverage Metric ************************')
    num_cppns = 1000  # overrides for looping experiment
    # t_list = [-100, -75, -50, 0, 50, 100, 150, 200, 230, 250, 270, 290] 
    # t_list = [-100, -75, -50, 0, 50,  230, 250, 270, 290] 
    # t_list = [-100, -50, 0, 50, 100, 150, 200, 250, 300] 
    t_list = [200] 


    k_list = [20, 35, 50, 75, 100]
    colors = ['b', 'g', 'r', 'c', 'deeppink', 'slategray', 'royalblue', 'orange']*2
    # assert len(colors) >= len(t_list)
    num_iters = 10
    metric = np.zeros((len(k_list), num_iters))
    kjnm_databrick = np.zeros((len(k_list), num_iters, num_cppns, len(self.optimizers)))
    
    fig, axs = plt.subplots(figsize=(16, 12), ) 
    if folder:
        folder = os.path.join(folder, f"coverage_metric_N{str(num_cppns)}")
    else:
        folder = os.path.join(self.args.log_file, self.args.logtag[1:-3], f"coverage_metric_N{str(num_cppns)}")
    os.makedirs(folder, exist_ok=True)
    kjnm_databrick_filename = os.path.join(folder, f"kjnm_databrick.npy")

    for color_index, threshold in enumerate(t_list):
        npy_filename = os.path.join(folder, f"coverage_metric_T{threshold}.npy")
        if os.path.exists(npy_filename):
            metric = np.load(npy_filename)
        else:
            for k_index, max_depth in enumerate(k_list):
                for current_iter in range(num_iters):
                    start = time.time()
                    if not hasattr(self, 'cppns'):
                        self.cppns = self.open_evolve_cppns(folder=folder, num_cppns=num_cppns, max_depth=max_depth, iter=current_iter)
                    logger.info('starting eval...')
                    results = np.zeros((num_cppns, len(self.optimizers)))
                    eval_tasks = []
                    for cppn_index, cppn_params in tqdm(enumerate(self.cppns)):
                        ephemoral_optimizer = self.create_optimizer(cppn_params, seed=0, nolog=True)
                        for optim_index, optim in enumerate(self.optimizers.values()):
                            task = ephemoral_optimizer.start_theta_eval(optim.theta)
                            assert all(task[1] == optim.theta)
                            eval_tasks.append((task, cppn_index, optim_index))
                    logging.info('getting eval results....')        
                    for task, cppn_index, optim_index in tqdm(eval_tasks):
                        stats = ephemoral_optimizer.get_theta_eval(task)
                        results[cppn_index, optim_index] = stats.eval_returns_mean
                        kjnm_databrick[k_index, current_iter, cppn_index, optim_index] = stats.eval_returns_mean
                    # compute the number of cppns that are solved by any active agent
                    num_solved_cppns = 0
                    for i in range(num_cppns):
                        if any(results[i, :] > threshold):
                            num_solved_cppns += 1


                    metric = num_solved_cppns / num_cppns
                    logger.info(f"************************** Coverage Metric Result: {metric} : threshold:{threshold} : {time.time()-start}secs")
                    metric[k_index, current_iter] = metric
                    
                    del self.cppns
            np.save(npy_filename, metric)        
        np.save(kjnm_databrick_filename, kjnm_databrick)





        # plt.plot(k_list, metric, 'o', color='lightgray')
        plt.errorbar(
            x=[k+color_index*0.3 for k in k_list], 
            y=np.mean(metric, axis=1), 
            yerr=np.std(metric, axis=1), 
            marker='.', 
            capsize=4, 
            capthick=4,
            label=f"T={threshold}",
            color=colors[color_index],
        )
        plt.grid(visible=True)
        plt.xlabel('K (max CPPN evolution depth)')
        plt.ylabel('Coverage Metric Score\n(Fraction of freely evolved CPPN environments solvable by any member of test population)')
        plt.title(f"Baseline_POET_checkpoint1000\n"+
                f"Coverage Metric Score vs K (Max CPPN Evolution Depth)\nfor various Thresholds T"+
                # f" (Note irregular intervals)"+
                f" (at regular intervals)"+
                f"\n N={num_cppns} CPPNs (re-evolved {num_iters} times)\nerror bars (Std. Dev.) artificially skewed for clarity")
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        fig.legend( ncols=1, loc='lower left', bbox_to_anchor = (0.07, 0.07), framealpha=1.0)
        fig.savefig(os.path.join(folder, f"score_vs_k_n{num_cppns}.png"))
    logger.info(f"total coverage metric time: {time.time() - begin_time}")


#####################################
##
## Get Coverage Metric
##
#####################################
def eval_coverage_kjnm(optimizer = None, folder=None):
    '''                                                                   
    Initial draft of a coverage metric as described here in mattermost: 
    https://chat.git.act3-ace.com/act3/pl/e45kc6gdsjryfqkqeuh67cjecw

    "Given a random environment uniformly drawn from this space of 
    environments, what is the probability that any agent in a population
    of agents gets a score above T." -Castanon

    Create a group of randomly evolved cppn environments of size N.  
    Limit the depth of an evolution sequence to k (max_depth).

    For each environment, evaluate against each agent in both the 
    active and archived populations.  Keep track of whether each environment
    is ever solved.

    Return the ratio of solved environments to total environments. 

    '''
    begin_time = time.time()
    logger.info('************************ Computing Coverage Metric ************************')

    cm_config_fn = os.path.join(optimizer.args.log_file, "cov_met_config.json")    
    with open(cm_config_fn) as f:
        cov_met_config = json.load(f)
    mode = cov_met_config['mode']
    # 
    if mode == 'only_last':
        folder = os.path.join(folder, f"fixedset_coverage_metric_onlyLast_wRollouts")
    if mode == 'make_gifs':
        folder = os.path.join(folder, f"fixedset_coverage_metric_make_gifs")
    elif isinstance(mode, int):
        folder = os.path.join(folder, f"fixedset_coverage_metric_every{cov_met_config['mode']}th_wRollouts")
    elif isinstance(mode, dict):
        if 'index' in mode:
            folder = os.path.join(folder, f"fixedset_coverage_metric_only_{cov_met_config['mode']['index']}th_wRollouts")
        elif 'specific_env' in mode:
            folder = os.path.join(folder, f"fixedset_coverage_metric_specifically_env_{cov_met_config['mode']['specific_env']}")
    else:
        folder = os.path.join(folder, f"fixedset_coverage_metric_N{str(cov_met_config['num_cppns'])}_wRollouts")


    num_cppns = cov_met_config['num_cppns'] 
    t_list = cov_met_config['t_list']
    k_list = cov_met_config['k_list']
    num_iters = cov_met_config['num_iters']
    


    colors = ['b', 'g', 'r', 'c', 'deeppink', 'slategray', 'royalblue', 'orange']*10
    iter_str = folder[folder.rfind('cp-')+3:folder.find('/', folder.rfind('cp-'))]
    os.makedirs(folder, exist_ok=True)
    width = 12
    height = 8
    kjnm_databrick_filename = os.path.join(folder, f"kjnm_databrick.npy")  
    rollouts_list_filename = os.path.join(folder, f"rollouts.pkl") 
    all_optimizers = OrderedDict()
    all_optimizers.update(optimizer.optimizers)
    all_optimizers.update(optimizer.archived_optimizers)

    if os.path.exists(kjnm_databrick_filename):
        kjnm_databrick=np.load(kjnm_databrick_filename)
        with open(rollouts_list_filename, 'rb') as f:
            rollouts = pickle.load(f)
    else:
        kjnm_databrick = np.zeros((len(k_list), num_iters, num_cppns, len(all_optimizers)))
        for k_index, max_depth in enumerate(k_list):  # run evals and populate databrick with eval scores
            for current_iter in range(num_iters):
                start = time.time()
                if not hasattr(optimizer, 'cppns'):
                    optimizer.cppns = load_cppns(folder=folder, debug=False, mode=mode) ############################
                    if mode=='make_gifs' or (isinstance(mode, dict) and 'specific_env' in mode): 
                        for i, c in enumerate(optimizer.cppns):
                            c.__doc__ = f"{str(i)}!{folder}" # used to convey the 0-9999 env id and alternate folder location to the gif writing code
                    logging.info(' loaded stored cppns') 

                # logger.info(' starting eval...')
                # results = np.zeros((num_cppns, len(all_optimizers)))
                # rollouts = []
                # for course_index in range(8,10):
                #     print(f"Course Index = {course_index * 1000}")
                #     eval_tasks = []
                #     start=course_index*1000
                #     end = 1000 * (course_index + 1)
                #     for cppn_index, cppn_params in enumerate(optimizer.cppns[start:end]):
                #         cppn_index += start
                #         ephemoral_optimizer = optimizer.create_optimizer(cppn_params, seed=0, nolog=True)
                #         for optim_index, optim in enumerate(all_optimizers.values()):
                #             task = ephemoral_optimizer.start_theta_eval(optim.theta)
                #             assert all(task[1] == optim.theta)
                #             eval_tasks.append((task, cppn_index, optim_index))
                #     logging.info(' getting eval results....')        
                #     for task, cppn_index, optim_index in eval_tasks:
                #         stats = ephemoral_optimizer.get_theta_eval(task)
                #         results[cppn_index, optim_index] = stats.eval_returns_mean
                #         rollouts.append(stats.eval_rollouts)
                #         kjnm_databrick[k_index, current_iter, cppn_index, optim_index] = stats.eval_returns_mean

                logger.info(' starting eval...')
                results = np.zeros((num_cppns, len(all_optimizers)))
                rollouts = []
                eval_tasks = []
                for cppn_index, cppn_params in enumerate(optimizer.cppns):
                    ephemoral_optimizer = optimizer.create_optimizer(cppn_params, seed=0, nolog=True)
                    for optim_index, optim in enumerate(all_optimizers.values()):
                        task = ephemoral_optimizer.start_theta_eval(optim.theta)
                        assert all(task[1] == optim.theta)
                        eval_tasks.append((task, cppn_index, optim_index))
                logging.info(' getting eval results....')        
                for task, cppn_index, optim_index in eval_tasks:
                    stats = ephemoral_optimizer.get_theta_eval(task)
                    results[cppn_index, optim_index] = stats.eval_returns_mean
                    rollouts.append(stats.eval_rollouts)
                    kjnm_databrick[k_index, current_iter, cppn_index, optim_index] = stats.eval_returns_mean

                del optimizer.cppns
        np.save(kjnm_databrick_filename, kjnm_databrick)
        with open(rollouts_list_filename, 'ab') as f:
            pickle.dump(rollouts, f)
    # extract coverage metric for score thresholds in t_list
    cov_fig_filename = os.path.join(folder, f"score_vs_k_n{num_cppns}.png")
    summary_filename = os.path.join(folder, f"summary.json")
    summary = {}
    if not os.path.exists(cov_fig_filename):
        fig, axs = plt.subplots(figsize=(width, height), )
        for color_index, threshold in enumerate(t_list):
            metric = np.zeros((len(k_list), num_iters))  # k x j
            for k_index in range(len(k_list)): # k
                for current_iter in range(num_iters): # j
                    results = np.zeros((num_cppns, len(all_optimizers)))  # n x m
                    for cppn_index in range(num_cppns): # n
                        for m_index in range(len(all_optimizers)):
                            results[cppn_index, m_index] = kjnm_databrick[k_index, current_iter, cppn_index, m_index]
                    num_solved_cppns = 0
                    for n in range(num_cppns):
                        if any(results[n, :] > threshold):
                            num_solved_cppns += 1
                    if mode == 'only_last':
                        metric[k_index, current_iter] = num_solved_cppns / (num_cppns / k_list[k_index])
                    elif isinstance(mode, int):
                        metric[k_index, current_iter] = num_solved_cppns / (num_cppns / mode)
                    #TODO shoudl have another entry here for dict case with only Mth env in evoluti9onary sequence
                    else:
                        metric[k_index, current_iter] = num_solved_cppns / num_cppns

            plt.errorbar(
                x=[k+color_index*0.3 for k in k_list], 
                y=np.mean(metric, axis=1), 
                yerr=np.std(metric, axis=1), 
                marker='.', 
                capsize=4, 
                capthick=4,
                label=f"T={threshold}",
                color=colors[color_index],
            )
            summary[f"K{k_list[k_index]}T{t_list[color_index]}_mean"]=float(np.mean(metric, axis=1))
            summary[f"K{k_list[k_index]}T{t_list[color_index]}_stdev"]=float(np.std(metric, axis=1))

            plt.grid(visible=True)
            plt.xlabel('K (max CPPN evolution depth)')
            plt.ylabel('Coverage Metric Score\n(Fraction of freely evolved CPPN environments solvable by any member of test population)')
            plt.title(
                    f"Iteration: {iter_str}\n" +
                    f"Coverage Metric Score vs K (Max CPPN Evolution Depth)\nfor various Thresholds T"+
                    # f" (Note irregular intervals)"+
                    f" (at regular intervals)"+
                    f"\n N={num_cppns} CPPNs (re-evolved {num_iters} times)\nerror bars (Std. Dev.) artificially skewed for clarity")
            plt.ylim([0.0, 1.0])
            plt.tight_layout()
            fig.legend( ncols=1, loc='lower left', bbox_to_anchor = (0.07, 0.07), framealpha=1.0)
            fig.savefig(cov_fig_filename)
        with open(summary_filename, "w") as f:
            json.dump(summary, f)
        logger.info(f"total coverage metric time: {time.time() - begin_time}")
    
    # plot success rate of each agent over n and j
    # subplot grid over T (vertical) and K (Horizontal)
    agents_fig_filename = os.path.join(folder, f"agent_performance_vs_k_t.png")
    if not os.path.exists(agents_fig_filename):
        fig, axs = plt.subplots(figsize=(width, height), ncols=len(k_list), nrows=len(t_list), squeeze=False)
        fig.suptitle(f"Per Agent Win Rate over {num_iters} Coverage Metric evals \n vs Score Threshold T, and Max Cppn Evolution Depth K"+ \
            f"\n(error bars are stdev)")
        plt_index = 1
        for color_index, threshold in enumerate(t_list):
            agent_perf = np.zeros((len(all_optimizers))) 
            
            for k_index in range(len(k_list)): # k
                results = np.zeros((num_iters, num_cppns, len(all_optimizers)))  # j x n x m
                for current_iter in range(num_iters): # j
                    for cppn_index in range(num_cppns): # n
                        for m_index in range(len(all_optimizers)):
                            results[current_iter, cppn_index, m_index] = kjnm_databrick[k_index, current_iter, cppn_index, m_index] > threshold
                axs[color_index, k_index].bar(
                    x=list(all_optimizers.keys()), #list(range(len(all_optimizers))),
                    height=results.mean(axis=1).mean(axis=0),
                    yerr = results.mean(axis=1).std(axis=0),
                )
                if color_index < len(t_list)-1:  # not bottom row
                    axs[color_index, k_index].xaxis.set_visible(False)
                else:  # bottom row
                    axs[color_index, k_index].xaxis.set_tick_params(rotation=-45)
                    # axs[color_index, k_index].set_xticks(list(all_optimizers.keys()), rotation=-45)
                    # axs[color_index, k_index].set_xlabel(f"population agent index")


                axs[color_index, k_index].grid(visible=True)
                if k_index==0:
                    axs[color_index, k_index].set_ylabel(f"T={threshold}")

                axs[color_index, k_index].set_ylim([0.0, 1.1])
        fig.savefig(agents_fig_filename)
    conjoined_fig_filename = os.path.join(folder, f"conjoined_figure.png")
    if not os.path.exists(conjoined_fig_filename):
        from PIL import Image    
        top = Image.open(cov_fig_filename)
        bottom = Image.open(agents_fig_filename)
        conjoined = Image.new('RGB', (top.width, 2*top.height))
        conjoined.paste(top, (0,0))
        conjoined.paste(bottom, (0, top.height))
        conjoined.save(conjoined_fig_filename)
    logger.info(f"total coverage metric time: {time.time() - begin_time}")

