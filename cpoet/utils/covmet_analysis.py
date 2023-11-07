import numpy as np
import os, pickle
import argparse
import matplotlib.pyplot as plt

#####################################
##
## Compute population generality / or maybe behavioral diversity?
##
#####################################
def explore(folder):
    THRESHOLD = 230
    ACTIVE_POP_SIZE = 10
    output_folder = os.path.join(folder, 'analysis')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # get run name
    end=folder[:folder.rfind('/')].rfind('/')
    start=folder[:folder[:folder.rfind('/')].rfind('/')].rfind('/')+1
    run_name = folder[start:end]

    # get checkpoint
    end=folder.rfind('/')
    start=folder[:folder.rfind('/')].rfind('/')+1
    checkpoint = folder[start:end]


    kjnm_databrick=np.load(os.path.join(folder, 'kjnm_databrick.npy'))
    
    assert kjnm_databrick.shape[0] == 1   
    assert kjnm_databrick.shape[1] == 1  


    ############ plot the whole agent score vs environments matrix

    data = np.transpose(kjnm_databrick[0,0,:,0:ACTIVE_POP_SIZE])  # N(envs) by M ( active agents)
    pop_size = min(ACTIVE_POP_SIZE, data.shape[0])
    fig, axs = plt.subplots(20, 1)
    fig.set_size_inches(20, 20)
    for i in range(20):
        a=axs[i].imshow(data[:, i*500:(i*500)+500], cmap='plasma')
    axs[0].set_title('Curious POET agent scores against 10k coverage metric environments')
    fig.colorbar(a)
    fig.savefig(os.path.join(output_folder,'kjnm_databrick.png'))

    ######### plot the overall mean agent scores
    fig, axs = plt.subplots(1,1)
    # fig.set_size_inches(7, 7)
    axs.bar(list(range(pop_size)), data[:pop_size].mean(axis=1), color=['g']*pop_size)
    active_mean = data[:10].mean()
    # archive_mean = data[10:].mean()
    axs.plot([-0.5, 9.5],[active_mean, active_mean], color='y', label=f"{active_mean:3.1f} (Active Population)")
    # axs.plot([9.5, pop_size-0.5],[archive_mean, archive_mean], color='r', label=f"{archive_mean:3.1f} (Archive Population)")
    axs.set_title("Mean agent scores over 10k envs - Active(green)")
    axs.set_xticks(range(pop_size))
    axs.set_xlabel('Agent ID')
    axs.set_ylabel('Mean score')
    axs.legend(loc='lower left')
    axs.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder,'overall_mean_agent_scores.png'))

    ############# plot the counterexamples mean agent scores
    remaining = 10000 - sum((np.sum(data>THRESHOLD, axis=0))==pop_size) - sum((np.sum(data>THRESHOLD, axis=0))<1)
    counterexamples = np.zeros((pop_size, remaining))
    idx = 0
    for i in range(10000):
        solved_by = np.sum(data[:,i]>230, axis=0)
        if solved_by>0 and solved_by<pop_size:
            counterexamples[:,idx] = data[:,i]
            idx += 1

    fig, axs = plt.subplots(1,1)
    # fig.set_size_inches(7, 7)
    axs.bar(list(range(pop_size)), counterexamples.mean(axis=1), color=['g']*ACTIVE_POP_SIZE)
    active_mean = counterexamples.mean()
    # archive_mean = counterexamples[10:].mean()
    axs.plot([-0.5, ACTIVE_POP_SIZE-0.5],[active_mean, active_mean], color='y', label=f"{active_mean:3.1f} (Active Population)")
    # axs.plot([9.5, (pop_size-0.5)],[archive_mean, archive_mean], color='r', label=f"{archive_mean:3.1f} (Archive Population)")
    axs.set_title(f"Mean agent scores over {remaining/1000:1.1f}k envs - Active(green)")
    axs.set_xticks(range(pop_size))
    axs.set_xlabel('Agent ID')
    axs.set_ylabel('Mean score')
    axs.legend()
    axs.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder,'counterexample_mean_agent_scores.png'))


        ############################ plot the histogram of solved counteresamples
    solved_by = np.sum(counterexamples>THRESHOLD, axis=0)

    fig, axs = plt.subplots(1,1)
    axs.hist(solved_by, [i+0.5 for i in range(pop_size) ])
    axs.set_title(f"({remaining/1000:1.1f}k not-too-easy-or-hard envs)")
    axs.set_xticks(range(1,pop_size))
    axs.set_xlabel('solved by this number of agents')
    axs.set_ylabel('number of environments')
    axs.grid()

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder,'counter_examples_histogram.png'))
    
    ############ plot counterexample environments

    fig, axs = plt.subplots(4,1)
    fig.set_size_inches(10, 4)
    for i in range(4):
        a=axs[i].imshow(counterexamples[:, i*500:(i*500)+500], cmap='plasma')
    axs[0].set_title(f"Curious POET agent scores against {remaining/1000:1.1f}k not-too-easy-or-hard envs")
    fig.colorbar(a)
    fig.savefig(os.path.join(output_folder,'kjnm_databrick_counterexamples.png'))

    ############## counter examples histogram
    remaining = 10000 - sum((np.sum(data>THRESHOLD, axis=0))==0) -sum((np.sum(data>THRESHOLD, axis=0))==pop_size)
    counterexamples = np.zeros((pop_size, remaining))
    idx = 0
    for i in range(10000):
        solved_by = np.sum(data[:,i]>THRESHOLD, axis=0)
        if solved_by>0 and solved_by<pop_size:
            counterexamples[:,idx] = data[:,i]
            idx += 1

    solved_by = np.sum(counterexamples>THRESHOLD, axis=0)


    fig, axs = plt.subplots(1,1)
    axs.hist(solved_by, [i+0.5 for i in range(pop_size) ])
    axs.set_title("Histogram of 'number of agents solved' ")
    axs.set_xticks(range(1,pop_size))
    axs.set_xlabel('solved by this number of agents')
    axs.set_ylabel('number of environments')
    axs.grid()
    fig.savefig(os.path.join(output_folder,'counter_examples_histogram.png'))
    
    ################################  Scatter bubble plot.  
    # x= solved by this num of agents, (aka Difficulty, where lower numbers are more difficult)
    # y = agent ID, 
    # bubble size = number of envs
    SCALE_FACTOR = 200
    fig, axs = plt.subplots(1,1)
    # fig.set_size_inches(10,6)
    for difficulty in range(1,pop_size):
        count = sum((np.sum(data>THRESHOLD, axis=0))==difficulty)
        examples = np.zeros((pop_size, count))
        idx = 0
        for i in range(10000):
            solved_by = np.sum(data[:,i]>THRESHOLD, axis=0)
            if solved_by == difficulty:
                examples[:,idx] = data[:,i]
                idx += 1
        proportion = np.sum(examples>THRESHOLD, axis=1) /  examples.shape[1]


        axs.scatter([difficulty]*pop_size, range(1,(pop_size+1)), c=range(1,(pop_size+1)), s=proportion * SCALE_FACTOR, alpha=0.5)
    axs.set_title('Proportion of envs solved\nnormalized by difficulty group size')
    axs.set_xlabel('Envs only solved by this number of agents')
    axs.set_ylabel('agent ID')
    axs.set_xticks(range(1,pop_size))
    axs.set_yticks(range(1,(pop_size+1)))
    fig.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder,'bubble_plot.png'))

    ###############  Account for varying K #####################
    # data is of shape M agents by N=10000 envs, where each kth env has the same value of k, typically 100
    # reorder the array into a cube of shape M agents X n envs X k evolutionary depth
    # then plot in various ways to see performance vs k depth

    assert data.shape[1] == 10000
    kbrick = np.zeros((data.shape[0], 100, 100))  # pop_size x k x n
    # loop over n envs grouping envs by k value
    for n in range(100):   # loop over n = N/k (100)  - the new smaller N, we'll call it n
        for k in range(100):  # loop over k
            kbrick[:, n, k] = data[:, n*100 + k]



    horizontal = max(int(np.ceil(np.sqrt(pop_size))), 2)
    vertical = max(int(np.ceil(pop_size/horizontal)), 2)
    fig, axs = plt.subplots(vertical, horizontal)
    fig.set_size_inches(horizontal*2.5, vertical * 2.5 )
    agent_id=0
    for v in range(vertical):
        for h in range(horizontal):
            if agent_id > pop_size-1:
                axs[v,h].set_visible(False)
            else:
                a=axs[v,h].imshow(kbrick[agent_id,:,:], cmap='plasma')
                axs[v,h].set_title(f"Agent {agent_id}")
                if h == 0:
                    axs[v,h].set_ylabel(f"Ennvironments")
                if v == vertical-1:
                    axs[v,h].set_xlabel(f"Evolution depth K")
            agent_id +=1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    fig.colorbar(a, cax=cbar_ax)
    fig.suptitle(f"Agent scores for N envs vs k evolutionary depth\nPopulation: {run_name} Checkpoint: {checkpoint}")
    fig.savefig(os.path.join(output_folder,'k_brick.png'))
    


    
    # for n in range(10):   # loop over n = N/k (100)  - the new smaller N, we'll call it n
    #     for k in range(10):  # loop over k
    #         a=axs[n,k].imshow(kbrick[:, n*100:(k*100)+100], cmap='plasma')
    # axs[0].set_title('Curious POET agent scores against 10k coverage metric environments')




    #############################   clustering
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    kmeans_kwargs = {
        "init":"random",
        "n_init":10,
        "max_iter":300,
    }
    try:
        sse = []
        stop = 40
        for k in range(1, stop):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)

            kmeans.fit(np.transpose(counterexamples))
            sse.append(kmeans.inertia_)

        fig, axs = plt.subplots(1,1)
        axs.plot(range(1,stop), sse)
        axs.set_xticks(range(1,stop))
        axs.set_xlabel('number of clusters')
        axs.set_ylabel('SSE')
        axs.grid()
        axs.set_title('K means error vs number of clusters')
        fig.savefig(os.path.join(output_folder,'k_means_elbow.png'))
    except ValueError:
        print(f"Experienced a ValueError running clustering")

    try:
        sil_coeffs=[]
        for k in range(2, stop):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)

            kmeans.fit(np.transpose(counterexamples))
            sil_coeffs.append(silhouette_score(np.transpose(counterexamples), kmeans.labels_))

        fig, axs = plt.subplots(1,1)
        axs.plot(range(2,stop), sil_coeffs)
        axs.set_xticks(range(2, stop))
        axs.set_xlabel('number of clusters')
        axs.set_ylabel('Silhouette Score')
        axs.set_title('K means silouette score vs number of clusters\n(Higher is better)')
        axs.grid()
        fig.savefig(os.path.join(output_folder,'k_means_silhouette.png'))



        kmeans = KMeans(n_clusters=2, **kmeans_kwargs)

        kmeans.fit(np.transpose(counterexamples))
    except ValueError:
        print(f"Experienced a ValueError running clustering")
    

    ############################################## Trajectory Analysis ##################################################
    
    print(f"loading rollouts...")
    #  Rollouts analysis
    with open(os.path.join(folder, 'rollouts.pkl'), 'rb') as f:
        rollouts = pickle.load(f)

    # state vector is first 14 entries, lidar is the next 10, then the next 4 are action space, followed by done
    # 0: hull angle
    # 1: hull angle velocity
    # 2: x velocity
    # 3: y velocity
    # 4: joint 0 angle hip
    # 5: joint 0 speed
    # 6: joint 1 angle knee
    # 7: joint 1 speed
    # 8: leg ground contact
    # 9:  joint 2 angle hip
    # 10: joint 2 speed
    # 11: joint 3 angle knee
    # 12: joint 3 speed
    # 13: leg ground contact
    # 14-23 lidar
    # 24-27 actions (joint speeds)
    # 28: done

    labels = [
            '0: hull angle',
            '1: hull angle velocity',
            '2: x velocity',
            '3: y velocity',
            '4: hip joint 0 angle',  # defining 0 as right and 1 as left
            '5: hip joint 0 speed',
            '6: knee joint 1 angle',
            '7: knee joint 1 speed',
            '8: leg ground contact',
            '9:  hip joint 2 angle',
            '10: knee joint 2 speed',
            '11: hip joint 3 angle',
            '12: kneejoint 3 speed',
            '13: leg ground contact',
            '24 action: hip joint speed',
            '25 action: knee joint speed',
            '26 action: hip joint speed',
            '27 action: knee joint speed',

    ]

    ################################### GAIT ANALYSIS ##########################################
    # define a stride as the ground distance between contact points for the same foot

    # compute adjacent differences over vector x, return with stats
    def get_diffs(x):
        diffs = []
        if len(x) < 2:
            return diffs, 0.00000001, 0.00000001
        
        for a, b in zip(x[0::], x[1::]):
            diffs.append(b-a)
        mean = sum(diffs)/len(diffs)
        std = np.std(diffs)

        return diffs, mean, std

    def get_gait_char(rollout):
        # Approximate position x by integrating velocity, assuming dt = 1
        x=[]
        current_x=0
        for step in range(rollout.shape[0]):
            current_x += rollout[step, 2] # assuming dt = 1
            x.append(current_x)

        x_on_right_contact = []
        x_on_left_contact = []
        step_on_right_contact = []
        step_on_left_contact = []
        right_contact=False # start with 'feet in the air'
        left_contact=False
        for step in range(rollout.shape[0]):
            #state machines to keep track of state
            if rollout[step,8] and not right_contact:   # touchdown right
                right_contact = True
                x_on_right_contact.append(x[step])
                step_on_right_contact.append(step)
            if not rollout[step,8] and right_contact:   # liftoff right
                right_contact = False

            if rollout[step,13] and not left_contact:   # touchdown left
                left_contact = True
                x_on_left_contact.append(x[step])
                step_on_left_contact.append(step)
            if not rollout[step,13] and left_contact:   # liftoff left
                left_contact = False
            # print(rollout[step,8], rollout[step, 13], step, x[step])
        

        
        left_stride_times, mean_left_stride_time, std_left_stride_time = get_diffs(step_on_left_contact)
        right_stride_times, mean_right_stride_time, std_right_stride_time = get_diffs(step_on_right_contact)

        left_stride_x, mean_left_stride_x, std_left_stride_x = get_diffs(x_on_left_contact)
        right_stride_x, mean_right_stride_x, std_right_stride_x = get_diffs(x_on_right_contact)


        # estimate the energy per stride by summing the absolute joint actuation values
        left_stride_energy = []
        for i in range(len(step_on_left_contact)-1):
            left_stride_energy.append(np.sum(np.abs(rollout[step_on_left_contact[i]:step_on_left_contact[i+1], [24,25]])))

        right_stride_energy = []
        for i in range(len(step_on_right_contact)-1):
            right_stride_energy.append(np.sum(np.abs(rollout[step_on_right_contact[i]:step_on_right_contact[i+1], [26,27]])))


        gait_characteristic = [
            left_stride_times,
            right_stride_times,
            left_stride_x,
            right_stride_x,
            left_stride_energy,
            right_stride_energy
        ]

        return gait_characteristic

    print(f"computing gait characteristics over rollouts...")
    gait_chars = []
    for rollout in rollouts:
        rollout = rollout[0]
        gait_chars.append(get_gait_char(rollout))

    checkpoint_gait_chars = {
        'run_name':     run_name,
        'checkpoint':   checkpoint,
        'gait_chars':    gait_chars,
        'gait_char_desc': [
            'left_stride_times',
            'right_stride_times',
            'left_stride_x',
            'right_stride_x',
            'left_stride_energy',
            'right_stride_energy'
        ]
    }
    
    fn=os.path.join(output_folder, 'gait_chars.pkl')
    print(f"saving gait characteristics to {fn}...")
    with open(fn, 'wb+') as f:
       pickle.dump(checkpoint_gait_chars, f)




    ############################# state-action statistics ###################################
    total_points = 0
    for rollout in rollouts:
        total_points += rollout.shape[1]
    print(f"rollouts converts to {total_points} state_action samples.")
    stateactions = np.zeros((total_points, 18))
    stateactions_index = 0
    for rollout in rollouts:
        length = rollout.shape[1]
        stateactions[stateactions_index:stateactions_index+length,:] = np.concatenate((rollout[0,:,0:14], rollout[0,:,24:28] ), axis=1)
        stateactions_index += length
    print('Saving stateactions.npy...')
    np.save(os.path.join(output_folder,'stateactions.npy'), stateactions, allow_pickle=False)
    print('done')


    # sa_var_agg = np.var(stateactions)
    # sa_var = np.var(stateactions, axis=0)
    # sa_std = np.std(stateactions, axis=0)
    # # sa_std = np.sqrt(sa_var)
    # sa_mean = np.mean(stateactions, axis=0)
    # print(f"State-Action variance (overall): {sa_var_agg}")
    # print(f"State-Action variance (indiv): {sa_var}")
    # print(f"State-Action means (indiv): {sa_mean}")

    # fig, axs = plt.subplots(1,1)
    # fig.set_size_inches(5,5)
    # b=axs.bar(x=range(18), height=sa_mean, yerr=sa_var, label=labels)
    # axs.grid()
    # axs.set_ylim((1,-1))
    # # axs.bar_label(b)#, rotation=-90)
    # axs.set_xticks(range(18))
    # axs.set_xticklabels(labels, rotation=-90)
    # axs.set_title(f"state-actions mean and variance(error bars)\n{run_name}\ncheckpoint: {checkpoint}")
    # fig.tight_layout()
    # fig.savefig(os.path.join(output_folder,'rollouts_stats.png'))
    # print('done')
    # return  {
    #     'run_name':     run_name,
    #     'checkpoint':   checkpoint,
    #     'sa_var_agg':   sa_var_agg,
    #     'sa_var':       sa_var,
    #     'sa_std':       sa_std,
    #     'total_points': total_points,

    # }
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder')
    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    args = parse_args()
    explore(

        folder=args.folder

    )