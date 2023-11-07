import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})

# experiment data
# % of interesting envs solved at x thousands of poet iterations for baseline run 21Feb

cm_scores_0a =[0.46, 0.66, 0.93, 1.0, 1.0, 1.0, 1.0]
cm_scores_0b =[0.53, 0.56, 0.56, 0.66, 0.66, 0.86]


cm_scores_10 = [0.8, 0.93, 0.97, 0.97, 0.97, 0.97, 1]

kiters = [1,2,3,4,5,6,7]

fig, ax = plt.subplots(1,1)
ax.plot(kiters[:len(cm_scores_10)], cm_scores_10, color='orange', marker='o')
ax.plot(kiters[:len(cm_scores_0a)], cm_scores_0a, color='blue', marker='o')
ax.plot(kiters[:len(cm_scores_0b)], cm_scores_0b, color='blue', marker='o')

# for i in range(len(cm_scores_0)):
#     ax.text(x=kiters[i]-0.5, y=cm_scores_0[i]+0.02, s=labels_0[i], fontsize=10)

ax.set_xlabel(f"POET iterations (k)")
ax.set_ylabel(f"Coverage Metric Score")
ax.grid()
ax.legend(['Curious POET','ePOET', 'ePOET'])
# ax.legend(['ePOET'])

fig.tight_layout()
fig.savefig(f"/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM/_cm_on_interesting_envs/baseline_perf_iterations.png")