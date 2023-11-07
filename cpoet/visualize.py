# %%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import report

# %%
## Visualize Bipedal Walker CPPN Environments
# set root directory and run name
root = "/root"
runname = "poet_03"

# directory for cppns
cppnDir = root + "/logs/" + runname + "/saved_envs/"
 
inhar = []

plt.figure(figsize = [10,10])

for dir in next(os.walk(cppnDir))[1]:
    coord_path = os.path.join(cppnDir,dir,"_xy.json")
    cofig_path = os.path.join(cppnDir,dir,"_config.json")
    
    with open(coord_path) as f:
        _xy = json.load(f)
        plt.plot(_xy["x"], _xy['y'], label=dir)

    with open(cofig_path) as f:
        _config = json.load(f)
        inhar.append((_config["parent"], dir))

plt.legend()
plt.show()

# %%
## Load report and view basic info
dir = f"{root}/ipp/{runname}/"

r = report.report(dir)
r.plot_evolution_to_scale()
r.plot_training()

# %%
####
# Example evolution:
####
import numpy as np
import poet_distributed.niches.ES_Bipedal.cppn as cppn

n = cppn.CppnEnvParams(0)

f, axes = plt.subplots(nrows=4,ncols=4, sharex=True,figsize=[16,16])

for ax in axes.ravel():
    xy = n.xy()
    ax.plot(xy["x"], xy["y"])
    n = n.get_mutated_params()
    n = n.get_mutated_params()
    n = n.get_mutated_params()

# %%
