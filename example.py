# Example of the EPP method

import pandas
from epp_functions import *

# Read file containing AIS DataFrame.
d = pd.read_pickle('ais_data.p')

# Digest AIS DataFrame into a list of trajectories. 
trajectories = digest_messages(d)

# Simplify and plot the trajectories
for i, trajectory in enumerate(trajectories):
    if len(trajectory) == 1:  # Skip trajectories with only one vertex
        continue
    st = epp_simplify(trajectory)
    plt.figure()
    plt.title('traj '+str(i))
    plot_subtrajectories(st,trajectory=trajectory,mercator=True)


    #plot_subtrajectories(st,mercator=True)       
plt.show()
