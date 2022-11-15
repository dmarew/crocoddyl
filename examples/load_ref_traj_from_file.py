import json
import numpy as np
# load json file with reference trajectory
with open('ref_trajectory.txt') as f:
    ref_traj = json.load(f)
print("Reference trajectory loaded from file")
print("Number of waypoints: ", np.array(ref_traj['Frames']).shape)
