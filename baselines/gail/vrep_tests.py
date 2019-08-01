import os
from baselines.gail.vrep_ur_env import UR5VrepEnvMultiObstacle

env = UR5VrepEnvMultiObstacle()
for i in range(10):
    print(i)
    n_path, path = env.reset_expert()
    if path:
        print('exists')
        for j in range(n_path):
            env.set_joints(path[j])
            env.step_simulation()
env.close()
