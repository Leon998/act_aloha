import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from hm_constants import SIM_TASK_CONFIGS, START_ARM_POSE
from hm_sim_env import make_sim_env, BOX_POSE

import IPython
e = IPython.embed


def main():
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = "humanoid_pnp"
    dataset_dir = "dataset_tmp/"
    num_episodes = 1
    onscreen_render = True
    render_cam_name = 'fixed'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']

    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        # setup the environment
        env = make_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = np.array(START_ARM_POSE)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()


if __name__ == '__main__':
    
    main()

