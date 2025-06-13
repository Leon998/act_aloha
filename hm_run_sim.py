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


def interpolate_action_sequence(action_sequence, episode_len):
    """
    Interpolate the action sequence to match the episode length.
    :param action_sequence: np.ndarray, shape (n_steps, n_actions)
    :param episode_len: int, desired length of the episode
    :return: np.ndarray, interpolated action sequence
    """
    if len(action_sequence) < episode_len:
        interpolated_actions = []
        for col in range(action_sequence.shape[1]):
            interpolated_col = np.interp(
                np.linspace(0, len(action_sequence) - 1, episode_len),
                np.arange(len(action_sequence)),
                action_sequence[:, col]
            )
            interpolated_actions.append(interpolated_col)
        return np.array(interpolated_actions).T
    else:
        return action_sequence[:episode_len]

def main():
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = "hm_pnp"
    dataset_dir = "hm_dataset/"
    num_episodes = 3
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
        # Load action sequence from file
        action_sequence_path = os.path.join(dataset_dir, f"source/pnp_side_grasp_{episode_idx}/action_sequence.txt")
        action_sequence = np.loadtxt(action_sequence_path)[:, :8]
        action_sequence = interpolate_action_sequence(action_sequence, episode_len)
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = action_sequence[step]
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.001)
        plt.close()


if __name__ == '__main__':
    
    main()

