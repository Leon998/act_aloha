import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from hm_constants import DT, XML_DIR, START_ARM_POSE

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name):
    xml_path = os.path.join(XML_DIR, f'fixed_robot.xml')
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PnPTask(random=False)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)
    return env

def sample_box_pose():
    x_range = [0.0, 0.0]
    y_range = [1.0, 1.0]
    z_range = [0.24, 0.24]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

class HumanoidTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        

    def before_step(self, action, physics):
        env_action = np.zeros(36)
        left_arm_action = action
        env_action[18:26] = left_arm_action
        super().before_step(env_action, physics)
        return
    
    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[18:26] = START_ARM_POSE

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[18:]
        left_arm_qpos = left_qpos_raw[:8]
        return np.array(left_arm_qpos)

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[18:]
        left_arm_qvel = left_qvel_raw[:8]
        return np.array(left_arm_qvel)

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['fixed'] = physics.render(height=480, width=640, camera_id='fixed')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PnPTask(HumanoidTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        super().initialize_episode(physics)
        

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[-7:]
        return env_state

    def get_reward(self, physics):
        reward = 1
        return reward


if __name__ == '__main__':
    pass

