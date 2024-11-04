import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from einops import rearrange

from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from constants import SIM_TASK_CONFIGS

import IPython
e = IPython.embed

# load policy_best.ckpt
# have it input each action: know the action by how action data was recorded (should be using mink cause action data controls the target position)
# run inference on policy, make sure to pass in correct sequence context (apparently just have to input current qpos and image)
# output action an run action on the sim

# should be same general format as record_sim_episodes.py and implementation as teleop_aloha.py
import numpy as np
import mujoco
import mujoco.viewer
import mink
import h5py
import time
from bigym.envs.dishwasher import DishwasherClose
from bigym.action_modes import AlohaPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot
from mink import SO3
from reduced_configuration import ReducedConfiguration
from loop_rate_limiters import RateLimiter
from pyjoycon import JoyCon, get_R_id, get_L_id
import os
from collections import deque


_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

class Simulator():
    def __init__(self):
        self.env = DishwasherClose(
            action_mode=AlohaPositionActionMode(floating_base=False, absolute=False, control_all_joints=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name="wrist_cam_left", rgb=True, depth=False, resolution=(480, 640)),
                    CameraConfig(name="wrist_cam_right", rgb=True, depth=False, resolution=(480, 640)),
                    CameraConfig(name="overhead_cam", rgb=True, depth=False, resolution=(480, 640)),
                    CameraConfig(name="teleoperator_pov", rgb=True, depth=False, resolution=(480, 640)),
                ],
            ),
            render_mode="human",
            robot_cls=AlohaRobot
        )
        self.env.reset()
        self.model = self.env.unwrapped._mojo.model
        self.data = self.env.unwrapped._mojo.data

        self.camera_renderers = {}
        for camera_config in self.env.observation_config.cameras:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_config.name)
            renderer = mujoco.Renderer(self.model, camera_config.resolution[0], camera_config.resolution[1])
            self.camera_renderers[camera_config.name] = (renderer, camera_id)

        self.target_l = np.array([-0.4, 0.5, 1.1])
        self.target_r = np.array([0.4, 0.5, 1.1])
        self.rot_l = SO3.identity()
        self.rot_r = SO3.from_matrix(-np.eye(3))

        self.update_rotation('z', np.pi/2, 'left')
        self.update_rotation('z', -np.pi/2, 'right')
        self.update_rotation('y', np.pi/2, 'left')
        self.update_rotation('y', np.pi/2, 'right')

        self.targets_updated = False

        self.x_min, self.x_max = -0.6, 0.6
        self.y_min, self.y_max = -0.6, 0.6
        self.z_min, self.z_max = 0.78, 1.6

        self.left_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_left/gripper_actuator").id
        self.right_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_right/gripper_actuator").id
        self.left_gripper_pos = 0.037
        self.right_gripper_pos = 0.037

        self.num_timesteps = 0

        self.action = np.zeros(14)
        self.camera_names = ['wrist_cam_left', 'wrist_cam_right', 'overhead_cam', 'teleoperator_pov']
        self.obs = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/images/wrist_cam_left': [],
            '/observations/images/wrist_cam_right': [],
            '/observations/images/overhead_cam': [],
            '/observations/images/teleoperator_pov': [],
        }

        self.left_joint_names = []
        self.right_joint_names = []
        self.velocity_limits = {}

        for n in _JOINT_NAMES:
            name_left = f"aloha_scene/left_{n}"
            name_right = f"aloha_scene/right_{n}"
            self.left_joint_names.append(name_left)
            self.right_joint_names.append(name_right)
            self.velocity_limits[name_left] = _VELOCITY_LIMITS[n]
            self.velocity_limits[name_right] = _VELOCITY_LIMITS[n]

        model = self.model
        data = self.data

        self.left_dof_ids = np.array([model.joint(name).id for name in self.left_joint_names])
        self.left_actuator_ids = np.array([model.actuator(name).id for name in self.left_joint_names])
        self.left_relevant_qpos_indices = np.array([model.jnt_qposadr[model.joint(name).id] for name in self.left_joint_names])
        self.left_relevant_qvel_indices = np.array([model.jnt_dofadr[model.joint(name).id] for name in self.left_joint_names])
        self.left_configuration = ReducedConfiguration(model, data, self.left_relevant_qpos_indices, self.left_relevant_qvel_indices)

        self.right_dof_ids = np.array([model.joint(name).id for name in self.right_joint_names])
        self.right_actuator_ids = np.array([model.actuator(name).id for name in self.right_joint_names])
        self.right_relevant_qpos_indices = np.array([model.jnt_qposadr[model.joint(name).id] for name in self.right_joint_names])
        self.right_relevant_qvel_indices = np.array([model.jnt_dofadr[model.joint(name).id] for name in self.right_joint_names])
        self.right_configuration = ReducedConfiguration(model, data, self.right_relevant_qpos_indices, self.right_relevant_qvel_indices)

        self.action_buffers = [deque(maxlen=100) for _ in range(100)]

    def so3_to_matrix(self, so3_rotation: SO3) -> np.ndarray:
        return so3_rotation.as_matrix()

    def matrix_to_so3(self, rotation_matrix: np.ndarray) -> SO3:
        return SO3.from_matrix(rotation_matrix)

    def apply_rotation(self, current_rotation: SO3, rotation_change: np.ndarray) -> SO3:
        rotation_matrix = self.so3_to_matrix(current_rotation)
        change_matrix = SO3.exp(rotation_change).as_matrix()
        new_rotation_matrix = change_matrix @ rotation_matrix
        return self.matrix_to_so3(new_rotation_matrix)

    def update_rotation(self, axis: str, angle: float, side: str):
        rotation_change = np.zeros(3)
        if axis == 'x':
            rotation_change[0] = angle
        elif axis == 'y':
            rotation_change[1] = angle
        elif axis == 'z':
            rotation_change[2] = angle

        if side == 'left':
            self.rot_l = self.apply_rotation(self.rot_l, rotation_change)
        else:
            self.rot_r = self.apply_rotation(self.rot_r, rotation_change)
        self.targets_updated = True

    def control_gripper(self, left_gripper_position, right_gripper_position):
        left_gripper_position = np.clip(left_gripper_position, 0.02, 0.037)
        right_gripper_position = np.clip(right_gripper_position, 0.02, 0.037)
        self.data.ctrl[self.left_gripper_actuator_id] = left_gripper_position
        self.data.ctrl[self.right_gripper_actuator_id] = right_gripper_position

    def add_target_sites(self):
        self.target_site_id_l = self.model.site('aloha_scene/target').id
        self.target_site_id_r = self.model.site('aloha_scene/target2').id
        self.update_target_sites(self.target_l, self.target_r, self.rot_l, self.rot_r)

    def update_target_sites(self, target_l, target_r, rot_l, rot_r):
        self.data.site_xpos[self.target_site_id_l] = target_l
        self.model.site_pos[self.target_site_id_l] = target_l
        self.data.site_xpos[self.target_site_id_r] = target_r
        self.model.site_pos[self.target_site_id_r] = target_r

        rot_l_matrix_flat = rot_l.as_matrix().flatten()
        rot_r_matrix_flat = rot_r.as_matrix().flatten()

        self.data.site_xmat[self.target_site_id_l] = rot_l_matrix_flat
        self.data.site_xmat[self.target_site_id_r] = rot_r_matrix_flat

    def get_qpos(self):
        self.l_qpos = self.data.qpos[self.left_relevant_qpos_indices]
        self.r_qpos = self.data.qpos[self.right_relevant_qpos_indices]

        qpos = np.concatenate((self.l_qpos, [self.left_gripper_pos], self.r_qpos, [self.right_gripper_pos]), axis=0)
        return qpos
        
    def get_qvel(self):
        left_gripper_vel = 1 if self.action[6] > 0 else -1 if self.action[6] < 0 else 0
        right_gripper_vel = 1 if self.action[13] > 0 else -1 if self.action[13] < 0 else 0

        self.l_qvel = self.data.qvel[self.left_relevant_qvel_indices]
        self.r_qvel = self.data.qvel[self.right_relevant_qvel_indices]

        qvel = np.concatenate((self.l_qvel, [left_gripper_vel], self.r_qvel, [right_gripper_vel]), axis=0)
        return qvel
        
    def get_obs(self, cam_name):
        renderer, cam_id = self.camera_renderers[cam_name]
        renderer.update_scene(self.data, cam_id)
        img = renderer.render()
        return img
    
    def get_image(self, ts, camera_names):
        #TODO: modify to work with current format
        curr_images = []
        for cam_name in self.camera_names:
            curr_image = rearrange(self.get_obs(cam_name), 'h w c -> c h w')
            curr_images.append(curr_image)

        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
        return curr_image
    
    #get next action
    def select_action(self, policy, stats):
        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        
        curr_image = self.get_image(self, self.camera_names)
        qpos_numpy = np.array(self.get_qpos())

        # print(f"qpos: {qpos}")

        qpos = qpos_numpy
        # qpos = pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().unsqueeze(0)

        actions = policy(qpos, curr_image)

        return actions
    
    #pass action through sim
    def forward_actions(self, action):
        self.action = action

        # translation
        self.target_r[0] += self.action[7]
        self.target_r[1] += self.action[8]
        self.target_r[2] += self.action[9]
        self.target_r = np.clip(self.target_r, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])

        self.target_l[0] += self.action[0]
        self.target_l[1] += self.action[1]
        self.target_l[2] += self.action[2]
        self.target_l = np.clip(self.target_l, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])

        print(f"target l 2 action: {self.action[2]}, target l 2: {self.target_l[2]}")

        #rotation
        self.update_rotation('x', self.action[3], 'left')
        self.update_rotation('y', self.action[4], 'left')
        self.update_rotation('z', self.action[5], 'left')
        self.update_rotation('x', self.action[10], 'right')
        self.update_rotation('y', self.action[11], 'right')
        self.update_rotation('z', self.action[12], 'right')

        #gripper
        self.left_gripper_pos = self.action[6]
        self.right_gripper_pos = self.action[13]

        self.left_gripper_pos = np.clip(self.left_gripper_pos, 0.02, 0.037)
        self.right_gripper_pos = np.clip(self.right_gripper_pos, 0.02, 0.037)

        self.targets_updated = True

        print(f"forwarded actions: {self.action}")

def main(args):
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    
    task_config = SIM_TASK_CONFIGS[task_name]
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_last.ckpt']
        # ckpt_names = [f'policy_best.ckpt']
        # ckpt_names = [f'policy_epoch_3900_seed_4.ckpt']
        for ckpt_name in ckpt_names:
            eval_bc(config, ckpt_name, save_episode=True)
        exit()

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    print(loading_status)
    
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries'] # num queries = chunk size = default 100 (for README command)

    sim = Simulator()
    model = sim.model
    data = sim.data

    l_ee_task = mink.FrameTask(
            frame_name="aloha_scene/left_gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

    r_ee_task = mink.FrameTask(
            frame_name="aloha_scene/right_gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=[],
        minimum_distance_from_collisions=0.1,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.VelocityLimit(model, sim.velocity_limits),
        collision_avoidance_limit,
    ]

    solver = "osqp"
    max_iters = 20

    num_loop_iters = 0
    current_timestep = 0

    with torch.inference_mode():
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=True, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            sim.add_target_sites()
            mujoco.mj_forward(model, data)

            l_target_pose = mink.SE3.from_rotation_and_translation(sim.rot_l, sim.target_l)
            r_target_pose = mink.SE3.from_rotation_and_translation(sim.rot_r, sim.target_r)

            l_ee_task.set_target(l_target_pose)
            r_ee_task.set_target(r_target_pose)

            sim_rate = RateLimiter(frequency=200.0) 

            next_actions = sim.select_action(policy, stats)
            raw_action = next_actions.squeeze(0).cpu().numpy()
            actions = post_process(raw_action)
            for i, action in enumerate(actions):
                sim.action_buffers[current_timestep + i].append(action)

            m = 0.1 # from paper: this governs the speed of incorporating new observations, and smaller m means faster incorporation 
            w_i = lambda i: np.exp(m * i)

            current_action_buffer = sim.action_buffers[current_timestep]
            action = np.sum([w_i(i) * a for i, a in enumerate(current_action_buffer)], axis=0) / np.sum([w_i(i) for i in range(len(current_action_buffer))])
            sim.forward_actions(action)
            sim.control_gripper(sim.left_gripper_pos, sim.right_gripper_pos)

            current_timestep += 1
            
            while viewer.is_running():
                
                num_loop_iters += 1

                if (num_loop_iters % 40 == 0):
                    next_actions = sim.select_action(policy, stats)
                    raw_action = next_actions.squeeze(0).cpu().numpy()
                    actions = post_process(raw_action)

                    for i, action in enumerate(actions):
                        sim.action_buffers[current_timestep + i].append(action)

                    current_action_buffer = sim.action_buffers[current_timestep]

                    action = np.sum([w_i(i) * a for i, a in enumerate(current_action_buffer)], axis=0) / np.sum([w_i(i) for i in range(len(current_action_buffer))])
                    
                    # action *= 40

                    print(f"current action: {action}")

                    sim.forward_actions(action)

                    sim.control_gripper(sim.left_gripper_pos, sim.right_gripper_pos)

                    current_timestep += 1
                
                sim.forward_actions(action)

                if sim.targets_updated:
                    l_target_pose = mink.SE3.from_rotation_and_translation(sim.rot_l, sim.target_l)
                    l_ee_task.set_target(l_target_pose)
                    r_target_pose = mink.SE3.from_rotation_and_translation(sim.rot_r, sim.target_r)
                    r_ee_task.set_target(r_target_pose)

                    sim.update_target_sites(sim.target_l, sim.target_r, sim.rot_l, sim.rot_r)
                    sim.targets_updated = False

                for _ in range(max_iters):
                    left_vel = mink.solve_ik(
                        sim.left_configuration,
                        [l_ee_task],
                        sim_rate.dt,
                        solver,
                        limits=limits,
                        damping=1e-1,
                    )

                    right_vel = mink.solve_ik(
                        sim.right_configuration,
                        [r_ee_task],
                        sim_rate.dt,
                        solver,
                        limits=limits,
                        damping=1e-1,
                    )

                    sim.left_configuration.integrate_inplace(left_vel, sim_rate.dt)
                    sim.right_configuration.integrate_inplace(right_vel, sim_rate.dt)

                    sim.data.qpos[sim.left_relevant_qpos_indices] = sim.left_configuration.q
                    sim.data.qpos[sim.right_relevant_qpos_indices] = sim.right_configuration.q

                    sim.data.qvel[sim.left_relevant_qvel_indices] = sim.left_configuration.dq
                    sim.data.qvel[sim.right_relevant_qvel_indices] = sim.right_configuration.dq

                    sim.data.ctrl[sim.left_actuator_ids] = sim.left_configuration.q
                    sim.data.ctrl[sim.right_actuator_ids] = sim.right_configuration.q

                    sim.control_gripper(sim.left_gripper_pos, sim.right_gripper_pos)

                    mujoco.mj_step(model, data)

                    viewer.sync()
                    sim_rate.sleep()

                if sim.num_timesteps == 200: # rate of data collection per loop iteration (see teleop_aloha.py data_recording_interval) * length of one data episode
                    sim_rate.sleep()
                    print(f"num loop iters: {num_loop_iters}")
                    print(f"num timesteps: {sim.num_timesteps}")
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))