from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from gymnasium.envs.mujoco.ant_v5 import AntEnv, DEFAULT_CAMERA_CONFIG
from pathlib import Path
import numpy as np


class CrippledAnt(AntEnv):
    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: int | str = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        injury: str = "little",
        **kwargs,
    ):
        
        assert injury=="little" or injury=="medium" or injury=="hard" ,\
            "supported values for severity are 'little', 'medium' or 'hard' only!"

        self._injury = injury
        
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self._healthy_state = np.ones(4)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size# + len(self._healthy_state)
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
        }

    def reset(self, *args, **kwargs):
        
        self._healthy_state = np.ones(4)
        
        if self._injury == "little":
            idx = np.random.choice(np.arange(4), 1, replace=False)
        elif self._injury == "medium":
            idx = np.random.choice(np.arange(4), 2, replace=False)
        elif self._injury == "hard":
            idx = np.random.choice(np.arange(4), 3, replace=False)
            
        self._healthy_state[idx] = 0
        
        return super().reset(*args, **kwargs)
    
    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        scales = np.zeros_like(action)
        scales[::2] = self._healthy_state
        scales[1::2] = self._healthy_state

        self.do_simulation(action*scales, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        health = self._healthy_state.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            # return np.concatenate((position, velocity, contact_force, health))
            return np.concatenate((position, velocity, contact_force))
        else:
            # return np.concatenate((position, velocity, health))
            return np.concatenate((position, velocity))
