
from gymnasium.envs.mujoco.ant_v5 import AntEnv, DEFAULT_CAMERA_CONFIG
from pathlib import Path


class RealAnt(AntEnv):
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
        **kwargs,
    ):
        
        self._xml_file = Path()
        
        super().__init__(
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
        
        
        pass
        
    
if __name__ == "__main__":
    from gymnasium import make
    from gymnasium.envs.registration import register
    register(
        id="RealAnt-V5",
        entry_point=RealAnt,
        order_enforce=True
    )
    
    env = make(id="RealAnt-V5")
    
    pass
        
        
        