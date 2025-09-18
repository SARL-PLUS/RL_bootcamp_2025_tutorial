# src/tools/render_policy.py
import os
from pathlib import Path
from typing import Any, Dict, Union, Callable
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import PPO  # or SAC/TD3/etc., swap as needed
from gymnasium import Env

from src.utils.env_utils import make_single_vec_env


def render_policy_to_mp4_from_paths(
    model_class,
    model_path: Union[str, Path],
    env_id: Union[str, Callable[..., Env]],
    env_kwargs: Dict[str, Any],
    out_dir: Union[str, Path] = "videos",
    video_length: int = 1000,
    deterministic: bool = True,
    # vec_env_kwargs: Dict[str, Any] = None,
) -> Path:
    out_dir = Path(out_dir)
    _prev_files = set([str(item) for item in out_dir.glob('*.mp4')])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model_class.load(str(model_path), device="cpu")

    env = make_single_vec_env(
        env_id=env_id,
        env_kwargs=env_kwargs,
        # vec_env_cls=DummyVecEnv,
        # vec_env_kwargs=vec_env_kwargs or {},
    )

    name = f"{Path(model_path).stem}"

    render_policy_to_mp4(model, env, name, out_dir=out_dir, video_length=video_length, deterministic=deterministic)

    # The exact filename is created by VecVideoRecorder; return the folder
    _new_files =  set([str(item) for item in out_dir.glob('*.mp4')]) - _prev_files
    return list(_new_files - _prev_files)


def render_policy_to_mp4(agent, env, name, out_dir, video_length, deterministic=True):
    rec_env = VecVideoRecorder(
        env,
        video_folder=str(out_dir),
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=name,
    )

    obs = rec_env.reset()
    for _ in range(video_length):
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, _, _, _ = rec_env.step(action)

    rec_env.close()
    env.close()
