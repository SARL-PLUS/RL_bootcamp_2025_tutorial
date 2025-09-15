# src/tools/render_policy.py
import os
from pathlib import Path
from typing import Any, Dict, Union, Callable
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import PPO  # or SAC/TD3/etc., swap as needed
from gymnasium import Env

from src.utils.env_util import make_vec_env


def render_policy_to_mp4(
    model_class,
    model_path: Union[str, Path],
    env_id: Union[str, Callable[..., Env]],
    env_kwargs: Dict[str, Any],
    out_dir: Union[str, Path] = "videos",
    video_length: int = 1000,
    deterministic: bool = True,
    vec_env_kwargs: Dict[str, Any] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model_class.load(str(model_path), device="cpu")

    env = make_vec_env(
        env_id=env_id,
        n_envs=1,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
        vec_env_kwargs=vec_env_kwargs or {},
    )

    name = f"{Path(model_path).stem}"
    rec_env = VecVideoRecorder(
        env,
        video_folder=str(out_dir),
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=name,
    )

    obs = rec_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, _, _ = rec_env.step(action)

    rec_env.close()
    env.close()
    # The exact filename is created by VecVideoRecorder; return the folder
    return out_dir
