from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
import inspect

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import gymnasium as gym

# from src.utils.env_util import make_vec_env  # your helper


def _make_single_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """Create a 1-env DummyVecEnv with rgb_array rendering for both string IDs and factories."""
    env_kwargs = env_kwargs or {}

    if isinstance(env_id, str):
        def thunk():
            return gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    else:
        # env_id is a callable factory; pass render_mode if it accepts it
        sig = None
        try:
            sig = inspect.signature(env_id)
        except (ValueError, TypeError):
            pass

        def thunk():
            if sig and "render_mode" in sig.parameters:
                return env_id(render_mode="rgb_array", **env_kwargs)
            # best-effort fallback
            env = env_id(**env_kwargs)
            if getattr(env, "render_mode", None) != "rgb_array":
                # This sets the attribute so SB3's recorder check passes.
                # Many Gymnasium envs also need it set at init; if your factory ignores it,
                # prefer wiring a string ID instead so we can call gym.make(..., render_mode=...)
                try:
                    env.render_mode = "rgb_array"
                except Exception:
                    pass
            return env

    return DummyVecEnv([thunk])


class PeriodicVideoCallback(BaseCallback):
    """
    Record an MP4 of the current policy every `video_freq` env steps.
    Creates a fresh 1-env VecEnv for recording so training speed is unaffected.
    """
    def __init__(
        self,
        env_id: Union[str, Callable[..., Env]],
        env_kwargs: Optional[Dict[str, Any]] = None,
        # eval_env,
        video_folder: str = "videos",
        video_freq: int = 100_000,
        video_length: int = 1000,
        deterministic: bool = True,
        vec_env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        # self.eval_env = eval_env
        self.video_folder = Path(video_folder)
        self.video_freq = int(video_freq)
        self.video_length = int(video_length)
        self.deterministic = deterministic
        self.vec_env_kwargs = vec_env_kwargs or {}

        self.video_folder.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Trigger exactly at multiples of video_freq (skip t=0)
        if self.num_timesteps > 0 and (self.num_timesteps % self.video_freq == 0):
            self._record_once()
        return True

    def _record_once(self) -> None:
        prefix = f"step_{self.num_timesteps:010d}"

        eval_env = _make_single_vec_env(self.env_id, self.env_kwargs)

        # --- safety check before recording
        try:
            # reach the underlying env (index 0) for a definitive value
            rm = getattr(eval_env.envs[0], "render_mode", None)
        except Exception:
            rm = getattr(eval_env, "render_mode", None)

        if rm != "rgb_array":
            raise RuntimeError(
                f"Video callback: expected render_mode='rgb_array' but got {rm}. "
                f"If you’re passing a factory, ensure it accepts render_mode or pass a string env id."
            )

        rec_env = VecVideoRecorder(
            eval_env,
            video_folder=str(self.video_folder),
            record_video_trigger=lambda step: step == 0,
            video_length=self.video_length,
            name_prefix=prefix,
        )

        obs = rec_env.reset()
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, _, _ = rec_env.step(action)

        rec_env.close()
        eval_env.close()
