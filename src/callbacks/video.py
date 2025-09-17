from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import gymnasium as gym
import os, tempfile, shutil
from multiprocessing import get_context

from src.utils.env_utils import make_single_vec_env


def _render_best_worker(model_cls, model_path, env_id, env_kwargs, video_folder, n_videos, video_length, deterministic, rm_on_complete=True):
    env_kwargs = env_kwargs or {}
    video_folder = Path(video_folder); video_folder.mkdir(parents=True, exist_ok=True)

    model = model_cls.load(model_path, device="cpu")

    for i in range(n_videos):
        def thunk(): return gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        eval_env = DummyVecEnv([thunk])
        rec_env = VecVideoRecorder(
            eval_env,
            video_folder=str(video_folder),
            record_video_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=f"async_{os.path.basename(model_path)}_{i+1}",
        )
        obs = rec_env.reset()
        for _ in range(video_length):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, _, _ = rec_env.step(action)
        rec_env.close(); eval_env.close()

    # cleanup the temp model
    if rm_on_complete:
        try: os.remove(model_path); shutil.rmtree(os.path.dirname(model_path), ignore_errors=True)
        except Exception: pass


class EvalSaveVideosCallback(EvalCallback):
    """
    On every new best mean reward found by EvalCallback, save N MP4s
    using non-deterministic policy actions (deterministic=False).
    """
    def __init__(
        self,
        # video params
        video_env_id: Union[str, Callable[..., gym.Env]],
        video_env_kwargs: Optional[Dict[str, Any]] = None,
        video_folder: str = "videos",
        n_videos: int = 5,
        video_length: int = 1000,
        video_deterministic: bool = False,  # <-- should be non-deterministic if n_videos > 1
        record_after_timestep: int = 102_400,
        # pass-through to EvalCallback:
        **eval_kwargs,
    ):
        super().__init__(**eval_kwargs)
        self.video_env_id = video_env_id
        self.video_env_kwargs = video_env_kwargs or {}
        self.video_folder = Path(video_folder)
        self.n_videos = int(n_videos)
        self.video_length = int(video_length)
        self.video_deterministic = bool(video_deterministic)

        self.video_folder.mkdir(parents=True, exist_ok=True)
        self._last_best = float("-inf")

        self.record_after_timestep = record_after_timestep

    def _on_step(self) -> bool:
        prev_best = self.best_mean_reward
        cont = super()._on_step()
        # If EvalCallback just improved best_mean_reward, dump videos
        if self.best_mean_reward > prev_best and self.num_timesteps > self.record_after_timestep:
            # self._save_best_videos()
            self._save_best_videos_async()
        return cont

    def _save_best_videos(self) -> None:
        # Fail-soft if MoviePy/ffmpeg not installed
        try:
            import moviepy  # noqa: F401
        except Exception as e:
            print(f"[EvalVideo] Skipping videos: MoviePy missing ({e}). "
                  f"Install with `pip install moviepy==2.2.1 imageio-ffmpeg==0.6.0`.")
            return

        for i in range(self.n_videos):
            prefix = f"best_step_{self.num_timesteps:010d}_{i+1}"
            eval_env = make_single_vec_env(self.video_env_id, self.video_env_kwargs)
            rec_env = VecVideoRecorder(
                eval_env,
                video_folder=str(self.video_folder),
                record_video_trigger=lambda step: step == 0,
                video_length=self.video_length,
                name_prefix=prefix,
            )
            obs = rec_env.reset()
            for _ in range(self.video_length):
                # non-deterministic sampling from the policy
                action, _ = self.model.predict(obs, deterministic=self.video_deterministic)
                obs, _, _, _ = rec_env.step(action)
            rec_env.close()
            eval_env.close()

    def _save_best_videos_async(self) -> None:
        try:
            import moviepy  # noqa: F401
        except Exception as e:
            print(f"[EvalVideo] Skipping videos: MoviePy missing ({e}). "
                  f"Install with `pip install -U moviepy imageio-ffmpeg`.")
            return

        # save current policy to a temp path for the worker
        tmpdir = tempfile.mkdtemp(prefix="best_for_video_")
        model_path = os.path.join(tmpdir, f"best_at_{self.num_timesteps}.zip")
        self.model.save(model_path)

        # p = Process(
        ctx = get_context("spawn")
        p = ctx.Process(
            target=_render_best_worker,
            args=(
                self.model.__class__, model_path,
                self.video_env_id, self.video_env_kwargs,
                str(self.video_folder), self.n_videos, self.video_length, self.video_deterministic,
            ),
            daemon=True,
        )
        p.start()  # return immediately; training continues


class PeriodicVideoCallback(BaseCallback):
    """
    Record an MP4 of the current policy every `video_freq` env steps.
    Creates a fresh 1-env VecEnv for recording so training speed is unaffected.
    """
    def __init__(
        self,
        env_id: Union[str, Callable[..., gym.Env]],
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

        eval_env = make_single_vec_env(self.env_id, self.env_kwargs)

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
