from typing import Any, Dict, Optional, Union, Callable
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import inspect
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import gymnasium as gym


def make_single_vec_env(
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
                # This sets the attribute so SB3's recorder checks pass.
                try:
                    env.render_mode = "rgb_array"
                except Exception as e:
                    print(e)
                    pass
            return env

    return DummyVecEnv([thunk])
