from typing import Callable, Dict, List, Optional, Type, Union, Any, Sequence
import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import _patch_env

def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: Optional[int] = None,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    Accepts either a shared env_kwargs dict or a list of dicts for per-env configs.
    """
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    
    if not isinstance(env_kwargs, Sequence):
        assert n_envs != None
        env_kwargs = [env_kwargs for _ in range(n_envs)]
    else:
        assert n_envs == None or n_envs == len(env_kwargs)
        n_envs = len(env_kwargs)
        
    env_kwargs = [nth_kwargs if nth_kwargs != None else {} for nth_kwargs in env_kwargs]

    def make_env(rank: int, kwargs: Dict[str, Any]) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            if seed is not None and not "seed" in kwargs.keys():
                kwargs["seed"] = seed + rank

            if isinstance(env_id, str):
                full_kwargs = {"render_mode": "rgb_array"}
                full_kwargs.update(kwargs)
                try:
                    env = gym.make(env_id, **full_kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **kwargs)
            else:
                env = env_id(**kwargs)
                env = _patch_env(env)

            if seed is not None:
                env.action_space.seed(seed + rank)

            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)

            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)

            return env
        return _init

    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv

    env_fns = [make_env(i + start_index, dict(**env_kwargs[i])) for i in range(n_envs)]
    vec_env = vec_env_cls(env_fns, **vec_env_kwargs)
    vec_env.seed(seed)
    return vec_env