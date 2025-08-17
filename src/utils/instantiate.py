from typing import List, Callable
import hydra
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy




def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[BaseCallback]:
    """ Instantiates callbacks from config. """
    callbacks: List[BaseCallback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return list(callbacks)
