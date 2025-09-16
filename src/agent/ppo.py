
import warnings
import torch as th
from stable_baselines3 import PPO as sb3_PPO

class PPO(sb3_PPO):
    def _maybe_recommend_cpu(self, mlp_class_name = "ActorCriticPolicy"):
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        # bugfix: our custom policy constructor can also be a functools.partial instance which has no __name__ member
        policy_class_name = self.policy.__class__.__name__  # self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )