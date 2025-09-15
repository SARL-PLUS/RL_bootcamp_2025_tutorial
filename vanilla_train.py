from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.utils.utils import seed_erverything




if __name__ == "__main__":
    
    env_id = "Ant"
    env_version = "v5"
    task_name = "train_" + env_id + "_default"
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # global settings
    seed = 1234
    
    # log_dir is the main storage for checkpoints, configs and tensorboard records
    log_dir = Path().cwd().joinpath("logs")
    
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_dir.joinpath("runs", task_name, run_id)
    
    # create directory where the models snapshots are stored
    checkpoint_path = run_dir.joinpath("checkpoints")
    checkpoint_path.mkdir(parents=True, exist_ok=False)
        
    # create directory for the tensorboard record
    tensorboard_path = run_dir.joinpath("tensorboard")
    tensorboard_path.mkdir(parents=True, exist_ok=False)
    
    # seed all random generators used for reproducability
    seed_erverything(seed=1234)
    
    
    # instantiate the training enviornments
    train_env = make_vec_env(
        env_id="Ant-v5",
        n_envs=4,
        vec_env_cls=DummyVecEnv,
        seed=seed, 
    )
    
    
    # instantiate validation environments
    eval_env = make_vec_env(
        env_id="Ant-v5",
        n_envs=4,
        vec_env_cls=DummyVecEnv,
        seed=seed + 10, 
    )
    
    
    # instantiate the agent
    agent = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.98,
        ent_coef=0.0,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        device="cpu",
        tensorboard_log=tensorboard_path.as_posix()
    )
    
    
    # instantiate callbacks
    callbacks = [
        EvalCallback(
            eval_env=eval_env,
            callback_on_new_best=None,   # alternatively one can pass e.g. StopTrainingOnRewardThreshold instance for early stopping
            best_model_save_path=checkpoint_path.joinpath("best_model").as_posix(),
            log_path=checkpoint_path.joinpath("logs").as_posix(),
            n_eval_episodes=5,
            eval_freq=10000,
            render=False,
            verbose=1,
        )
    ]
    
    # train the agent
    agent.learn(total_timesteps=1000000, callback=callbacks, progress_bar=True)
    
 
    print("training finished in %d timesteps!"%agent.num_timesteps)
    
    