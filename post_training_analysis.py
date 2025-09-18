from argparse import ArgumentParser
from pathlib import Path
import hydra
import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from src.utils.postprocessing import get_tensorboard_record, get_synced_traces, resolve_tags
from src.tools.render_policy import render_policy_to_mp4, render_policy_to_mp4_from_paths
from src.utils.utils import plot_multiple_axes

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cripple", action="store_true")

    args = parser.parse_args()    
    
    run_path = Path().cwd().joinpath(args.run)
    cfg = OmegaConf.load(run_path.joinpath(".hydra", "config.yaml"))

    snapshot_path = run_path.joinpath("checkpoints", "best_model", "best_model.zip")

    if args.cripple:
        gym.register(id="CrippledAnt-v5", entry_point="src.envs.ant:CrippledAnt")
        _temp_cfg = cfg.env.eval_env
        _temp_cfg.env_id['id'] = env_id = "CrippledAnt-v5"
        env = hydra.utils.instantiate(_temp_cfg)
        # env_id = cfg.env.eval_env.env_id['id']
    else:
        env = hydra.utils.instantiate(cfg.env.eval_env)
        env_id = cfg.env.eval_env.env_id['id']

    agent_class = hydra.utils.get_class(cfg.agent._target_)
    agent = agent_class.load(snapshot_path, device="cpu")

    n_eval_eps = cfg.callbacks.eval_callback.n_eval_episodes
    mean_rewards, std_rewards, *_ = evaluate_policy(
        model=agent,
        env=env,
        n_eval_episodes=n_eval_eps,
        deterministic=True,
    )
    print(f"Mean reward over {n_eval_eps} episodes: {mean_rewards:.2f}±{std_rewards:.2f}")

    if args.render:
        from stable_baselines3 import PPO
        from src.tools.render_policy import render_policy_to_mp4

        name = snapshot_path.stem
        save_name = name + '.mp4'
        save_path = snapshot_path.parent / save_name

        # model_class = PPO
        # model_path = 'logs/runs/train_save_best/2025-08-22_18-55-24/checkpoints/best_model/best_model.zip'
        # env_id = 'CrippledAnt-v5'
        env_kwargs = {}
        out_dir = 'renders'
        video_length = 1000
        deterministic = True

        render_policy_to_mp4_from_paths(
            model_class=agent_class,
            model_path=snapshot_path,
            env_id=env_id,
            env_kwargs=env_kwargs,
            video_length=video_length,
            deterministic=deterministic,
            out_dir="videos" if not args.cripple else "videos-cripple"
        )

    if args.plot:
        # analyze tensorboard record
        tb_record = get_tensorboard_record(run_path)
        rollout_tags = resolve_tags(obj=tb_record, prefix="rollout/")
        rollout_data = get_synced_traces(ea=tb_record, tags=rollout_tags)

        train_tags = resolve_tags(obj=tb_record, prefix="train/")
        train_data = get_synced_traces(ea=tb_record, tags=train_tags)

        eval_tags = resolve_tags(obj=tb_record, prefix="eval/")
        eval_data = get_synced_traces(ea=tb_record, tags=eval_tags)

        df_list = [rollout_data, train_data, eval_data]
        for i, df in enumerate(df_list):
            df_list[i] = df.set_index('steps')
            # print(df.columns)
            # break
        #
        plot_dict = {
            0: {
                'tags': ['rollout/ep_len_mean', 'eval/mean_ep_length'],
                'ylabel': 'Episode length',
            },
            1: {
                'tags': ['rollout/ep_rew_mean', 'eval/mean_reward'],
                'ylabel': 'Episode reward',
            },
            2: {
                'tags': ['train/loss', 'train/value_loss'],
                'ylabel': 'Loss',
            },
            3: {
                'tags': ['train/policy_gradient_loss', 'train/entropy_loss'],
                'ylabel': 'Loss',
            },
            4: {
                'tags': ['train/explained_variance', 'train/clip_fraction'],
                'ylabel': 'Train metrics',
            }
        }
        plot_multiple_axes(df_list, plot_dict)
    # pass

